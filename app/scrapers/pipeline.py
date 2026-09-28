"""Scrape orchestration: browser -> parsed payloads -> database rows.

Step 1 (cleaning, spam/language filtering, de-duplication) runs at ingestion time
so the database never holds two copies of the same review text for one phone. The
raw body is still stored verbatim alongside the cleaned version, which keeps the
dataset auditable.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.database import session_scope
from app.core.logging import get_logger
from app.models.entities import PriceObservation, Review, Smartphone
from app.nlp.preprocess import preprocess_review
from app.scrapers.amazon import AmazonScraper
from app.scrapers.base import (
    BotCheckError,
    BrowserSession,
    ScrapedPrice,
    ScrapedProduct,
    ScrapedReview,
    ScraperError,
    human_delay,
)

logger = get_logger(__name__)

ProgressCallback = Callable[[int, int, str], None]


def upsert_product(db: Session, product: ScrapedProduct) -> Smartphone:
    """Insert or update a phone, keyed on (source, source_product_id)."""
    phone = db.scalar(
        select(Smartphone).where(
            Smartphone.source == product.source,
            Smartphone.source_product_id == product.source_product_id,
        )
    )
    now = datetime.now(timezone.utc)

    if phone is None:
        phone = Smartphone(
            source=product.source,
            source_product_id=product.source_product_id,
            first_seen_at=now,
        )
        db.add(phone)

    phone.product_url = product.product_url or phone.product_url
    phone.raw_title = product.raw_title or phone.raw_title
    phone.brand = product.brand or phone.brand
    phone.model = product.model or phone.model
    phone.canonical_name = product.canonical_name or phone.canonical_name
    phone.image_url = product.image_url or phone.image_url
    if product.site_rating is not None:
        phone.site_rating = product.site_rating
    if product.site_rating_count is not None:
        phone.site_rating_count = product.site_rating_count
    if product.specs_raw:
        phone.specs_raw = product.specs_raw
    if product.feature_bullets:
        phone.feature_bullets = product.feature_bullets

    for field_name, value in (product.normalized_specs or {}).items():
        if value is not None and hasattr(phone, field_name):
            setattr(phone, field_name, value)

    phone.last_seen_at = now
    db.flush()
    return phone


def store_price(db: Session, phone: Smartphone, price: ScrapedPrice | None) -> bool:
    """Append a price observation, skipping unchanged consecutive readings."""
    if price is None or price.price is None:
        return False

    latest = db.scalar(
        select(PriceObservation)
        .where(PriceObservation.smartphone_id == phone.id)
        .order_by(PriceObservation.observed_at.desc())
        .limit(1)
    )
    if (
        latest is not None
        and latest.price == price.price
        and latest.currency == price.currency
        and latest.availability == price.availability
    ):
        return False

    db.add(
        PriceObservation(
            smartphone_id=phone.id,
            price=price.price,
            list_price=price.list_price,
            currency=price.currency,
            discount_pct=price.discount_pct,
            availability=price.availability,
            source_url=phone.product_url,
        )
    )
    db.flush()
    return True


def store_reviews(
    db: Session,
    phone: Smartphone,
    reviews: list[ScrapedReview],
    settings: Settings | None = None,
) -> dict[str, int]:
    """Persist reviews with Step 1 preprocessing and hash-based de-duplication."""
    settings = settings or get_settings()

    existing_hashes = set(
        db.scalars(select(Review.content_hash).where(Review.smartphone_id == phone.id)).all()
    )

    inserted = duplicates = excluded = 0

    for scraped in reviews:
        cleaned = preprocess_review(
            scraped.body,
            title=scraped.title,
            min_chars=settings.min_review_chars,
            language_filter=settings.language_filter or None,
        )

        if cleaned.content_hash in existing_hashes:
            duplicates += 1
            continue
        existing_hashes.add(cleaned.content_hash)

        if cleaned.excluded_reason:
            excluded += 1

        db.add(
            Review(
                smartphone_id=phone.id,
                source=phone.source,
                source_review_id=scraped.source_review_id,
                source_url=scraped.source_url,
                title=scraped.title,
                body=scraped.body,
                cleaned_body=cleaned.cleaned_text or None,
                content_hash=cleaned.content_hash,
                rating=scraped.rating,
                review_date=scraped.review_date,
                reviewer_name=scraped.reviewer_name,
                verified_purchase=scraped.verified_purchase,
                helpful_votes=scraped.helpful_votes,
                variant=scraped.variant,
                country=scraped.country,
                language=cleaned.language,
                is_spam=cleaned.is_spam,
                excluded_reason=cleaned.excluded_reason,
                pipeline_version=settings.pipeline_version,
            )
        )
        inserted += 1

    db.flush()
    return {"inserted": inserted, "duplicates": duplicates, "excluded": excluded}


async def scrape_to_db(
    *,
    queries: list[str] | None = None,
    product_ids: list[str] | None = None,
    max_phones_per_query: int = 5,
    max_reviews_per_phone: int = 100,
    max_review_pages: int = 10,
    review_sort: str = "recent",
    headless: bool | None = None,
    settings: Settings | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run a full scrape and return a summary dictionary."""
    settings = settings or get_settings()
    queries = [q for q in (queries or []) if q.strip()]
    product_ids = [p for p in (product_ids or []) if p.strip()]

    if not queries and not product_ids:
        raise ValueError("Provide at least one search query or product id.")

    summary: dict[str, Any] = {
        "phone_ids": [],
        "phones_scraped": 0,
        "reviews_inserted": 0,
        "reviews_duplicate": 0,
        "reviews_excluded": 0,
        "prices_recorded": 0,
        "errors": [],
    }

    def report(done: int, total: int, message: str) -> None:
        if progress is not None:
            progress(done, total, message)

    report(0, 1, "Launching browser")

    async with BrowserSession(settings, headless=headless) as session:
        scraper = AmazonScraper(session, settings)

        report(0, 1, "Resolving targets")
        try:
            asins = await scraper.resolve_targets(queries, product_ids, max_phones_per_query)
        except BotCheckError as exc:
            summary["errors"].append(f"bot_check: {exc}")
            return summary

        if not asins:
            summary["errors"].append("No phone listings matched the given queries.")
            return summary

        total = len(asins)
        logger.info("Scraping %s phone(s): %s", total, ", ".join(asins))

        for index, asin in enumerate(asins, start=1):
            report(index - 1, total, f"Scraping {asin} ({index}/{total})")
            try:
                product = await scraper.fetch_product(asin)
                await human_delay(settings)

                paginated = await scraper.fetch_reviews(
                    asin,
                    max_reviews=max_reviews_per_phone,
                    max_pages=max_review_pages,
                    sort=review_sort,
                )

                # Merge product-page reviews with the paginated listing.
                merged: list[ScrapedReview] = list(paginated)
                known = {r.source_review_id or r.body[:120] for r in merged}
                for review in product.inline_reviews:
                    key = review.source_review_id or review.body[:120]
                    if key not in known:
                        known.add(key)
                        merged.append(review)

                with session_scope() as db:
                    phone = upsert_product(db, product)
                    phone_id = phone.id
                    if store_price(db, phone, product.price):
                        summary["prices_recorded"] += 1
                    counts = store_reviews(db, phone, merged, settings)

                summary["phone_ids"].append(phone_id)
                summary["phones_scraped"] += 1
                summary["reviews_inserted"] += counts["inserted"]
                summary["reviews_duplicate"] += counts["duplicates"]
                summary["reviews_excluded"] += counts["excluded"]

                logger.info(
                    "%s -> %s: +%s review(s), %s duplicate(s) skipped.",
                    asin, product.canonical_name, counts["inserted"], counts["duplicates"],
                )

                report(index, total, f"Stored {product.canonical_name or asin}")
                await human_delay(settings)

            except BotCheckError as exc:
                summary["errors"].append(f"{asin}: bot_check: {exc}")
                logger.error("Aborting scrape: %s", exc)
                break
            except (ScraperError, Exception) as exc:  # noqa: BLE001
                summary["errors"].append(f"{asin}: {type(exc).__name__}: {exc}")
                logger.exception("Failed to scrape %s", asin)
                continue

    report(1, 1, "Scrape finished")
    return summary


async def save_login_session(settings: Settings | None = None) -> None:
    """Open a visible browser so the user can sign in once; persists cookies."""
    settings = settings or get_settings()

    async with BrowserSession(settings, headless=False, use_storage_state=False) as session:
        page = await session.new_page()
        await page.goto(f"{settings.marketplace.rstrip('/')}/gp/sign-in.html")

        print("\n" + "=" * 72)
        print("A browser window is open. Sign in to your marketplace account there.")
        print("Complete any CAPTCHA / two-factor step until you reach the homepage.")
        print("Then return here and press Enter to save the session.")
        print("=" * 72 + "\n")

        # input() blocks the loop; run it off-thread so the browser stays responsive.
        import asyncio

        await asyncio.get_running_loop().run_in_executor(None, input, "Press Enter when signed in... ")

        await session.save_storage_state()
        await page.close()

    print(f"Session saved to {settings.storage_state_path}")
