"""Import a pre-built HF corpus folder into SQLite and optionally run ABSA.

Expects the folder produced by ``scripts/build_hf_corpus.py`` or the Colab notebook:

    phones.jsonl
    reviews.jsonl
    manifest.json
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.models.entities import Smartphone
from app.scrapers.base import ScrapedPrice, ScrapedProduct, ScrapedReview
from app.scrapers.pipeline import store_price, store_reviews, upsert_product

logger = get_logger(__name__)
ProgressCallback = Callable[[int, int, str], None]


def reset_corpus(db: Session) -> dict[str, int]:
    """Delete every phone and cascading related rows (demo, scrape, and HF)."""
    phones = list(db.scalars(select(Smartphone)).all())
    count = len(phones)
    for phone in phones:
        db.delete(phone)
    db.flush()
    logger.info("Reset corpus: deleted %s phone(s).", count)
    return {"phones_deleted": count}


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def import_corpus_folder(
    db: Session,
    folder: Path,
    *,
    settings: Settings | None = None,
    replace: bool = False,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    settings = settings or get_settings()
    folder = Path(folder)
    phones_path = folder / "phones.jsonl"
    reviews_path = folder / "reviews.jsonl"
    if not phones_path.is_file() or not reviews_path.is_file():
        raise FileNotFoundError(
            f"Expected phones.jsonl and reviews.jsonl in {folder}. "
            "Build them with scripts/build_hf_corpus.py or the Colab notebook."
        )

    def report(done: int, total: int, message: str) -> None:
        if progress is not None:
            progress(done, total, message)
        logger.info("[%s/%s] %s", done, total, message)

    summary: dict[str, Any] = {
        "folder": str(folder),
        "phones_upserted": 0,
        "prices_recorded": 0,
        "reviews_inserted": 0,
        "reviews_duplicate": 0,
        "reviews_excluded": 0,
        "phone_ids": [],
        "replaced": replace,
    }

    if replace:
        report(0, 1, "Clearing existing corpus…")
        reset_corpus(db)
        db.commit()

    phones: list[dict[str, Any]] = []
    with phones_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                phones.append(json.loads(line))

    asin_to_id: dict[str, int] = {}
    report(0, len(phones), f"Importing {len(phones)} phone(s)…")
    for index, row in enumerate(phones, start=1):
        product = ScrapedProduct(
            source=row.get("source") or "amazon_hf",
            source_product_id=row["source_product_id"],
            product_url=row.get("product_url"),
            raw_title=row.get("raw_title"),
            brand=row.get("brand"),
            model=row.get("model"),
            canonical_name=row.get("canonical_name"),
            image_url=row.get("image_url"),
            site_rating=row.get("site_rating"),
            site_rating_count=row.get("site_rating_count"),
            specs_raw=row.get("specs_raw") or {},
            feature_bullets=row.get("feature_bullets") or [],
            normalized_specs=row.get("normalized_specs") or {},
            price=(
                ScrapedPrice(price=row["price"], currency=row.get("currency") or "USD")
                if row.get("price") is not None
                else None
            ),
        )
        phone = upsert_product(db, product)
        if store_price(db, phone, product.price):
            summary["prices_recorded"] += 1
        asin_to_id[product.source_product_id] = phone.id
        summary["phones_upserted"] += 1
        summary["phone_ids"].append(phone.id)
        if index % 10 == 0:
            db.commit()
            report(index, len(phones), phone.display_name)

    db.commit()

    by_asin: dict[str, list[ScrapedReview]] = {}
    with reviews_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            asin = row["parent_asin"]
            if asin not in asin_to_id:
                continue
            scraped = ScrapedReview(
                body=row["body"],
                source_review_id=row.get("source_review_id"),
                source_url=f"https://www.amazon.com/dp/{asin}",
                title=row.get("title"),
                rating=row.get("rating"),
                review_date=_parse_date(row.get("review_date")),
                reviewer_name=row.get("reviewer_name"),
                verified_purchase=row.get("verified_purchase"),
                helpful_votes=row.get("helpful_votes"),
            )
            by_asin.setdefault(asin, []).append(scraped)

    report(0, len(by_asin), f"Importing reviews for {len(by_asin)} phone(s)…")
    for index, (asin, scraped_list) in enumerate(by_asin.items(), start=1):
        phone = db.get(Smartphone, asin_to_id[asin])
        if phone is None:
            continue
        counts = store_reviews(db, phone, scraped_list, settings=settings)
        summary["reviews_inserted"] += counts["inserted"]
        summary["reviews_duplicate"] += counts["duplicates"]
        summary["reviews_excluded"] += counts["excluded"]
        db.commit()
        report(index, len(by_asin), f"{phone.display_name}: +{counts['inserted']}")

    report(len(by_asin), len(by_asin), "Import finished")
    return summary
