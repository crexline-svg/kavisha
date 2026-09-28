"""Ingest a local scraped Amazon reviews CSV (``data/full_reviews.csv``).

Expected columns (one review row per line; product fields repeated per ASIN):

    asin, product_title, product_name, brand, price, currency, image_url,
    product_url, product_avg_rating, reviews_count, availability, seller_name,
    categories, description, scraped_at, review_number, reviewer,
    review_rating, review_title, review_comment, review_date, verified_purchase

Methodology mapping (Step 1 on insert):

    dedupe / empty / spam / HTML / normalise / **English-only** language filter

Non-phone listings (user manuals / guide books) are dropped before insert.
"""

from __future__ import annotations

import ast
import json
import re
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.models.entities import Smartphone
from app.scrapers.base import ScrapedPrice, ScrapedProduct, ScrapedReview
from app.scrapers.parsers import (
    KNOWN_BRANDS,
    extract_brand,
    extract_model,
    parse_number,
    parse_review_date,
)
from app.scrapers.pipeline import store_price, store_reviews, upsert_product
from app.services.corpus_import import reset_corpus
from sqlalchemy.orm import Session

logger = get_logger(__name__)

ProgressCallback = Callable[[int, int, str], None]

SOURCE = "amazon_scraped"
DEFAULT_CSV = Path("data") / "full_reviews.csv"

# Title patterns that are books/manuals, not smartphones.
_MANUAL_TITLE = re.compile(
    r"\b(?:"
    r"user\s+guide|user\s+manual|owners?\s+manual|beginners?\s+and\s+seniors|"
    r"step[- ]by[- ]step\s+manual|complete\s+manual|guidebook|how\s+to\s+use|"
    r"manual\s+for\s+beginners|for\s+beginners|unlock\s+features|"
    r"use\s+it\s+like\s+a\s+pro|\binstruction\s+manual\b|\bguidebook\b|"
    r"\bmanual\b"
    r")\b",
    re.IGNORECASE,
)

# Marketplace carriers / retailers often appear in the CSV "brand" column.
_CARRIER_OR_RETAIL_BRANDS = {
    "at&t",
    "at&t prepaid",
    "att",
    "att prepaid",
    "tmobile",
    "t-mobile",
    "t mobile",
    "verizon",
    "sprint",
    "tracfone",
    "cricket",
    "metro",
    "metro by t-mobile",
    "boost",
    "boost mobile",
    "amazon renewed",
    "amazon",
    "unlocked",
}

_KNOWN_BRAND_LOWER = {b.lower() for b in KNOWN_BRANDS} | {
    "apple",
    "moto",
    "amazon renewed",
}


def _is_manual_listing(title: str) -> bool:
    return bool(_MANUAL_TITLE.search(title or ""))


def _resolve_brand(raw_brand: str | None, title: str) -> str | None:
    brand = (raw_brand or "").strip() or None
    title_brand = extract_brand(title)

    if brand and brand.lower() in _KNOWN_BRAND_LOWER:
        if brand.lower() in {"moto"}:
            return "Motorola"
        if brand.lower() == "amazon renewed":
            return title_brand or brand
        return brand if brand[0].isupper() else brand.title()

    # Carrier / seller names (AT&T Prepaid, Tmobile, Paul Hollman, …) — use title.
    if brand and (
        brand.lower() in _CARRIER_OR_RETAIL_BRANDS or brand.lower() not in _KNOWN_BRAND_LOWER
    ):
        if title_brand:
            return title_brand

    return extract_brand(title, spec_brand=brand) or title_brand


def _parse_rating(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:out of|/)\s*5", text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    try:
        return float(text)
    except ValueError:
        return None


def _parse_verified(value: Any) -> bool | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "t", "yes", "y", "1", "verified"}:
        return True
    if text in {"false", "f", "no", "n", "0"}:
        return False
    return None


def _parse_categories(value: Any) -> list[str] | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except (SyntaxError, ValueError):
        pass
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except json.JSONDecodeError:
        pass
    return [text]


def _row_to_product(row: pd.Series) -> ScrapedProduct:
    asin = str(row["asin"]).strip().upper()
    title = str(row.get("product_title") or row.get("product_name") or "").strip()
    brand = _resolve_brand(
        str(row["brand"]).strip() if pd.notna(row.get("brand")) else None,
        title,
    )
    model = extract_model(title, brand or "", "")
    canonical = f"{brand} {model}".strip() if brand or model else title[:200]

    price_val = parse_number(str(row["price"])) if pd.notna(row.get("price")) else None
    currency = str(row["currency"]).strip() if pd.notna(row.get("currency")) else "USD"

    site_rating = None
    if pd.notna(row.get("product_avg_rating")):
        try:
            site_rating = float(row["product_avg_rating"])
        except (TypeError, ValueError):
            site_rating = None

    site_rating_count = None
    if pd.notna(row.get("reviews_count")):
        try:
            site_rating_count = int(float(row["reviews_count"]))
        except (TypeError, ValueError):
            site_rating_count = None

    url = (
        str(row["product_url"]).strip()
        if pd.notna(row.get("product_url"))
        else f"https://www.amazon.com/dp/{asin}"
    )
    image = str(row["image_url"]).strip() if pd.notna(row.get("image_url")) else None
    availability = str(row["availability"]).strip() if pd.notna(row.get("availability")) else None
    categories = _parse_categories(row.get("categories"))
    description = str(row["description"]).strip() if pd.notna(row.get("description")) else None
    seller = str(row["seller_name"]).strip() if pd.notna(row.get("seller_name")) else None

    specs_raw: dict[str, Any] = {"dataset": "full_reviews.csv"}
    if description:
        specs_raw["description"] = description[:2000]
    if seller:
        specs_raw["seller_name"] = seller
    if categories:
        specs_raw["categories"] = categories

    return ScrapedProduct(
        source=SOURCE,
        source_product_id=asin,
        product_url=url,
        raw_title=title,
        brand=brand,
        model=model or None,
        canonical_name=canonical,
        image_url=image,
        site_rating=site_rating,
        site_rating_count=site_rating_count,
        specs_raw=specs_raw,
        feature_bullets=[description] if description else [],
        price=ScrapedPrice(
            price=price_val,
            currency=currency,
            availability=availability,
        )
        if price_val is not None
        else None,
    )


def _row_to_review(row: pd.Series, product_url: str | None) -> ScrapedReview | None:
    body = str(row.get("review_comment") or "").strip()
    if not body:
        return None

    asin = str(row["asin"]).strip().upper()
    reviewer = str(row.get("reviewer") or "")[:64] or None
    title = str(row.get("review_title") or "").strip() or None
    rating = _parse_rating(row.get("review_rating"))

    review_date = None
    country = None
    if pd.notna(row.get("review_date")) and str(row.get("review_date")).strip():
        review_date, country = parse_review_date(str(row["review_date"]))

    stamp = str(row.get("review_number") or row.get("review_date") or "")
    source_review_id = f"{asin}:{stamp}:{reviewer}"[:120]

    return ScrapedReview(
        body=body,
        source_review_id=source_review_id,
        source_url=product_url,
        title=title,
        rating=rating,
        review_date=review_date,
        reviewer_name=reviewer,
        verified_purchase=_parse_verified(row.get("verified_purchase")),
        country=country,
    )


def ingest_full_reviews_csv(
    db: Session,
    *,
    csv_path: Path | None = None,
    replace: bool = True,
    max_phones: int = 0,
    max_reviews_per_phone: int = 0,
    min_reviews: int = 0,
    brand_filter: list[str] | None = None,
    drop_manuals: bool = True,
    settings: Settings | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Load ``full_reviews.csv`` into SQLite with methodology Step 1 on insert."""
    settings = settings or get_settings()
    # Methodology: English reviews only.
    if not settings.language_filter:
        settings.language_filter = "en"

    def report(done: int, total: int, message: str) -> None:
        if progress is not None:
            progress(done, total, message)
        logger.info("[%s/%s] %s", done, total, message)

    csv_path = Path(csv_path) if csv_path else DEFAULT_CSV
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path.resolve()}")

    report(0, 1, f"Loading {csv_path}…")
    frame = pd.read_csv(csv_path)
    required = {"asin", "product_title", "review_comment"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"CSV missing columns {sorted(missing)}. Found: {list(frame.columns)}")

    summary: dict[str, Any] = {
        "dataset": str(csv_path),
        "source": SOURCE,
        "rows_in_csv": int(len(frame)),
        "manuals_dropped": 0,
        "phones_upserted": 0,
        "prices_recorded": 0,
        "reviews_inserted": 0,
        "reviews_duplicate": 0,
        "reviews_excluded": 0,
        "phone_ids": [],
        "asins": [],
        "brands": [],
        "language_filter": settings.language_filter,
        "loaded_at": datetime.now(timezone.utc).isoformat(),
    }

    if drop_manuals:
        titles = frame["product_title"].fillna("").astype(str)
        mask_manual = titles.map(_is_manual_listing)
        summary["manuals_dropped"] = int(mask_manual.sum())
        frame = frame[~mask_manual].copy()

    # One product row per ASIN (first occurrence keeps product metadata).
    products = frame.drop_duplicates(subset=["asin"], keep="first").copy()
    products["asin"] = products["asin"].astype(str).str.strip().str.upper()

    if brand_filter:
        wanted = {b.strip().lower() for b in brand_filter if b.strip()}
        resolved = products.apply(
            lambda r: (_resolve_brand(str(r.get("brand") or ""), str(r.get("product_title") or "")) or "").lower(),
            axis=1,
        )
        products = products[resolved.isin(wanted)]

    if min_reviews > 0:
        counts = frame.groupby(frame["asin"].astype(str).str.upper()).size()
        keep = set(counts[counts >= min_reviews].index)
        products = products[products["asin"].isin(keep)]

    products = products.sort_values(
        by=["reviews_count", "product_avg_rating"],
        ascending=[False, False],
        na_position="last",
    )
    if max_phones > 0:
        products = products.head(max_phones)

    if replace:
        report(0, 1, "Clearing existing corpus…")
        reset_corpus(db)
        db.commit()

    selected_asins = set(products["asin"].tolist())
    url_by_asin = {
        str(row["asin"]).strip().upper(): str(row.get("product_url") or "")
        for _, row in products.iterrows()
    }

    report(0, len(products), f"Importing {len(products)} phone(s)…")
    asin_to_id: dict[str, int] = {}
    brand_set: set[str] = set()

    for index, (_, row) in enumerate(products.iterrows(), start=1):
        product = _row_to_product(row)
        phone = upsert_product(db, product)
        if store_price(db, phone, product.price):
            summary["prices_recorded"] += 1
        asin_to_id[product.source_product_id] = phone.id
        summary["phones_upserted"] += 1
        summary["phone_ids"].append(phone.id)
        summary["asins"].append(product.source_product_id)
        if product.brand:
            brand_set.add(product.brand)
        if index % 25 == 0:
            db.commit()
            report(index, len(products), phone.display_name)

    db.commit()

    reviews = frame[frame["asin"].astype(str).str.upper().isin(selected_asins)].copy()
    by_asin: dict[str, list[ScrapedReview]] = defaultdict(list)

    for _, row in reviews.iterrows():
        asin = str(row["asin"]).strip().upper()
        if asin not in asin_to_id:
            continue
        scraped = _row_to_review(row, url_by_asin.get(asin) or None)
        if scraped is None:
            continue
        bucket = by_asin[asin]
        if max_reviews_per_phone > 0 and len(bucket) >= max_reviews_per_phone:
            continue
        bucket.append(scraped)

    report(0, len(by_asin), f"Importing reviews for {len(by_asin)} phone(s) (English Step 1)…")
    for index, (asin, scraped_list) in enumerate(by_asin.items(), start=1):
        phone = db.get(Smartphone, asin_to_id[asin])
        if phone is None:
            continue
        counts = store_reviews(db, phone, scraped_list, settings=settings)
        summary["reviews_inserted"] += counts["inserted"]
        summary["reviews_duplicate"] += counts["duplicates"]
        summary["reviews_excluded"] += counts["excluded"]
        if index % 25 == 0:
            db.commit()
            report(index, len(by_asin), f"{phone.display_name}: +{counts['inserted']}")

    db.commit()
    summary["brands"] = sorted(brand_set)
    report(len(by_asin), len(by_asin), "Scraped CSV ingest finished")
    return summary
