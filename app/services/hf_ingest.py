"""Ingest McAuley-Lab / Amazon-Reviews-2023 into the existing SQLite schema.

Uses the public Hugging Face configs the methodology expects for offline
research (no live Amazon scrape):

    raw_meta_Cell_Phones_and_Accessories
    raw_review_Cell_Phones_and_Accessories

Step 1 preprocessing (clean, language/spam filter, content-hash de-dupe) runs
at insert time via the same helpers as the scraper path, so the rest of the
pipeline (segment → ABSA → aggregate → recommend) is unchanged.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.models.entities import Smartphone
from app.scrapers.amazon import looks_like_phone
from app.scrapers.base import ScrapedPrice, ScrapedProduct, ScrapedReview
from app.scrapers.parsers import (
    extract_brand,
    extract_model,
    normalize_specs,
    parse_number,
)
from app.scrapers.pipeline import store_price, store_reviews, upsert_product

logger = get_logger(__name__)

ProgressCallback = Callable[[int, int, str], None]

HF_DATASET = "McAuley-Lab/Amazon-Reviews-2023"
META_CONFIG = "raw_meta_Cell_Phones_and_Accessories"
REVIEW_CONFIG = "raw/review_categories/Cell_Phones_and_Accessories.jsonl"
META_PARQUET = (
    f"hf://datasets/{HF_DATASET}/raw_meta_Cell_Phones_and_Accessories/full-*.parquet"
)
REVIEW_JSONL = f"hf://datasets/{HF_DATASET}/{REVIEW_CONFIG}"


def _stream_parquet(pattern: str):
    """Stream McAuley meta parquet shards."""
    from datasets import load_dataset

    return load_dataset("parquet", data_files=pattern, split="train", streaming=True)


def _stream_reviews():
    """Yield Cell Phones reviews from a locally cached JSONL (faster than HF streaming)."""
    from huggingface_hub import hf_hub_download

    print(
        "Opening review JSONL (first run downloads several GB into the Hugging Face cache)…",
        flush=True,
    )
    path = hf_hub_download(
        repo_id=HF_DATASET,
        filename=REVIEW_CONFIG,
        repo_type="dataset",
    )
    print(f"Reading reviews from {path}", flush=True)
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def review_scan_exhausted(scanned: int, max_review_scan: int) -> bool:
    """True when the optional row cap is hit. ``max_review_scan <= 0`` means unlimited."""
    if max_review_scan is None or max_review_scan <= 0:
        return False
    return scanned >= max_review_scan

_PHONE_CATEGORY = re.compile(
    r"\b(?:cell\s*phones?|smartphones?|unlocked\s*phones?|mobile\s*phones?)\b",
    re.IGNORECASE,
)
_ACCESSORY_CATEGORY = re.compile(
    r"\b(?:cases?|covers?|screen\s*protectors?|chargers?|cables?|headphones?|"
    r"earbuds?|power\s*banks?|holders?|mounts?|stands?)\b",
    re.IGNORECASE,
)


def _as_dict(details: Any) -> dict[str, Any]:
    if isinstance(details, dict):
        return {str(k): v for k, v in details.items() if v is not None}
    if isinstance(details, str) and details.strip():
        try:
            parsed = json.loads(details)
            if isinstance(parsed, dict):
                return {str(k): v for k, v in parsed.items() if v is not None}
        except json.JSONDecodeError:
            return {}
    return {}


def _first_image(images: Any) -> str | None:
    if not images:
        return None
    # HF schema: dict of parallel lists, or list of dicts.
    if isinstance(images, dict):
        for key in ("hi_res", "large", "thumb"):
            values = images.get(key) or []
            for value in values:
                if value:
                    return str(value)
        return None
    if isinstance(images, list):
        for item in images:
            if isinstance(item, dict):
                for key in ("hi_res", "large", "thumb"):
                    if item.get(key):
                        return str(item[key])
            elif item:
                return str(item)
    return None


def _categories_text(categories: Any) -> str:
    if not categories:
        return ""
    if isinstance(categories, (list, tuple)):
        return " | ".join(str(c) for c in categories if c)
    return str(categories)


_ACCESSORY_EXTRA = re.compile(
    r"\b(?:watch\s*band|watch\s*strap|smart\s*watch|galaxy\s*watch|apple\s*watch|"
    r"band\s*for|compatible\s*with|case\s*for|cover\s*for|protector\s*for|"
    r"charger\s*for|cable\s*for|holder\s*for|mount\s*for|film|tempered\s*glass|"
    r"replacement|sim\s*card|memory\s*card|sd\s*card|micro\s*sd|"
    r"earphone|earbuds?|headphones?|bluetooth\s*speaker|"
    r"car\s*mount|car\s*cradle|car\s*holder|phone\s*holder|phone\s*mount|"
    r"magnetic\s*(?:mount|holder|cradle)|air\s*vent|"
    r"charging\s*cable|usb\s*cable|sync\s*(?:and|&)\s*charging|"
    r"screen\s*protector|wallet\s*case|flip\s*case|clear\s*case|"
    r"power\s*bank|battery\s*pack|stylus|selfie\s*stick)\b",
    re.IGNORECASE,
)

# Title describes a handset as the product, not an accessory that mentions a phone.
_PHONE_PRODUCT = re.compile(
    r"(?:^|\b)(?:Apple\s+)?iPhone\s+(?:\d{1,2}|X[RS]?|SE|Mini|Pro)\b|"
    r"(?:^|\b)(?:Samsung\s+)?Galaxy\s+(?:S|A|Z\s*(?:Fold|Flip)?|M|F|Note|J|Y)\s*\d|"
    r"(?:^|\b)(?:Google\s+)?Pixel\s+\d|"
    r"(?:^|\b)(?:OnePlus|Xiaomi|Redmi|POCO|Nothing\s+Phone|Motorola|Moto|Nokia|Sony\s+Xperia|OPPO|vivo|realme|Honor)\b|"
    r"\b(?:smartphone|smart\s*phone|factory\s+unlocked\s+(?:android\s+)?phone|"
    r"unlocked\s+(?:android\s+)?(?:cell\s+)?phone|dual\s*sim\s+(?:android\s+)?phone)\b",
    re.IGNORECASE,
)


def _has_phone_category(categories: Any) -> bool:
    """True when the listing sits under a phone subcategory (not root accessories)."""
    if not categories:
        return False
    items = categories if isinstance(categories, (list, tuple)) else [categories]
    for raw in items:
        name = str(raw).strip().lower()
        if name in {
            "cell phones",
            "unlocked cell phones",
            "smartphones",
            "unlocked phones",
            "mobile phones",
        }:
            return True
        if name.endswith("cell phones") and "accessories" not in name:
            return True
    return False


def is_smartphone_meta(row: dict[str, Any]) -> bool:
    """Keep handsets; drop the accessory-heavy majority of the category."""
    title = (row.get("title") or "").strip()
    if not title:
        return False
    if _ACCESSORY_EXTRA.search(title):
        return False
    if not looks_like_phone(title):
        return False
    if not _PHONE_PRODUCT.search(title):
        return False
    if not _has_phone_category(row.get("categories")):
        return False
    return True


def meta_to_product(row: dict[str, Any]) -> ScrapedProduct | None:
    parent_asin = (row.get("parent_asin") or "").strip().upper()
    title = (row.get("title") or "").strip()
    if not parent_asin or not title:
        return None
    if not is_smartphone_meta(row):
        return None

    details = _as_dict(row.get("details"))
    # Flatten details + feature bullets into a haystack for normalize_specs.
    specs_raw = {str(k): str(v) for k, v in details.items()}
    features = row.get("features") or []
    if isinstance(features, list):
        for index, bullet in enumerate(features[:30]):
            if bullet:
                specs_raw[f"feature_{index}"] = str(bullet)

    brand = extract_brand(title, str(details.get("Brand") or details.get("Manufacturer") or ""))
    model = extract_model(
        title,
        brand,
        str(details.get("Model Name") or details.get("Item model number") or details.get("Model") or ""),
    )
    normalized = normalize_specs(specs_raw, title=title)

    price_value = parse_number(str(row.get("price"))) if row.get("price") not in (None, "None", "") else None
    price = None
    if price_value is not None and price_value > 0:
        price = ScrapedPrice(price=price_value, currency="USD")

    rating = row.get("average_rating")
    try:
        site_rating = float(rating) if rating is not None else None
    except (TypeError, ValueError):
        site_rating = None

    rating_number = row.get("rating_number")
    try:
        site_rating_count = int(rating_number) if rating_number is not None else None
    except (TypeError, ValueError):
        site_rating_count = None

    feature_bullets = [str(b) for b in features if b][:20] if isinstance(features, list) else []

    return ScrapedProduct(
        source="amazon_hf",
        source_product_id=parent_asin,
        product_url=f"https://www.amazon.com/dp/{parent_asin}",
        raw_title=title,
        brand=brand,
        model=model,
        canonical_name=f"{brand} {model}".strip() if brand or model else title[:200],
        image_url=_first_image(row.get("images")),
        site_rating=site_rating,
        site_rating_count=site_rating_count,
        specs_raw=specs_raw,
        feature_bullets=feature_bullets,
        normalized_specs=normalized,
        price=price,
    )


def review_to_scraped(row: dict[str, Any]) -> ScrapedReview | None:
    text = (row.get("text") or "").strip()
    if not text:
        return None

    parent_asin = (row.get("parent_asin") or row.get("asin") or "").strip().upper()
    if not parent_asin:
        return None

    rating = row.get("rating")
    try:
        rating_f = float(rating) if rating is not None else None
    except (TypeError, ValueError):
        rating_f = None

    ts = row.get("timestamp")
    review_date = None
    if ts is not None:
        try:
            # Dataset timestamps are milliseconds since epoch.
            seconds = int(ts) / 1000.0 if int(ts) > 10_000_000_000 else float(ts)
            review_date = datetime.fromtimestamp(seconds, tz=timezone.utc)
        except (TypeError, ValueError, OSError, OverflowError):
            review_date = None

    helpful = row.get("helpful_vote")
    try:
        helpful_votes = int(helpful) if helpful is not None else None
    except (TypeError, ValueError):
        helpful_votes = None

    user_id = row.get("user_id") or ""
    stamp = str(ts or "")
    source_review_id = f"{parent_asin}:{user_id}:{stamp}"[:120]

    return ScrapedReview(
        source_review_id=source_review_id,
        source_url=f"https://www.amazon.com/dp/{parent_asin}",
        title=(row.get("title") or None),
        body=text,
        rating=rating_f,
        review_date=review_date,
        reviewer_name=str(user_id)[:64] if user_id else None,
        verified_purchase=bool(row.get("verified_purchase")),
        helpful_votes=helpful_votes,
        variant=None,
        country=None,
    )


def ingest_amazon_reviews_2023(
    db: Session,
    *,
    max_phones: int = 40,
    max_reviews_per_phone: int = 80,
    min_rating_count: int = 50,
    max_review_scan: int = 0,
    brand_filter: list[str] | None = None,
    settings: Settings | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Stream HF meta + reviews into the existing database."""
    settings = settings or get_settings()

    try:
        import datasets  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Install the Hugging Face datasets package: pip install datasets"
        ) from exc

    def report(done: int, total: int, message: str) -> None:
        if progress is not None:
            progress(done, total, message)
        logger.info("[%s/%s] %s", done, total, message)

    brands = {b.strip().lower() for b in (brand_filter or []) if b.strip()}
    summary: dict[str, Any] = {
        "dataset": HF_DATASET,
        "meta_config": META_CONFIG,
        "review_config": REVIEW_CONFIG,
        "phones_upserted": 0,
        "prices_recorded": 0,
        "reviews_inserted": 0,
        "reviews_duplicate": 0,
        "reviews_excluded": 0,
        "meta_scanned": 0,
        "reviews_scanned": 0,
        "phone_ids": [],
        "asins": [],
    }

    report(0, max_phones, "Loading product metadata stream…")
    meta_stream = _stream_parquet(META_PARQUET)

    wanted: dict[str, int] = {}  # parent_asin -> smartphone.id
    for row in meta_stream:
        summary["meta_scanned"] += 1
        product = meta_to_product(dict(row))
        if product is None:
            continue
        if product.site_rating_count is not None and product.site_rating_count < min_rating_count:
            continue
        if brands:
            brand = (product.brand or "").lower()
            title = (product.raw_title or "").lower()
            if not any(b in brand or b in title for b in brands):
                continue

        phone = upsert_product(db, product)
        if store_price(db, phone, product.price):
            summary["prices_recorded"] += 1
        wanted[product.source_product_id] = phone.id
        summary["phones_upserted"] += 1
        summary["phone_ids"].append(phone.id)
        summary["asins"].append(product.source_product_id)
        report(len(wanted), max_phones, f"Kept {product.canonical_name}")
        db.commit()

        if max_phones > 0 and len(wanted) >= max_phones:
            break

    if not wanted:
        report(0, 1, "No smartphones matched the filters.")
        return summary

    report(0, len(wanted), f"Loading reviews for {len(wanted)} phone(s)…")
    review_stream = _stream_reviews()

    per_phone_left = {asin: max_reviews_per_phone for asin in wanted}
    remaining = set(wanted)

    for row in review_stream:
        summary["reviews_scanned"] += 1
        parent = (row.get("parent_asin") or row.get("asin") or "").strip().upper()
        if parent not in remaining:
            if review_scan_exhausted(summary["reviews_scanned"], max_review_scan):
                break
            continue

        scraped = review_to_scraped(dict(row))
        if scraped is None:
            continue

        phone = db.get(Smartphone, wanted[parent])
        if phone is None:
            remaining.discard(parent)
            continue

        counts = store_reviews(db, phone, [scraped], settings=settings)
        summary["reviews_inserted"] += counts["inserted"]
        summary["reviews_duplicate"] += counts["duplicates"]
        summary["reviews_excluded"] += counts["excluded"]

        if counts["inserted"]:
            per_phone_left[parent] -= 1
            if per_phone_left[parent] <= 0:
                remaining.discard(parent)

        if summary["reviews_scanned"] % 5000 == 0:
            db.commit()
            done = len(wanted) - len(remaining)
            report(
                done,
                len(wanted),
                f"Reviews filled for {done}/{len(wanted)} phones "
                f"(scanned {summary['reviews_scanned']:,})",
            )

        if not remaining or review_scan_exhausted(summary["reviews_scanned"], max_review_scan):
            break

    db.commit()
    report(len(wanted), len(wanted), "Ingest finished")
    return summary
