"""Build a real McAuley Amazon-Reviews-2023 smartphone corpus (no demo data).

Designed to run in Google Colab or locally. It:

1. Streams product metadata and keeps real handsets (accessories filtered out)
2. Streams the Cell Phones review JSONL and keeps *exact* review text for those ASINs
3. Applies methodology Step 1 preprocessing (dedupe, spam/empty, HTML, normalise, language)
4. Writes a portable folder you can import into the FastAPI app:

       phones.jsonl
       reviews.jsonl          # raw + cleaned + exclusion flags
       manifest.json

Usage (local)::

    python scripts/build_hf_corpus.py --max-phones 40 --max-reviews 100 --out data/exports/hf_corpus

Then::

    python run.py reset-corpus --yes
    python run.py import-corpus data/exports/hf_corpus --analyze
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Allow `python scripts/build_hf_corpus.py` from the repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.nlp.preprocess import preprocess_review
from app.scrapers.amazon import looks_like_phone
from app.scrapers.parsers import extract_brand, extract_model, normalize_specs, parse_number
from app.services.hf_ingest import (
    HF_DATASET,
    META_PARQUET,
    REVIEW_JSONL,
    _as_dict,
    _first_image,
    _has_phone_category,
    _ACCESSORY_EXTRA,
    _PHONE_PRODUCT,
    _stream_parquet,
    _stream_reviews,
    review_scan_exhausted,
)

ASPECTS = ("battery", "camera", "display", "performance", "price")


def is_real_smartphone(row: dict[str, Any]) -> bool:
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


def meta_row_to_phone(row: dict[str, Any]) -> dict[str, Any] | None:
    if not is_real_smartphone(row):
        return None
    parent_asin = (row.get("parent_asin") or "").strip().upper()
    title = (row.get("title") or "").strip()
    if not parent_asin or not title:
        return None

    details = _as_dict(row.get("details"))
    specs_raw = {str(k): str(v) for k, v in details.items()}
    features = row.get("features") or []
    if isinstance(features, list):
        for i, bullet in enumerate(features[:30]):
            if bullet:
                specs_raw[f"feature_{i}"] = str(bullet)

    brand = extract_brand(title, str(details.get("Brand") or details.get("Manufacturer") or ""))
    model = extract_model(
        title,
        brand,
        str(
            details.get("Model Name")
            or details.get("Item model number")
            or details.get("Model")
            or ""
        ),
    )
    normalized = normalize_specs(specs_raw, title=title)
    price_value = (
        parse_number(str(row.get("price")))
        if row.get("price") not in (None, "None", "")
        else None
    )

    try:
        site_rating = float(row["average_rating"]) if row.get("average_rating") is not None else None
    except (TypeError, ValueError):
        site_rating = None
    try:
        site_rating_count = int(row["rating_number"]) if row.get("rating_number") is not None else None
    except (TypeError, ValueError):
        site_rating_count = None

    return {
        "source": "amazon_hf",
        "source_product_id": parent_asin,
        "product_url": f"https://www.amazon.com/dp/{parent_asin}",
        "raw_title": title,
        "brand": brand,
        "model": model,
        "canonical_name": f"{brand} {model}".strip() if brand or model else title[:200],
        "image_url": _first_image(row.get("images")),
        "site_rating": site_rating,
        "site_rating_count": site_rating_count,
        "price": price_value,
        "currency": "USD" if price_value is not None else None,
        "specs_raw": specs_raw,
        "feature_bullets": [str(b) for b in features if b][:20] if isinstance(features, list) else [],
        "normalized_specs": normalized,
        "categories": list(row.get("categories") or []),
    }


def review_row_to_record(
    row: dict[str, Any],
    *,
    min_chars: int = 15,
    language_filter: str = "en",
) -> dict[str, Any] | None:
    text = (row.get("text") or "").strip()
    if not text:
        return None
    parent_asin = (row.get("parent_asin") or row.get("asin") or "").strip().upper()
    if not parent_asin:
        return None

    cleaned = preprocess_review(
        text,
        title=row.get("title"),
        min_chars=min_chars,
        language_filter=language_filter or None,
    )

    ts = row.get("timestamp")
    review_date = None
    if ts is not None:
        try:
            seconds = int(ts) / 1000.0 if int(ts) > 10_000_000_000 else float(ts)
            review_date = datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat()
        except (TypeError, ValueError, OSError, OverflowError):
            review_date = None

    user_id = row.get("user_id") or ""
    stamp = str(ts or "")
    try:
        rating = float(row["rating"]) if row.get("rating") is not None else None
    except (TypeError, ValueError):
        rating = None
    try:
        helpful = int(row["helpful_vote"]) if row.get("helpful_vote") is not None else None
    except (TypeError, ValueError):
        helpful = None

    return {
        "parent_asin": parent_asin,
        "source_review_id": f"{parent_asin}:{user_id}:{stamp}"[:120],
        "title": row.get("title"),
        "body": text,  # exact original review text from the dataset
        "cleaned_body": cleaned.cleaned_text or None,
        "content_hash": cleaned.content_hash,
        "rating": rating,
        "review_date": review_date,
        "reviewer_name": str(user_id)[:64] if user_id else None,
        "verified_purchase": bool(row.get("verified_purchase")),
        "helpful_votes": helpful,
        "language": cleaned.language,
        "is_spam": cleaned.is_spam,
        "excluded_reason": cleaned.excluded_reason,
        "usable": cleaned.usable,
    }


def build_corpus(
    *,
    max_phones: int = 40,
    max_reviews_per_phone: int = 100,
    min_rating_count: int = 200,
    max_review_scan: int = 0,
    brand_filter: list[str] | None = None,
    out_dir: Path,
    min_chars: int = 15,
    language_filter: str = "en",
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    brands = {b.strip().lower() for b in (brand_filter or []) if b.strip()}

    print(f"Dataset: {HF_DATASET}")
    print(f"Meta:    {META_PARQUET}")
    print(f"Reviews: {REVIEW_JSONL}")
    print(f"Output:  {out_dir}")

    phones: list[dict[str, Any]] = []
    asin_set: set[str] = set()
    meta_scanned = 0

    print("\n[1/3] Selecting smartphones from metadata…")
    for row in _stream_parquet(META_PARQUET):
        meta_scanned += 1
        phone = meta_row_to_phone(dict(row))
        if phone is None:
            continue
        if phone["site_rating_count"] is not None and phone["site_rating_count"] < min_rating_count:
            continue
        if brands:
            blob = f"{phone.get('brand') or ''} {phone.get('raw_title') or ''}".lower()
            if not any(b in blob for b in brands):
                continue
        if phone["source_product_id"] in asin_set:
            continue
        phones.append(phone)
        asin_set.add(phone["source_product_id"])
        print(f"  + {phone['canonical_name']} [{phone['source_product_id']}] "
              f"ratings={phone['site_rating_count']}")
        if max_phones > 0 and len(phones) >= max_phones:
            break

    if not phones:
        raise SystemExit("No smartphones matched. Loosen --min-ratings or --brand filters.")

    # Prefer popular phones first so review matching finds more hits earlier.
    phones.sort(key=lambda p: p.get("site_rating_count") or 0, reverse=True)
    if max_phones > 0:
        phones = phones[:max_phones]
    asin_set = {p["source_product_id"] for p in phones}
    print(f"  selected {len(phones)} smartphone(s) "
          f"(cap={'all' if max_phones <= 0 else max_phones})")

    scan_label = (
        "until phones are filled or the JSONL ends"
        if (max_review_scan is None or max_review_scan <= 0)
        else f"up to {max_review_scan:,} review rows"
    )
    print(
        f"\n[2/3] Collecting exact reviews for {len(phones)} phone(s) ({scan_label})…",
        flush=True,
    )
    print(
        "The website still shows the old corpus until this step finishes and SQLite is replaced.",
        flush=True,
    )
    per_phone: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_hash: dict[str, set[str]] = defaultdict(set)
    reviews_scanned = 0
    reviews_kept = 0
    remaining = set(asin_set)
    last_report = time.monotonic()

    def _report() -> None:
        with_reviews = sum(1 for a in asin_set if per_phone[a])
        filled = sum(1 for a in asin_set if len(per_phone[a]) >= max_reviews_per_phone)
        print(
            f"  scanned {reviews_scanned:,} | kept {reviews_kept:,} | "
            f"phones with reviews {with_reviews}/{len(asin_set)} | full {filled}",
            flush=True,
        )

    for row in _stream_reviews():
        reviews_scanned += 1
        if reviews_scanned == 1:
            print("  first review row read — matching ASINs…", flush=True)
        parent = (row.get("parent_asin") or row.get("asin") or "").strip().upper()
        if parent not in remaining:
            now = time.monotonic()
            if reviews_scanned % 25_000 == 0 or now - last_report >= 20:
                _report()
                last_report = now
            if review_scan_exhausted(reviews_scanned, max_review_scan):
                break
            continue

        record = review_row_to_record(
            dict(row), min_chars=min_chars, language_filter=language_filter
        )
        if record is None:
            continue
        # Step 1 dedupe by content hash within each phone
        if record["content_hash"] in seen_hash[parent]:
            continue
        seen_hash[parent].add(record["content_hash"])
        per_phone[parent].append(record)
        reviews_kept += 1

        if len(per_phone[parent]) >= max_reviews_per_phone:
            remaining.discard(parent)

        now = time.monotonic()
        if reviews_scanned % 25_000 == 0 or now - last_report >= 20:
            _report()
            last_report = now

        if not remaining or review_scan_exhausted(reviews_scanned, max_review_scan):
            break

    _report()

    print(f"\n[3/3] Writing corpus files…")
    # Drop phones that got zero usable reviews — they cannot feed ABSA.
    phones_out = []
    reviews_out = []
    for phone in phones:
        asin = phone["source_product_id"]
        rows = per_phone.get(asin) or []
        usable = [r for r in rows if r["usable"]]
        phone = dict(phone)
        phone["review_count_raw"] = len(rows)
        phone["review_count_usable"] = len(usable)
        if not usable:
            print(f"  skip (no usable reviews): {phone['canonical_name']}")
            continue
        phones_out.append(phone)
        reviews_out.extend(rows)

    phones_path = out_dir / "phones.jsonl"
    reviews_path = out_dir / "reviews.jsonl"
    with phones_path.open("w", encoding="utf-8") as fh:
        for phone in phones_out:
            fh.write(json.dumps(phone, ensure_ascii=False) + "\n")
    with reviews_path.open("w", encoding="utf-8") as fh:
        for review in reviews_out:
            fh.write(json.dumps(review, ensure_ascii=False) + "\n")

    exclusion = defaultdict(int)
    for review in reviews_out:
        if review.get("excluded_reason"):
            exclusion[review["excluded_reason"]] += 1
        elif review.get("is_spam"):
            exclusion["spam"] += 1

    manifest = {
        "dataset": HF_DATASET,
        "meta_source": META_PARQUET,
        "review_source": REVIEW_JSONL,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "methodology": {
            "step_1": "preprocess (dedupe, spam/empty, HTML, normalise, language)",
            "step_2_to_6": "run locally via: python run.py import-corpus … --analyze",
            "aspects": list(ASPECTS),
        },
        "filters": {
            "max_phones": max_phones,
            "max_reviews_per_phone": max_reviews_per_phone,
            "min_rating_count": min_rating_count,
            "max_review_scan": max_review_scan,
            "brand_filter": sorted(brands),
            "language_filter": language_filter,
            "min_chars": min_chars,
        },
        "counts": {
            "meta_scanned": meta_scanned,
            "reviews_scanned": reviews_scanned,
            "phones_written": len(phones_out),
            "reviews_written": len(reviews_out),
            "reviews_usable": sum(1 for r in reviews_out if r["usable"]),
            "exclusion_breakdown": dict(exclusion),
        },
        "phones": [
            {
                "asin": p["source_product_id"],
                "name": p["canonical_name"],
                "usable_reviews": p["review_count_usable"],
            }
            for p in phones_out
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest["counts"], indent=2))
    print(f"\nWrote:\n  {phones_path}\n  {reviews_path}\n  {out_dir / 'manifest.json'}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-phones",
        type=int,
        default=0,
        help="How many real smartphones to keep. 0 = every listing that passes the phone filter.",
    )
    parser.add_argument("--max-reviews", type=int, default=100)
    parser.add_argument("--min-ratings", type=int, default=200)
    parser.add_argument(
        "--max-review-scan",
        type=int,
        default=0,
        help="Max review rows to scan. 0 = scan until selected phones are filled or JSONL ends.",
    )
    parser.add_argument("--brand", action="append", default=[])
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "exports" / "hf_corpus")
    parser.add_argument("--min-chars", type=int, default=15)
    parser.add_argument("--language", default="en")
    args = parser.parse_args()

    build_corpus(
        max_phones=args.max_phones,
        max_reviews_per_phone=args.max_reviews,
        min_rating_count=args.min_ratings,
        max_review_scan=args.max_review_scan,
        brand_filter=args.brand,
        out_dir=args.out,
        min_chars=args.min_chars,
        language_filter=args.language,
    )


if __name__ == "__main__":
    main()
