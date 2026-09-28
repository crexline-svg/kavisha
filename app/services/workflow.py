"""Staged corpus workflow behind the Streamlit research pages.

One stage at a time, so every methodology step can be inspected before the next
one runs:

    scan metadata (capped)  ->  candidate smartphones  ->  manual selection
    ->  targeted review collection (Step 1)  ->  SQLite import
    ->  Steps 2-6 via app.services.analysis

Nothing here talks to Hugging Face except :func:`scan_metadata` and
:func:`collect_reviews`; every other stage reads the files those two produce, so
a stage can be repeated without downloading again.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import func, select

from app.core.config import get_settings
from app.core.database import session_scope
from app.core.logging import get_logger
from app.models.entities import (
    AspectScore,
    AspectSentiment,
    Review,
    Sentence,
    Smartphone,
)

logger = get_logger(__name__)

ProgressCallback = Callable[[int, int, str], None]

settings = get_settings()

INTERIM_DIR = settings.data_dir / "interim"
CANDIDATE_RECORDS_PATH = INTERIM_DIR / "candidate_phones.jsonl"
CANDIDATES_CSV_PATH = INTERIM_DIR / "candidate_smartphones.csv"
SELECTION_CSV_PATH = INTERIM_DIR / "selected_smartphones.csv"
CORPUS_DIR = settings.exports_dir / "hf_corpus"
MANIFEST_PATH = CORPUS_DIR / "manifest.json"

CANDIDATE_COLUMNS: tuple[str, ...] = (
    "parent_asin",
    "product_name",
    "brand",
    "model",
    "average_rating",
    "rating_number",
    "price",
)


def _report(progress: ProgressCallback | None, done: int, total: int, message: str) -> None:
    if progress is not None:
        progress(done, total, message)


def _candidate_row(record: dict[str, Any]) -> dict[str, Any]:
    """Display row for the candidate table."""
    return {
        "parent_asin": record["source_product_id"],
        "product_name": record.get("canonical_name") or record.get("raw_title"),
        "brand": record.get("brand") or "Unknown",
        "model": record.get("model") or "",
        "average_rating": record.get("site_rating"),
        "rating_number": record.get("site_rating_count"),
        "price": record.get("price"),
    }


# --------------------------------------------------------------------------- #
# Stage 1: metadata scan  ->  candidate smartphones
# --------------------------------------------------------------------------- #
def scan_metadata(
    *,
    metadata_limit: int = 3000,
    min_rating_count: int = 0,
    brands: Sequence[str] | None = None,
    progress: ProgressCallback | None = None,
) -> pd.DataFrame:
    """Scan a capped slice of HF product metadata and keep phone-like listings.

    ``metadata_limit`` bounds how many metadata rows are read, which is what
    actually bounds the runtime. The Cell Phones & Accessories category is mostly
    accessories, so only a fraction of the scanned rows become candidates.
    """
    from app.services.hf_ingest import META_PARQUET, _stream_parquet
    from scripts.build_hf_corpus import meta_row_to_phone

    wanted_brands = {b.strip().lower() for b in (brands or []) if b.strip()}

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    scanned = 0

    _report(progress, 0, metadata_limit, "Opening product metadata stream…")

    # Metadata arrives shard by shard, so a slow network can stall the scan for
    # minutes. Checkpointing means a stall or Ctrl+C still leaves a usable pool.
    try:
        for row in _stream_parquet(META_PARQUET):
            scanned += 1
            record = meta_row_to_phone(dict(row))

            if record is not None and record["source_product_id"] not in seen:
                rating_count = record.get("site_rating_count")
                passes_rating = (
                    min_rating_count <= 0
                    or rating_count is None
                    or rating_count >= min_rating_count
                )
                blob = f"{record.get('brand') or ''} {record.get('raw_title') or ''}".lower()
                passes_brand = not wanted_brands or any(b in blob for b in wanted_brands)

                if passes_rating and passes_brand:
                    records.append(record)
                    seen.add(record["source_product_id"])

            if scanned % 2000 == 0:
                _write_candidates(records)
                _report(
                    progress,
                    min(scanned, metadata_limit),
                    metadata_limit,
                    f"Scanned {scanned:,} metadata rows · {len(records)} candidate phone(s) saved",
                )
            if scanned >= metadata_limit:
                break
    except KeyboardInterrupt:
        logger.warning("Metadata scan interrupted after %s rows.", scanned)

    frame = _write_candidates(records)
    _report(
        progress,
        metadata_limit,
        metadata_limit,
        f"Kept {len(frame)} candidate phone(s) from {scanned:,} metadata rows",
    )
    logger.info("Metadata scan: %s rows -> %s candidates", scanned, len(frame))
    return frame


def _write_candidates(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Persist the candidate pool so far and return the display table."""
    with CANDIDATE_RECORDS_PATH.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    frame = pd.DataFrame([_candidate_row(r) for r in records], columns=list(CANDIDATE_COLUMNS))
    if not frame.empty:
        frame = frame.sort_values(
            ["rating_number", "average_rating"], ascending=False, na_position="last"
        ).reset_index(drop=True)
    frame.to_csv(CANDIDATES_CSV_PATH, index=False, encoding="utf-8")
    return frame


def load_candidates() -> pd.DataFrame:
    if not CANDIDATES_CSV_PATH.is_file():
        return pd.DataFrame(columns=list(CANDIDATE_COLUMNS))
    return pd.read_csv(CANDIDATES_CSV_PATH)


def load_candidate_records() -> dict[str, dict[str, Any]]:
    """ASIN -> full metadata record (specs, price, images) kept by the scan."""
    if not CANDIDATE_RECORDS_PATH.is_file():
        return {}
    records: dict[str, dict[str, Any]] = {}
    with CANDIDATE_RECORDS_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                record = json.loads(line)
                records[record["source_product_id"]] = record
    return records


def filter_candidates(
    candidates: pd.DataFrame,
    *,
    brand: str | None = None,
    name_query: str | None = None,
    min_rating_count: int = 0,
    min_average_rating: float = 0.0,
) -> pd.DataFrame:
    """Display-only narrowing of the candidate table."""
    if candidates.empty:
        return candidates

    visible = candidates
    if brand and brand != "All":
        visible = visible[visible["brand"].astype(str) == brand]
    if name_query:
        needle = name_query.strip().lower()
        visible = visible[visible["product_name"].astype(str).str.lower().str.contains(needle)]
    if min_rating_count:
        visible = visible[visible["rating_number"].fillna(0) >= min_rating_count]
    if min_average_rating:
        visible = visible[visible["average_rating"].fillna(0) >= min_average_rating]
    return visible.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Stage 2: manual selection
# --------------------------------------------------------------------------- #
def save_selection(rows: pd.DataFrame) -> Path:
    if rows.empty:
        raise ValueError("Select at least one smartphone before saving.")
    missing = [column for column in ("parent_asin", "product_name") if column not in rows.columns]
    if missing:
        raise ValueError(f"Selection is missing required columns: {missing}")

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    rows.to_csv(SELECTION_CSV_PATH, index=False, encoding="utf-8")
    logger.info("Saved %s selected smartphone(s).", len(rows))
    return SELECTION_CSV_PATH


def load_selection() -> pd.DataFrame:
    if not SELECTION_CSV_PATH.is_file():
        return pd.DataFrame(columns=list(CANDIDATE_COLUMNS))
    return pd.read_csv(SELECTION_CSV_PATH)


# --------------------------------------------------------------------------- #
# Stage 3: targeted review collection + Step 1 preprocessing
# --------------------------------------------------------------------------- #
def collect_reviews(
    *,
    asins: Iterable[str],
    max_reviews_per_phone: int = 60,
    max_review_scan: int = 0,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Scan the review stream once, keeping reviews for the selected ASINs only.

    Writes the same ``phones.jsonl`` / ``reviews.jsonl`` / ``manifest.json``
    folder that :func:`app.services.corpus_import.import_corpus_folder` reads, so
    the Colab notebook and this page produce interchangeable corpora.
    """
    from app.services.hf_ingest import _stream_reviews, review_scan_exhausted
    from scripts.build_hf_corpus import ASPECTS, HF_DATASET, META_PARQUET, REVIEW_JSONL
    from scripts.build_hf_corpus import review_row_to_record

    records = load_candidate_records()
    wanted = [str(asin).strip().upper() for asin in asins if str(asin).strip()]
    phones = [records[asin] for asin in wanted if asin in records]
    if not phones:
        raise ValueError(
            "None of the selected ASINs are in the candidate file. Re-run the metadata scan."
        )

    remaining = {phone["source_product_id"] for phone in phones}
    per_phone: dict[str, list[dict[str, Any]]] = {asin: [] for asin in remaining}
    seen_hashes: dict[str, set[str]] = {asin: set() for asin in remaining}
    total_phones = len(phones)
    scanned = 0
    kept = 0

    _report(progress, 0, total_phones, f"Scanning reviews for {total_phones} phone(s)…")

    for row in _stream_reviews():
        scanned += 1
        parent = (row.get("parent_asin") or row.get("asin") or "").strip().upper()

        if parent in remaining:
            record = review_row_to_record(
                dict(row),
                min_chars=settings.min_review_chars,
                language_filter=settings.language_filter or None,
            )
            if record is not None and record["content_hash"] not in seen_hashes[parent]:
                seen_hashes[parent].add(record["content_hash"])
                per_phone[parent].append(record)
                kept += 1
                if len(per_phone[parent]) >= max_reviews_per_phone:
                    remaining.discard(parent)

        if scanned % 25_000 == 0:
            filled = total_phones - len(remaining)
            _report(
                progress,
                filled,
                total_phones,
                f"Scanned {scanned:,} reviews · kept {kept:,} · {filled}/{total_phones} phone(s) full",
            )

        if not remaining or review_scan_exhausted(scanned, max_review_scan):
            break

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    phones_out: list[dict[str, Any]] = []
    reviews_out: list[dict[str, Any]] = []
    skipped: list[str] = []

    for phone in phones:
        asin = phone["source_product_id"]
        rows = per_phone.get(asin) or []
        usable = [row for row in rows if row["usable"]]
        if not usable:
            skipped.append(phone.get("canonical_name") or asin)
            continue
        enriched = dict(phone)
        enriched["review_count_raw"] = len(rows)
        enriched["review_count_usable"] = len(usable)
        phones_out.append(enriched)
        reviews_out.extend(rows)

    with (CORPUS_DIR / "phones.jsonl").open("w", encoding="utf-8") as fh:
        for phone in phones_out:
            fh.write(json.dumps(phone, ensure_ascii=False) + "\n")
    with (CORPUS_DIR / "reviews.jsonl").open("w", encoding="utf-8") as fh:
        for review in reviews_out:
            fh.write(json.dumps(review, ensure_ascii=False) + "\n")

    exclusions: dict[str, int] = {}
    for review in reviews_out:
        reason = review.get("excluded_reason") or ("spam" if review.get("is_spam") else None)
        if reason:
            exclusions[reason] = exclusions.get(reason, 0) + 1

    manifest = {
        "dataset": HF_DATASET,
        "meta_source": META_PARQUET,
        "review_source": REVIEW_JSONL,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "built_by": "streamlit_workflow",
        "methodology": {
            "step_1": "preprocess (dedupe, spam/empty, HTML, normalise, language)",
            "step_2_to_6": "run from the ABSA Analysis and Aspect Scores pages",
            "aspects": list(ASPECTS),
        },
        "filters": {
            "selected_phones": total_phones,
            "max_reviews_per_phone": max_reviews_per_phone,
            "max_review_scan": max_review_scan,
            "language_filter": settings.language_filter,
            "min_chars": settings.min_review_chars,
        },
        "counts": {
            "reviews_scanned": scanned,
            "phones_written": len(phones_out),
            "reviews_written": len(reviews_out),
            "reviews_usable": sum(1 for r in reviews_out if r["usable"]),
            "exclusion_breakdown": exclusions,
            "phones_without_reviews": skipped,
        },
        "phones": [
            {
                "asin": phone["source_product_id"],
                "name": phone.get("canonical_name"),
                "usable_reviews": phone["review_count_usable"],
            }
            for phone in phones_out
        ],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    _report(
        progress,
        total_phones,
        total_phones,
        f"Wrote {len(phones_out)} phone(s) and {len(reviews_out)} review(s)",
    )
    logger.info(
        "Review collection: scanned %s rows, kept %s reviews for %s phones",
        scanned,
        kept,
        len(phones_out),
    )
    return manifest


def load_manifest() -> dict[str, Any] | None:
    if not MANIFEST_PATH.is_file():
        return None
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


# --------------------------------------------------------------------------- #
# Stage 4: import into SQLite
# --------------------------------------------------------------------------- #
def import_corpus(
    *,
    replace: bool = True,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    from app.services.corpus_import import import_corpus_folder

    with session_scope() as db:
        return import_corpus_folder(
            db,
            CORPUS_DIR,
            settings=settings,
            replace=replace,
            progress=progress,
        )


# --------------------------------------------------------------------------- #
# Status shown on every page
# --------------------------------------------------------------------------- #
def _csv_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        return len(pd.read_csv(path, usecols=[0]))
    except (OSError, ValueError):
        return 0


def _jsonl_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def pipeline_status() -> dict[str, Any]:
    """File and database counts for each methodology stage."""
    status: dict[str, Any] = {
        "candidates": _csv_rows(CANDIDATES_CSV_PATH),
        "selected": _csv_rows(SELECTION_CSV_PATH),
        "corpus_phones": _jsonl_rows(CORPUS_DIR / "phones.jsonl"),
        "corpus_reviews": _jsonl_rows(CORPUS_DIR / "reviews.jsonl"),
        "has_candidates": CANDIDATES_CSV_PATH.is_file(),
        "has_selection": SELECTION_CSV_PATH.is_file(),
        "has_corpus": (CORPUS_DIR / "phones.jsonl").is_file(),
    }

    with session_scope() as db:
        status["db_phones"] = db.scalar(select(func.count(Smartphone.id))) or 0
        status["db_reviews"] = db.scalar(select(func.count(Review.id))) or 0
        status["db_reviews_usable"] = (
            db.scalar(
                select(func.count(Review.id)).where(
                    Review.excluded_reason.is_(None), Review.is_spam.is_(False)
                )
            )
            or 0
        )
        status["db_reviews_analysed"] = (
            db.scalar(select(func.count(Review.id)).where(Review.processed_at.is_not(None))) or 0
        )
        status["db_sentences"] = db.scalar(select(func.count(Sentence.id))) or 0
        status["db_aspect_rows"] = db.scalar(select(func.count(AspectSentiment.id))) or 0
        status["db_scored_phones"] = (
            db.scalar(select(func.count(func.distinct(AspectScore.smartphone_id)))) or 0
        )

    status["engine"] = settings.resolved_absa_engine()
    return status
