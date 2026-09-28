"""Runs Steps 1-6 over reviews already in the database.

Per phone:
    reviews -> (re)preprocess -> segment -> ABSA -> aspect_sentiments -> aggregate

Sentences for a whole phone are annotated in one `engine.analyze` call so the LLM
engine can batch them; that is where nearly all the wall-clock time goes.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.database import session_scope
from app.core.logging import get_logger
from app.models.entities import AspectSentiment, Review, Sentence, Smartphone
from app.nlp.absa import AbsaEngine, get_absa_engine
from app.nlp.aggregate import recompute_aspect_scores
from app.nlp.preprocess import preprocess_review
from app.nlp.segment import count_words, segment_sentences

logger = get_logger(__name__)

ProgressCallback = Callable[[int, int, str], None]


def _phones_to_process(db: Session, phone_ids: list[int] | None) -> list[int]:
    stmt = select(Smartphone.id).order_by(Smartphone.id)
    if phone_ids:
        stmt = stmt.where(Smartphone.id.in_(phone_ids))
    return list(db.scalars(stmt).all())


def _pending_reviews(
    db: Session, phone_id: int, force: bool, limit: int | None
) -> list[Review]:
    stmt = select(Review).where(Review.smartphone_id == phone_id)
    if not force:
        stmt = stmt.where(
            Review.processed_at.is_(None),
            Review.excluded_reason.is_(None),
            Review.is_spam.is_(False),
        )
    stmt = stmt.order_by(Review.helpful_votes.desc().nullslast(), Review.id)
    if limit:
        stmt = stmt.limit(limit)
    return list(db.scalars(stmt).all())


def analyze_phone(
    db: Session,
    phone_id: int,
    engine: AbsaEngine,
    *,
    force: bool = False,
    max_reviews: int | None = None,
    settings: Settings | None = None,
) -> dict[str, int]:
    """Segment + annotate one phone's outstanding reviews."""
    settings = settings or get_settings()
    stats = {
        "reviews_processed": 0,
        "reviews_skipped": 0,
        "sentences_created": 0,
        "aspects_extracted": 0,
    }

    reviews = _pending_reviews(db, phone_id, force, max_reviews)
    if not reviews:
        return stats

    if force:
        review_ids = [review.id for review in reviews]
        # Cascade removes the dependent aspect_sentiments rows.
        db.execute(delete(Sentence).where(Sentence.review_id.in_(review_ids)))
        db.flush()

    now = datetime.now(timezone.utc)
    pending: list[tuple[Sentence, str]] = []

    for review in reviews:
        # Re-run Step 1 when forcing, so changed filters/settings take effect.
        if force or not review.cleaned_body:
            cleaned = preprocess_review(
                review.body,
                title=review.title,
                min_chars=settings.min_review_chars,
                language_filter=settings.language_filter or None,
            )
            review.cleaned_body = cleaned.cleaned_text or None
            review.language = cleaned.language
            review.is_spam = cleaned.is_spam
            review.excluded_reason = cleaned.excluded_reason
            review.pipeline_version = settings.pipeline_version

        if not review.cleaned_body or review.excluded_reason or review.is_spam:
            review.processed_at = now
            stats["reviews_skipped"] += 1
            continue

        sentences = segment_sentences(review.cleaned_body)
        if not sentences:
            review.processed_at = now
            stats["reviews_skipped"] += 1
            continue

        for position, text in enumerate(sentences):
            row = Sentence(
                review_id=review.id,
                smartphone_id=phone_id,
                position=position,
                text=text,
                char_count=len(text),
                word_count=count_words(text),
            )
            db.add(row)
            pending.append((row, text))

        review.processed_at = now
        review.pipeline_version = settings.pipeline_version
        stats["reviews_processed"] += 1

    db.flush()  # assigns sentence ids
    stats["sentences_created"] = len(pending)

    if not pending:
        return stats

    texts = [text for _row, text in pending]
    logger.info("Annotating %s sentence(s) for phone %s.", len(texts), phone_id)
    annotations = engine.analyze(texts)

    for (row, _text), opinions in zip(pending, annotations, strict=False):
        for opinion in opinions:
            db.add(
                AspectSentiment(
                    sentence_id=row.id,
                    review_id=row.review_id,
                    smartphone_id=phone_id,
                    aspect=opinion.aspect,
                    sentiment=opinion.sentiment,
                    confidence=opinion.confidence,
                    opinion_term=opinion.opinion_term,
                    method=opinion.method,
                    model_name=opinion.model_name,
                    pipeline_version=settings.pipeline_version,
                )
            )
            stats["aspects_extracted"] += 1

    db.flush()
    return stats


def segment_phone(
    db: Session,
    phone_id: int,
    *,
    force: bool = False,
    max_reviews: int | None = None,
    settings: Settings | None = None,
) -> dict[str, int]:
    """Steps 1-2 only: clean and split reviews into sentences, no ABSA.

    Used when aspect sentiment is produced elsewhere (for example a transformer
    model running in Colab) and imported afterwards.
    """
    settings = settings or get_settings()
    stats = {"reviews_processed": 0, "reviews_skipped": 0, "sentences_created": 0}

    reviews = _pending_reviews(db, phone_id, force, max_reviews)
    if not reviews:
        return stats

    if force:
        review_ids = [review.id for review in reviews]
        db.execute(delete(Sentence).where(Sentence.review_id.in_(review_ids)))
        db.flush()

    now = datetime.now(timezone.utc)

    for review in reviews:
        if force or not review.cleaned_body:
            cleaned = preprocess_review(
                review.body,
                title=review.title,
                min_chars=settings.min_review_chars,
                language_filter=settings.language_filter or None,
            )
            review.cleaned_body = cleaned.cleaned_text or None
            review.language = cleaned.language
            review.is_spam = cleaned.is_spam
            review.excluded_reason = cleaned.excluded_reason
            review.pipeline_version = settings.pipeline_version

        if not review.cleaned_body or review.excluded_reason or review.is_spam:
            review.processed_at = now
            stats["reviews_skipped"] += 1
            continue

        sentences = segment_sentences(review.cleaned_body)
        if not sentences:
            review.processed_at = now
            stats["reviews_skipped"] += 1
            continue

        for position, text in enumerate(sentences):
            db.add(
                Sentence(
                    review_id=review.id,
                    smartphone_id=phone_id,
                    position=position,
                    text=text,
                    char_count=len(text),
                    word_count=count_words(text),
                )
            )
            stats["sentences_created"] += 1

        review.processed_at = now
        review.pipeline_version = settings.pipeline_version
        stats["reviews_processed"] += 1

    db.flush()
    return stats


def run_segmentation(
    *,
    phone_ids: list[int] | None = None,
    force: bool = False,
    max_reviews_per_phone: int | None = None,
    settings: Settings | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Clean and segment every phone's reviews without running ABSA."""
    settings = settings or get_settings()
    totals: dict[str, Any] = {
        "phones_processed": 0,
        "reviews_processed": 0,
        "reviews_skipped": 0,
        "sentences_created": 0,
        "errors": [],
    }

    with session_scope() as db:
        targets = _phones_to_process(db, phone_ids)

    if not targets:
        totals["errors"].append("No phones found. Build a corpus first.")
        return totals

    total = len(targets)
    if progress:
        progress(0, total, f"Cleaning and segmenting {total} phone(s)")

    for index, phone_id in enumerate(targets, start=1):
        try:
            with session_scope() as db:
                stats = segment_phone(
                    db,
                    phone_id,
                    force=force,
                    max_reviews=max_reviews_per_phone,
                    settings=settings,
                )
            totals["phones_processed"] += 1
            for key in ("reviews_processed", "reviews_skipped", "sentences_created"):
                totals[key] += stats[key]
            if progress:
                progress(index, total, f"Segmented phone {phone_id} ({index}/{total})")
        except Exception as exc:  # noqa: BLE001
            totals["errors"].append(f"phone {phone_id}: {type(exc).__name__}: {exc}")
            logger.exception("Segmentation failed for phone %s", phone_id)

    return totals


def run_analysis(
    *,
    phone_ids: list[int] | None = None,
    force: bool = False,
    max_reviews_per_phone: int | None = None,
    engine_name: str | None = None,
    settings: Settings | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Analyse many phones and rebuild the aspect score table."""
    settings = settings or get_settings()
    engine = get_absa_engine(engine_name, settings)

    totals: dict[str, Any] = {
        "engine": engine.name,
        "model": engine.model_name,
        "phones_analyzed": 0,
        "reviews_processed": 0,
        "reviews_skipped": 0,
        "sentences_created": 0,
        "aspects_extracted": 0,
        "errors": [],
    }

    def report(done: int, total: int, message: str) -> None:
        if progress is not None:
            progress(done, total, message)

    with session_scope() as db:
        targets = _phones_to_process(db, phone_ids)

    if not targets:
        totals["errors"].append("No phones found to analyse. Scrape some data first.")
        return totals

    total = len(targets)
    report(0, total, f"Analysing {total} phone(s) with the {engine.name} engine")

    for index, phone_id in enumerate(targets, start=1):
        try:
            with session_scope() as db:
                stats = analyze_phone(
                    db,
                    phone_id,
                    engine,
                    force=force,
                    max_reviews=max_reviews_per_phone,
                    settings=settings,
                )
            totals["phones_analyzed"] += 1
            for key in ("reviews_processed", "reviews_skipped", "sentences_created", "aspects_extracted"):
                totals[key] += stats[key]
            report(index, total, f"Analysed phone {phone_id} ({index}/{total})")
        except Exception as exc:  # noqa: BLE001
            totals["errors"].append(f"phone {phone_id}: {type(exc).__name__}: {exc}")
            logger.exception("Analysis failed for phone %s", phone_id)

    close = getattr(engine, "close", None)
    if callable(close):
        close()

    report(total, total, "Aggregating aspect scores")
    with session_scope() as db:
        scores = recompute_aspect_scores(db, None, settings)
        totals["phones_scored"] = len(scores)
        totals["aspect_sentiment_rows"] = db.scalar(select(func.count(AspectSentiment.id))) or 0

    return totals
