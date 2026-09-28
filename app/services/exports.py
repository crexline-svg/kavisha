"""CSV exports of every pipeline stage, for analysis in R/SPSS/Excel and for the
reproducibility appendix of the dissertation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.models.entities import AspectScore, AspectSentiment, Review, Sentence, Smartphone
from app.nlp.aggregate import corpus_stats
from app.nlp.aspects import aspects_for
from app.services.recommender import build_feature_vectors

logger = get_logger(__name__)


def _frame(db: Session, stmt, columns: list[str]) -> pd.DataFrame:  # noqa: ANN001
    rows = db.execute(stmt).all()
    return pd.DataFrame(rows, columns=columns)


def export_dataset(
    db: Session, out_dir: Path | None = None, settings: Settings | None = None
) -> dict[str, str]:
    """Write one CSV per stage plus the final feature matrix. Returns {name: path}."""
    settings = settings or get_settings()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    directory = (out_dir or settings.exports_dir) / stamp
    directory.mkdir(parents=True, exist_ok=True)

    written: dict[str, str] = {}

    def write(name: str, frame: pd.DataFrame) -> None:
        path = directory / f"{name}.csv"
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        written[name] = str(path)
        logger.info("Exported %s rows to %s", len(frame), path.name)

    write(
        "phones",
        _frame(
            db,
            select(
                Smartphone.id, Smartphone.source, Smartphone.source_product_id,
                Smartphone.brand, Smartphone.model, Smartphone.canonical_name,
                Smartphone.raw_title, Smartphone.site_rating, Smartphone.site_rating_count,
                Smartphone.ram_gb, Smartphone.storage_gb, Smartphone.display_inches,
                Smartphone.battery_mah, Smartphone.rear_camera_mp, Smartphone.refresh_rate_hz,
                Smartphone.chipset, Smartphone.operating_system, Smartphone.release_year,
                Smartphone.product_url,
            ).order_by(Smartphone.id),
            [
                "id", "source", "source_product_id", "brand", "model", "canonical_name",
                "raw_title", "site_rating", "site_rating_count", "ram_gb", "storage_gb",
                "display_inches", "battery_mah", "rear_camera_mp", "refresh_rate_hz",
                "chipset", "operating_system", "release_year", "product_url",
            ],
        ),
    )

    write(
        "reviews",
        _frame(
            db,
            select(
                Review.id, Review.smartphone_id, Review.source_review_id, Review.rating,
                Review.review_date, Review.verified_purchase, Review.helpful_votes,
                Review.language, Review.is_spam, Review.excluded_reason, Review.title,
                Review.cleaned_body, Review.scraped_at, Review.processed_at,
                Review.pipeline_version,
            ).order_by(Review.id),
            [
                "id", "smartphone_id", "source_review_id", "rating", "review_date",
                "verified_purchase", "helpful_votes", "language", "is_spam",
                "excluded_reason", "title", "cleaned_body", "scraped_at", "processed_at",
                "pipeline_version",
            ],
        ),
    )

    write(
        "sentences",
        _frame(
            db,
            select(
                Sentence.id, Sentence.review_id, Sentence.smartphone_id,
                Sentence.position, Sentence.text, Sentence.word_count,
            ).order_by(Sentence.id),
            ["id", "review_id", "smartphone_id", "position", "text", "word_count"],
        ),
    )

    write(
        "aspect_sentiments",
        _frame(
            db,
            select(
                AspectSentiment.id, AspectSentiment.smartphone_id, AspectSentiment.review_id,
                AspectSentiment.sentence_id, AspectSentiment.aspect, AspectSentiment.sentiment,
                AspectSentiment.confidence, AspectSentiment.opinion_term,
                AspectSentiment.method, AspectSentiment.model_name,
            ).order_by(AspectSentiment.id),
            [
                "id", "smartphone_id", "review_id", "sentence_id", "aspect", "sentiment",
                "confidence", "opinion_term", "method", "model_name",
            ],
        ),
    )

    write(
        "aspect_scores",
        _frame(
            db,
            select(
                AspectScore.smartphone_id, AspectScore.aspect, AspectScore.positive_count,
                AspectScore.negative_count, AspectScore.neutral_count,
                AspectScore.mention_count, AspectScore.raw_score, AspectScore.score,
                AspectScore.confidence, AspectScore.method, AspectScore.computed_at,
            ).order_by(AspectScore.smartphone_id, AspectScore.aspect),
            [
                "smartphone_id", "aspect", "positive_count", "negative_count",
                "neutral_count", "mention_count", "raw_score", "score", "confidence",
                "method", "computed_at",
            ],
        ),
    )

    # Wide feature matrix: exactly the table in the methodology's output layer.
    allowed = aspects_for(settings.aspect_set)
    vectors = build_feature_vectors(db, settings=settings)
    matrix_rows = []
    for vector in vectors:
        row: dict[str, object] = {
            "smartphone_id": vector.smartphone_id,
            "name": vector.name,
            "brand": vector.brand,
            "price": vector.price,
            "currency": vector.currency,
            "review_count": vector.review_count,
        }
        for aspect in allowed:
            row[aspect] = vector.scores.get(aspect)
            row[f"{aspect}_mentions"] = vector.mentions.get(aspect, 0)
        row["affordability"] = vector.affordability
        matrix_rows.append(row)
    write("feature_matrix", pd.DataFrame(matrix_rows))

    stats = corpus_stats(db, settings)
    write("corpus_stats", pd.DataFrame([{k: str(v) for k, v in stats.items()}]))

    return written
