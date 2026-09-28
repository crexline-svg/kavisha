"""Step 6 - Aspect score aggregation, plus corpus statistics and validation.

Score definition (documented explicitly because the dissertation must state it):

    raw_score = (positive + NEUTRAL_WEIGHT * neutral) / mentions          in [0, 1]

`raw_score` is unbiased but unstable when an aspect has only a couple of
mentions. The stored `score` therefore applies Bayesian shrinkage toward the
corpus mean for that aspect:

    score = (mentions * raw_score + k * corpus_mean) / (mentions + k)

with k = SHRINKAGE_STRENGTH. A phone with 400 battery mentions keeps its own
score almost unchanged; a phone with 2 mentions is pulled toward the average,
which stops a single lucky review from topping the ranking.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.models.entities import (
    AspectScore,
    AspectSentiment,
    PriceObservation,
    Review,
    Sentence,
    Smartphone,
)
from app.nlp.aspects import aspects_for

logger = get_logger(__name__)

NEUTRAL_PRIOR = 0.5


def _counts_by_phone_aspect(
    db: Session, phone_ids: list[int] | None
) -> dict[tuple[int, str], dict[str, int]]:
    """Aggregate (phone, aspect, sentiment) counts in a single grouped query."""
    stmt = select(
        AspectSentiment.smartphone_id,
        AspectSentiment.aspect,
        AspectSentiment.sentiment,
        func.count(AspectSentiment.id),
    ).group_by(
        AspectSentiment.smartphone_id, AspectSentiment.aspect, AspectSentiment.sentiment
    )
    if phone_ids:
        stmt = stmt.where(AspectSentiment.smartphone_id.in_(phone_ids))

    counts: dict[tuple[int, str], dict[str, int]] = defaultdict(
        lambda: {"positive": 0, "negative": 0, "neutral": 0}
    )
    for phone_id, aspect, sentiment, count in db.execute(stmt).all():
        if sentiment in ("positive", "negative", "neutral"):
            counts[(phone_id, aspect)][sentiment] = count
    return counts


def _dominant_method(db: Session, phone_ids: list[int] | None) -> dict[int, str]:
    stmt = select(
        AspectSentiment.smartphone_id, AspectSentiment.method, func.count(AspectSentiment.id)
    ).group_by(AspectSentiment.smartphone_id, AspectSentiment.method)
    if phone_ids:
        stmt = stmt.where(AspectSentiment.smartphone_id.in_(phone_ids))

    best: dict[int, tuple[str, int]] = {}
    for phone_id, method, count in db.execute(stmt).all():
        current = best.get(phone_id)
        if current is None or count > current[1]:
            best[phone_id] = (method, count)
    return {phone_id: method for phone_id, (method, _) in best.items()}


def recompute_aspect_scores(
    db: Session,
    phone_ids: list[int] | None = None,
    settings: Settings | None = None,
) -> dict[int, dict[str, float]]:
    """Rebuild the AspectScore table. Returns {phone_id: {aspect: score}}."""
    settings = settings or get_settings()
    allowed = aspects_for(settings.aspect_set)
    neutral_weight = settings.neutral_weight
    k = max(0.0, settings.shrinkage_strength)

    # Corpus means are always computed over every phone, so shrinkage targets stay
    # stable even when only a subset of phones is being rescored.
    all_counts = _counts_by_phone_aspect(db, None)

    raw_scores: dict[tuple[int, str], float] = {}
    for key, sentiment_counts in all_counts.items():
        total = sum(sentiment_counts.values())
        if total:
            raw_scores[key] = (
                sentiment_counts["positive"] + neutral_weight * sentiment_counts["neutral"]
            ) / total

    corpus_mean: dict[str, float] = {}
    grouped: dict[str, list[float]] = defaultdict(list)
    for (_phone_id, aspect), score in raw_scores.items():
        grouped[aspect].append(score)
    for aspect, values in grouped.items():
        corpus_mean[aspect] = sum(values) / len(values) if values else NEUTRAL_PRIOR

    target_ids = phone_ids or sorted({phone_id for phone_id, _ in all_counts})
    methods = _dominant_method(db, phone_ids)
    now = datetime.now(timezone.utc)

    existing_rows = {}
    if target_ids:
        rows = db.scalars(
            select(AspectScore).where(AspectScore.smartphone_id.in_(target_ids))
        ).all()
        existing_rows = {(row.smartphone_id, row.aspect): row for row in rows}

    output: dict[int, dict[str, float]] = defaultdict(dict)

    for phone_id in target_ids:
        for aspect in allowed:
            key = (phone_id, aspect)
            sentiment_counts = all_counts.get(key)
            if not sentiment_counts:
                # No mentions: drop any stale row so the API reports null, not 0.
                stale = existing_rows.get(key)
                if stale is not None:
                    db.delete(stale)
                continue

            mentions = sum(sentiment_counts.values())
            raw = raw_scores[key]
            prior = corpus_mean.get(aspect, NEUTRAL_PRIOR)
            score = (mentions * raw + k * prior) / (mentions + k) if (mentions + k) else raw

            row = existing_rows.get(key)
            if row is None:
                row = AspectScore(smartphone_id=phone_id, aspect=aspect)
                db.add(row)
                existing_rows[key] = row

            row.positive_count = sentiment_counts["positive"]
            row.negative_count = sentiment_counts["negative"]
            row.neutral_count = sentiment_counts["neutral"]
            row.mention_count = mentions
            row.raw_score = round(raw, 4)
            row.score = round(score, 4)
            row.confidence = round(mentions / (mentions + settings.min_mentions_for_score), 4)
            row.method = methods.get(phone_id)
            row.pipeline_version = settings.pipeline_version
            row.computed_at = now

            output[phone_id][aspect] = row.score

    db.flush()
    logger.info("Recomputed aspect scores for %s phone(s).", len(target_ids))
    return dict(output)


# --------------------------------------------------------------------------- #
# Numeric price -> affordability index
# --------------------------------------------------------------------------- #
def latest_prices(db: Session, phone_ids: list[int] | None = None) -> dict[int, tuple[float, str | None]]:
    """Most recent price observation per phone."""
    newest = (
        select(
            PriceObservation.smartphone_id,
            func.max(PriceObservation.observed_at).label("observed_at"),
        )
        .where(PriceObservation.price.is_not(None))
        .group_by(PriceObservation.smartphone_id)
        .subquery()
    )
    stmt = select(
        PriceObservation.smartphone_id, PriceObservation.price, PriceObservation.currency
    ).join(
        newest,
        (PriceObservation.smartphone_id == newest.c.smartphone_id)
        & (PriceObservation.observed_at == newest.c.observed_at),
    )
    if phone_ids:
        stmt = stmt.where(PriceObservation.smartphone_id.in_(phone_ids))

    result: dict[int, tuple[float, str | None]] = {}
    for phone_id, price, currency in db.execute(stmt).all():
        if price is not None:
            result[phone_id] = (float(price), currency)
    return result


def affordability_index(
    prices: dict[int, float],
    currencies: dict[int, str | None] | None = None,
) -> dict[int, float]:
    """Map numeric prices to [0, 1] where 1 is the cheapest phone in the set.

    Kept separate from the review-derived `price` aspect: one measures what
    buyers *say* about value, the other is the objective cost.

    Normalisation happens **within each currency**. Amazon localises prices to
    the delivery country, so a corpus can mix e.g. LKR and USD figures; min-max
    scaling those together pins the LKR phone at 0 and squashes every USD phone
    into the top of the range, destroying the dimension. Phones that are alone
    in their currency are omitted rather than scored, because there is nothing
    to compare them against; callers treat a missing value as unknown.
    """
    if not prices:
        return {}

    groups: dict[str | None, dict[int, float]] = defaultdict(dict)
    for phone_id, price in prices.items():
        groups[(currencies or {}).get(phone_id)][phone_id] = price

    index: dict[int, float] = {}
    for group in groups.values():
        if len(group) < 2:
            continue
        values = list(group.values())
        low, high = min(values), max(values)
        if high - low < 1e-9:
            index.update({phone_id: 1.0 for phone_id in group})
            continue
        index.update(
            {
                phone_id: round((high - price) / (high - low), 4)
                for phone_id, price in group.items()
            }
        )
    return index


# --------------------------------------------------------------------------- #
# Corpus statistics
# --------------------------------------------------------------------------- #
def corpus_stats(db: Session, settings: Settings | None = None) -> dict[str, object]:
    settings = settings or get_settings()

    phones = db.scalar(select(func.count(Smartphone.id))) or 0
    reviews_total = db.scalar(select(func.count(Review.id))) or 0
    sentences = db.scalar(select(func.count(Sentence.id))) or 0
    aspect_sentiments = db.scalar(select(func.count(AspectSentiment.id))) or 0

    exclusion_rows = db.execute(
        select(Review.excluded_reason, func.count(Review.id))
        .where(Review.excluded_reason.is_not(None))
        .group_by(Review.excluded_reason)
    ).all()
    exclusion_breakdown = {reason: count for reason, count in exclusion_rows}
    reviews_excluded = sum(exclusion_breakdown.values())

    sentiment_rows = db.execute(
        select(AspectSentiment.sentiment, func.count(AspectSentiment.id)).group_by(
            AspectSentiment.sentiment
        )
    ).all()
    aspect_rows = db.execute(
        select(AspectSentiment.aspect, func.count(AspectSentiment.id)).group_by(
            AspectSentiment.aspect
        )
    ).all()

    analysed_reviews = (
        db.scalar(select(func.count(Review.id)).where(Review.processed_at.is_not(None))) or 0
    )

    # Amazon localises prices to the delivery country, so a corpus can silently
    # end up mixing currencies, which makes cross-phone price comparison invalid.
    currency_rows = db.execute(
        select(PriceObservation.currency, func.count(func.distinct(PriceObservation.smartphone_id)))
        .group_by(PriceObservation.currency)
    ).all()
    currency_breakdown = {(currency or "unknown"): count for currency, count in currency_rows}
    sources = sorted(
        row[0] for row in db.execute(select(Smartphone.source).distinct()).all() if row[0]
    )

    return {
        "phones": phones,
        "reviews_total": reviews_total,
        "reviews_usable": reviews_total - reviews_excluded,
        "reviews_excluded": reviews_excluded,
        "exclusion_breakdown": exclusion_breakdown,
        "sentences": sentences,
        "aspect_sentiments": aspect_sentiments,
        "sentiment_distribution": {s: c for s, c in sentiment_rows},
        "aspect_distribution": {a: c for a, c in aspect_rows},
        "mean_sentences_per_review": round(sentences / analysed_reviews, 2)
        if analysed_reviews
        else 0.0,
        "mean_aspects_per_sentence": round(aspect_sentiments / sentences, 2) if sentences else 0.0,
        "pipeline_version": settings.pipeline_version,
        "absa_engine": settings.resolved_absa_engine(),
        "currency_breakdown": currency_breakdown,
        "sources": sources,
        "has_demo_data": "demo" in sources,
    }


# --------------------------------------------------------------------------- #
# Internal validity check: star rating vs ABSA sentiment
# --------------------------------------------------------------------------- #
def validation_report(db: Session, settings: Settings | None = None) -> dict[str, object]:
    """Compare each review's star rating with the polarity of its own aspects.

    The star rating is an independent human signal that the pipeline never sees,
    so agreement between the two is evidence that the ABSA step is working. It is
    a weak label (a 4-star review can still pan the camera), hence "agreement"
    rather than "accuracy".
    """
    settings = settings or get_settings()
    neutral_weight = settings.neutral_weight

    rows = db.execute(
        select(
            Review.id,
            Review.rating,
            AspectSentiment.sentiment,
            func.count(AspectSentiment.id),
        )
        .join(AspectSentiment, AspectSentiment.review_id == Review.id)
        .where(Review.rating.is_not(None))
        .group_by(Review.id, Review.rating, AspectSentiment.sentiment)
    ).all()

    per_review: dict[int, dict[str, float]] = defaultdict(
        lambda: {"rating": 0.0, "positive": 0, "negative": 0, "neutral": 0}
    )
    for review_id, rating, sentiment, count in rows:
        entry = per_review[review_id]
        entry["rating"] = float(rating)
        if sentiment in ("positive", "negative", "neutral"):
            entry[sentiment] = count

    compared = 0
    agreements = 0
    absolute_errors: list[float] = []
    by_rating: dict[str, dict[str, float]] = {}
    bucket: dict[int, list[tuple[bool, float]]] = defaultdict(list)

    for entry in per_review.values():
        total = entry["positive"] + entry["negative"] + entry["neutral"]
        if not total:
            continue

        absa_score = (entry["positive"] + neutral_weight * entry["neutral"]) / total
        # Stars 1..5 -> [0, 1]
        rating_score = (entry["rating"] - 1.0) / 4.0

        absa_label = "positive" if absa_score > 0.6 else "negative" if absa_score < 0.4 else "neutral"
        rating_label = (
            "positive" if entry["rating"] >= 4 else "negative" if entry["rating"] <= 2 else "neutral"
        )

        agreed = absa_label == rating_label
        compared += 1
        agreements += int(agreed)
        error = abs(absa_score - rating_score)
        absolute_errors.append(error)
        bucket[int(round(entry["rating"]))].append((agreed, error))

    for stars, values in sorted(bucket.items()):
        by_rating[str(stars)] = {
            "reviews": len(values),
            "agreement_rate": round(sum(1 for a, _ in values if a) / len(values), 4),
            "mean_absolute_error": round(sum(e for _, e in values) / len(values), 4),
        }

    return {
        "reviews_compared": compared,
        "agreement_rate": round(agreements / compared, 4) if compared else 0.0,
        "mean_absolute_error": round(sum(absolute_errors) / len(absolute_errors), 4)
        if absolute_errors
        else 0.0,
        "by_rating": by_rating,
        "note": (
            "Star ratings are a weak, review-level label while ABSA output is "
            "aspect-level, so perfect agreement is neither expected nor desirable. "
            "Use this as a sanity check for systematic polarity bias, not as accuracy."
        ),
    }
