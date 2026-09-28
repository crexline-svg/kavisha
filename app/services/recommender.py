"""Recommendation engine over the aspect score database.

Scoring:

    final = sum_a( w_a * score_a ) / sum_a( w_a )

Two details matter for defensibility:

1. Missing aspects are imputed with the mean score of that aspect across the
   candidate set and flagged `imputed`, instead of being treated as 0. Scoring a
   never-mentioned camera as zero would punish phones for having sparse reviews
   rather than for having a bad camera.
2. `coverage` reports the share of the requested weight that rests on real
   mentions, so a recommendation built mostly on imputed values is visibly weaker.
"""

from __future__ import annotations

from collections import defaultdict

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.models.entities import AspectScore, Review, Smartphone
from app.models.schemas import (
    AspectContribution,
    FeatureVector,
    Recommendation,
    RecommendRequest,
    RecommendResponse,
)
from app.nlp.aggregate import affordability_index, latest_prices
from app.nlp.aspects import ASPECT_LABELS, aspects_for

logger = get_logger(__name__)

AFFORDABILITY_KEY = "affordability"

STRENGTH_THRESHOLD = 0.70
WEAKNESS_THRESHOLD = 0.45


def _pct(score: float | None) -> str:
    if score is None:
        return "n/a"
    return f"{round(float(score) * 100)}%"


def build_explanation(
    *,
    phone_name: str,
    breakdown: list[AspectContribution],
    method: str,
    site_rating: float | None,
    site_rating_count: int | None,
) -> str:
    """Plain-language why this phone appears in the Top-N list."""
    if method == "star_rating":
        rating = f"{site_rating:.1f}" if site_rating is not None else "n/a"
        count = f"{site_rating_count:,}" if site_rating_count else "few"
        return (
            f"{phone_name} is listed here because of its Amazon star rating "
            f"({rating}/5 from {count} ratings). This is the rating-only baseline — "
            f"it does not use your feature priorities or review aspect scores."
        )

    weighted = [
        item
        for item in breakdown
        if item.weight > 0 and not item.imputed and item.score is not None
    ]
    weighted.sort(key=lambda c: c.contribution, reverse=True)
    if not weighted:
        return (
            f"{phone_name} is recommended from your priorities, but review evidence "
            f"for the weighted features is thin, so some scores were estimated."
        )

    top = weighted[:3]
    parts = [
        f"{ASPECT_LABELS.get(item.aspect, item.aspect)} ({_pct(item.score)} positive review score, "
        f"{round(item.weight * 100)}% of your priority)"
        for item in top
    ]
    if len(parts) == 1:
        detail = parts[0]
    elif len(parts) == 2:
        detail = f"{parts[0]} and {parts[1]}"
    else:
        detail = f"{parts[0]}, {parts[1]}, and {parts[2]}"

    lead = (
        f"{phone_name} is recommended based on your priorities. "
        f"The strongest matches from reviews are {detail}."
    )
    weak = [
        item
        for item in breakdown
        if item.weight > 0
        and not item.imputed
        and item.score is not None
        and item.score <= WEAKNESS_THRESHOLD
        and item.mention_count > 0
    ]
    if weak:
        w = weak[0]
        lead += (
            f" Note: {ASPECT_LABELS.get(w.aspect, w.aspect)} scores lower "
            f"({_pct(w.score)}) in reviews, so check that trade-off."
        )
    return lead


def _analysed_review_counts(db: Session, phone_ids: list[int] | None = None) -> dict[int, int]:
    stmt = (
        select(Review.smartphone_id, func.count(Review.id))
        .where(Review.processed_at.is_not(None), Review.excluded_reason.is_(None))
        .group_by(Review.smartphone_id)
    )
    if phone_ids:
        stmt = stmt.where(Review.smartphone_id.in_(phone_ids))
    return {phone_id: count for phone_id, count in db.execute(stmt).all()}


def build_feature_vectors(
    db: Session,
    phone_ids: list[int] | None = None,
    *,
    apply_shrinkage: bool = True,
    settings: Settings | None = None,
) -> list[FeatureVector]:
    """The 'Smartphone Feature Score Database' from the output layer."""
    settings = settings or get_settings()
    allowed = aspects_for(settings.aspect_set)

    phone_stmt = select(Smartphone).order_by(Smartphone.id)
    if phone_ids:
        phone_stmt = phone_stmt.where(Smartphone.id.in_(phone_ids))
    phones = list(db.scalars(phone_stmt).all())
    if not phones:
        return []

    ids = [phone.id for phone in phones]

    score_rows = db.scalars(select(AspectScore).where(AspectScore.smartphone_id.in_(ids))).all()
    by_phone: dict[int, dict[str, AspectScore]] = defaultdict(dict)
    for row in score_rows:
        by_phone[row.smartphone_id][row.aspect] = row

    prices = latest_prices(db, ids)
    affordability = affordability_index(
        {pid: value[0] for pid, value in prices.items()},
        {pid: value[1] for pid, value in prices.items()},
    )
    review_counts = _analysed_review_counts(db, ids)

    vectors: list[FeatureVector] = []
    for phone in phones:
        rows = by_phone.get(phone.id, {})
        scores: dict[str, float | None] = {}
        mentions: dict[str, int] = {}
        confidence: dict[str, float] = {}

        for aspect in allowed:
            row = rows.get(aspect)
            if row is None:
                scores[aspect] = None
                mentions[aspect] = 0
                confidence[aspect] = 0.0
                continue
            scores[aspect] = row.score if apply_shrinkage else row.raw_score
            mentions[aspect] = row.mention_count
            confidence[aspect] = row.confidence or 0.0

        price_value, currency = prices.get(phone.id, (None, None))

        vectors.append(
            FeatureVector(
                smartphone_id=phone.id,
                name=phone.display_name,
                brand=phone.brand,
                price=price_value,
                currency=currency,
                review_count=review_counts.get(phone.id, 0),
                mention_count=sum(mentions.values()),
                image_url=phone.image_url,
                product_url=phone.product_url,
                site_rating=phone.site_rating,
                site_rating_count=phone.site_rating_count,
                scores=scores,
                mentions=mentions,
                confidence=confidence,
                affordability=affordability.get(phone.id),
            )
        )

    return vectors


def _normalised_weights(raw: dict[str, float], allowed: tuple[str, ...]) -> dict[str, float]:
    valid_keys = set(allowed) | {AFFORDABILITY_KEY}

    weights = {
        key.strip().lower(): float(value)
        for key, value in (raw or {}).items()
        if key.strip().lower() in valid_keys and float(value) > 0
    }

    if not weights:
        # No preference expressed: weight every aspect equally.
        weights = {aspect: 1.0 for aspect in allowed}

    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()} if total else weights


def recommend(
    db: Session, request: RecommendRequest, settings: Settings | None = None
) -> RecommendResponse:
    settings = settings or get_settings()
    allowed = aspects_for(settings.aspect_set)
    weights = _normalised_weights(request.weights, allowed)

    vectors = build_feature_vectors(
        db,
        request.phone_ids or None,
        apply_shrinkage=request.apply_shrinkage,
        settings=settings,
    )

    # ---------------- filters ---------------- #
    candidates: list[FeatureVector] = []
    wanted_brands = {brand.strip().lower() for brand in request.brands if brand.strip()}

    for vector in vectors:
        if wanted_brands and (vector.brand or "").lower() not in wanted_brands:
            continue
        if request.min_reviews and vector.review_count < request.min_reviews:
            continue
        if request.budget_min is not None:
            if vector.price is None or vector.price < request.budget_min:
                continue
        if request.budget_max is not None:
            if vector.price is None or vector.price > request.budget_max:
                continue
        candidates.append(vector)

    if not candidates:
        return RecommendResponse(weights_used=weights, candidates_considered=0, results=[])

    # Imputation targets: mean of each aspect over the candidates that have it.
    observed: dict[str, list[float]] = defaultdict(list)
    for vector in candidates:
        for aspect, score in vector.scores.items():
            if score is not None:
                observed[aspect].append(score)
        if vector.affordability is not None:
            observed[AFFORDABILITY_KEY].append(vector.affordability)

    fallback = {
        aspect: sum(values) / len(values) for aspect, values in observed.items() if values
    }

    # ---------------- scoring ---------------- #
    scored: list[Recommendation] = []

    for vector in candidates:
        breakdown: list[AspectContribution] = []
        weighted_sum = 0.0
        real_weight = 0.0

        for aspect, weight in weights.items():
            if aspect == AFFORDABILITY_KEY:
                score = vector.affordability
                mention_count = 0
            else:
                score = vector.scores.get(aspect)
                mention_count = vector.mentions.get(aspect, 0)

            imputed = score is None
            if imputed:
                score = fallback.get(aspect, 0.5)
            else:
                real_weight += weight

            contribution = weight * float(score)
            weighted_sum += contribution

            breakdown.append(
                AspectContribution(
                    aspect=aspect,
                    score=round(float(score), 4),
                    weight=round(weight, 4),
                    contribution=round(contribution, 4),
                    mention_count=mention_count,
                    imputed=imputed,
                )
            )

        strengths = [
            ASPECT_LABELS.get(item.aspect, item.aspect)
            for item in sorted(breakdown, key=lambda c: (c.score or 0), reverse=True)
            if not item.imputed
            and (item.score or 0) >= STRENGTH_THRESHOLD
            and item.mention_count >= settings.min_mentions_for_score
        ][:3]

        weaknesses = [
            ASPECT_LABELS.get(item.aspect, item.aspect)
            for item in sorted(breakdown, key=lambda c: (c.score or 1))
            if not item.imputed
            and (item.score or 1) <= WEAKNESS_THRESHOLD
            and item.mention_count >= settings.min_mentions_for_score
        ][:3]

        method = getattr(request, "method", None) or "weighted"
        explanation = build_explanation(
            phone_name=vector.name,
            breakdown=breakdown,
            method=method,
            site_rating=vector.site_rating,
            site_rating_count=vector.site_rating_count,
        )

        # For the star baseline, final_score mirrors Amazon rating on a 0–1 scale.
        if method == "star_rating":
            final = (
                round((float(vector.site_rating) - 1.0) / 4.0, 4)
                if vector.site_rating is not None
                else 0.0
            )
        else:
            final = round(weighted_sum, 4)

        scored.append(
            Recommendation(
                rank=0,
                smartphone_id=vector.smartphone_id,
                name=vector.name,
                brand=vector.brand,
                price=vector.price,
                currency=vector.currency,
                image_url=vector.image_url,
                product_url=vector.product_url,
                site_rating=vector.site_rating,
                site_rating_count=vector.site_rating_count,
                final_score=final,
                review_count=vector.review_count,
                coverage=round(real_weight, 4),
                breakdown=sorted(breakdown, key=lambda c: c.contribution, reverse=True),
                strengths=strengths,
                weaknesses=weaknesses,
                explanation=explanation,
            )
        )

    method = getattr(request, "method", None) or "weighted"
    if method == "star_rating":
        scored.sort(
            key=lambda r: (
                r.site_rating is not None,
                r.site_rating or 0.0,
                r.site_rating_count or 0,
                r.review_count,
            ),
            reverse=True,
        )
    else:
        # Ties broken by evidence volume, so the better-supported phone wins.
        scored.sort(key=lambda r: (r.final_score, r.coverage, r.review_count), reverse=True)

    for position, item in enumerate(scored[: request.top_k], start=1):
        item.rank = position
        # Rebuild explanation with final rank context already in name; keep as-is.

    return RecommendResponse(
        weights_used={key: round(value, 4) for key, value in weights.items()},
        candidates_considered=len(candidates),
        results=scored[: request.top_k],
        method=method,
    )
