"""Viva / thesis evidence: pipeline stages, sample walkthrough, and CSV packs.

Produces a fixed folder ``data/exports/evidence/`` so examiners can be shown
the same files every time (preprocessed reviews, sentences, ABSA, scores).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.models.entities import (
    AspectScore,
    AspectSentiment,
    Review,
    Sentence,
    Smartphone,
)
from app.models.schemas import RecommendRequest
from app.services.recommender import build_feature_vectors, recommend

PIPELINE_STAGES: list[dict[str, str]] = [
    {
        "id": "raw",
        "title": "Raw Amazon review",
        "file": "01_raw_reviews.csv",
        "what": "Original review text stored from Amazon (title + body).",
        "show": "Open CSV → columns title, body, rating, brand/phone.",
    },
    {
        "id": "preprocess",
        "title": "Preprocessing decision",
        "file": "02_preprocessed_reviews.csv",
        "what": "Each review is kept or dropped (language, spam, length, etc.).",
        "show": "Filter decision=kept vs excluded_reason — this is your preprocessed dataset audit trail.",
    },
    {
        "id": "kept",
        "title": "Kept reviews (analysis set)",
        "file": "03_kept_reviews.csv",
        "what": "Only English usable reviews that enter ABSA.",
        "show": "This is the file to open if they ask for the preprocessed dataset.",
    },
    {
        "id": "sentences",
        "title": "Segmented sentence",
        "file": "04_sentences.csv",
        "what": "Each kept review is split into sentences.",
        "show": "Show review_id → several sentence rows.",
    },
    {
        "id": "aspects",
        "title": "Detected aspect + sentiment",
        "file": "05_aspect_sentiments.csv",
        "what": "ABSA labels: aspect (camera/battery/…) and positive/neutral/negative.",
        "show": "Pick a sentence_id and show aspect + sentiment + opinion_term.",
    },
    {
        "id": "scores",
        "title": "Feature score",
        "file": "06_aspect_scores.csv",
        "what": "Per-phone scores aggregated from all aspect mentions.",
        "show": "One phone × six aspects with score and mention_count.",
    },
    {
        "id": "matrix",
        "title": "Feature matrix",
        "file": "07_feature_matrix.csv",
        "what": "Wide table used by the recommender (all phones × aspects).",
        "show": "Scroll to a phone row and point at camera/battery/… columns.",
    },
    {
        "id": "rank",
        "title": "User-weighted ranking",
        "file": "08_sample_ranking.csv",
        "what": "Example ranking: user weights × feature scores → Top phones.",
        "show": "contribution columns = score × weight (explainable ranking).",
    },
]


def evidence_dir(settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    path = settings.exports_dir / "evidence"
    path.mkdir(parents=True, exist_ok=True)
    return path


def brand_distribution(db: Session) -> list[dict[str, Any]]:
    rows = db.execute(
        select(Smartphone.brand, func.count())
        .where(Smartphone.brand.isnot(None))
        .group_by(Smartphone.brand)
        .order_by(func.count().desc(), Smartphone.brand)
    ).all()
    return [{"brand": b or "Unknown", "phones": int(c)} for b, c in rows]


def funnel_counts(db: Session) -> dict[str, int]:
    total_reviews = db.scalar(select(func.count()).select_from(Review)) or 0
    spam = db.scalar(select(func.count()).select_from(Review).where(Review.is_spam.is_(True))) or 0
    excluded = (
        db.scalar(
            select(func.count())
            .select_from(Review)
            .where(Review.excluded_reason.isnot(None))
        )
        or 0
    )
    kept = (
        db.scalar(
            select(func.count())
            .select_from(Review)
            .where(
                Review.is_duplicate.is_(False),
                Review.is_spam.is_(False),
                Review.excluded_reason.is_(None),
            )
        )
        or 0
    )
    sentences = db.scalar(select(func.count()).select_from(Sentence)) or 0
    aspects = db.scalar(select(func.count()).select_from(AspectSentiment)) or 0
    scores = db.scalar(select(func.count()).select_from(AspectScore)) or 0
    phones = db.scalar(select(func.count()).select_from(Smartphone)) or 0
    scored_phones = (
        db.scalar(select(func.count(func.distinct(AspectScore.smartphone_id)))) or 0
    )
    return {
        "phones": int(phones),
        "raw_reviews": int(total_reviews),
        "spam": int(spam),
        "excluded": int(excluded),
        "kept_reviews": int(kept),
        "sentences": int(sentences),
        "aspect_labels": int(aspects),
        "aspect_score_rows": int(scores),
        "scored_phones": int(scored_phones),
    }


def _review_decision(review: Review) -> str:
    if review.is_spam:
        return "excluded_spam"
    if review.is_duplicate:
        return "excluded_duplicate"
    if review.excluded_reason:
        return f"excluded_{review.excluded_reason}"
    return "kept"


def export_evidence_csvs(db: Session, settings: Settings | None = None) -> dict[str, str]:
    """Write fixed evidence CSVs (overwrite). Returns {stage_id or name: path}."""
    settings = settings or get_settings()
    out = evidence_dir(settings)
    written: dict[str, str] = {}

    phones = {
        p.id: p
        for p in db.execute(select(Smartphone)).scalars().all()
    }

    # --- reviews with decisions ---
    raw_rows: list[dict[str, Any]] = []
    pre_rows: list[dict[str, Any]] = []
    kept_rows: list[dict[str, Any]] = []
    for rev in db.execute(select(Review).order_by(Review.id)).scalars().all():
        phone = phones.get(rev.smartphone_id)
        decision = _review_decision(rev)
        base = {
            "review_id": rev.id,
            "smartphone_id": rev.smartphone_id,
            "brand": phone.brand if phone else None,
            "phone_name": (phone.canonical_name or phone.raw_title) if phone else None,
            "asin": phone.source_product_id if phone else None,
            "rating": rev.rating,
            "language": rev.language,
            "title": rev.title,
            "body": rev.body,
            "cleaned_body": rev.cleaned_body,
            "is_spam": bool(rev.is_spam),
            "is_duplicate": bool(rev.is_duplicate),
            "excluded_reason": rev.excluded_reason,
            "decision": decision,
        }
        raw_rows.append(
            {
                "review_id": rev.id,
                "smartphone_id": rev.smartphone_id,
                "brand": base["brand"],
                "phone_name": base["phone_name"],
                "asin": base["asin"],
                "rating": rev.rating,
                "language": rev.language,
                "title": rev.title,
                "body": rev.body,
            }
        )
        pre_rows.append(base)
        if decision == "kept":
            kept_rows.append(
                {
                    "review_id": rev.id,
                    "smartphone_id": rev.smartphone_id,
                    "brand": base["brand"],
                    "phone_name": base["phone_name"],
                    "rating": rev.rating,
                    "language": rev.language,
                    "title": rev.title,
                    "cleaned_body": rev.cleaned_body or rev.body,
                }
            )

    def write(name: str, rows: list[dict[str, Any]]) -> Path:
        path = out / name
        pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
        return path

    written["raw"] = str(write("01_raw_reviews.csv", raw_rows))
    written["preprocess"] = str(write("02_preprocessed_reviews.csv", pre_rows))
    written["kept"] = str(write("03_kept_reviews.csv", kept_rows))

    sent_rows = [
        {
            "sentence_id": s.id,
            "review_id": s.review_id,
            "smartphone_id": s.smartphone_id,
            "position": s.position,
            "text": s.text,
            "word_count": s.word_count,
        }
        for s in db.execute(select(Sentence).order_by(Sentence.id)).scalars().all()
    ]
    written["sentences"] = str(write("04_sentences.csv", sent_rows))

    as_rows = []
    for a in db.execute(select(AspectSentiment).order_by(AspectSentiment.id)).scalars().all():
        as_rows.append(
            {
                "aspect_sentiment_id": a.id,
                "smartphone_id": a.smartphone_id,
                "review_id": a.review_id,
                "sentence_id": a.sentence_id,
                "aspect": a.aspect,
                "sentiment": a.sentiment,
                "confidence": a.confidence,
                "opinion_term": a.opinion_term,
                "method": a.method,
            }
        )
    written["aspects"] = str(write("05_aspect_sentiments.csv", as_rows))

    score_rows = [
        {
            "smartphone_id": s.smartphone_id,
            "brand": phones[s.smartphone_id].brand if s.smartphone_id in phones else None,
            "phone_name": (
                phones[s.smartphone_id].canonical_name or phones[s.smartphone_id].raw_title
            )
            if s.smartphone_id in phones
            else None,
            "aspect": s.aspect,
            "positive_count": s.positive_count,
            "neutral_count": s.neutral_count,
            "negative_count": s.negative_count,
            "mention_count": s.mention_count,
            "raw_score": s.raw_score,
            "score": s.score,
            "confidence": s.confidence,
        }
        for s in db.execute(
            select(AspectScore).order_by(AspectScore.smartphone_id, AspectScore.aspect)
        )
        .scalars()
        .all()
    ]
    written["scores"] = str(write("06_aspect_scores.csv", score_rows))

    vectors = build_feature_vectors(db, settings=settings)
    matrix_rows = []
    for v in vectors:
        row: dict[str, Any] = {
            "smartphone_id": v.smartphone_id,
            "name": v.name,
            "brand": v.brand,
            "price": v.price,
            "review_count": v.review_count,
        }
        for aspect, score in (v.scores or {}).items():
            row[aspect] = score
            row[f"{aspect}_mentions"] = (v.mentions or {}).get(aspect, 0)
        matrix_rows.append(row)
    written["matrix"] = str(write("07_feature_matrix.csv", matrix_rows))

    # Sample ranking for viva (balanced-ish weights)
    try:
        sample = recommend(
            db,
            RecommendRequest(
                weights={
                    "camera": 0.2,
                    "battery": 0.2,
                    "display": 0.15,
                    "performance": 0.15,
                    "design": 0.1,
                    "price": 0.2,
                },
                top_k=10,
                min_reviews=0,
                apply_shrinkage=True,
                method="weighted",
            ),
            settings=settings,
        )
        rank_rows = []
        for item in sample.results:
            row = {
                "rank": item.rank,
                "smartphone_id": item.smartphone_id,
                "name": item.name,
                "brand": item.brand,
                "final_score": item.final_score,
                "explanation": item.explanation,
            }
            for c in item.breakdown:
                row[f"{c.aspect}_score"] = c.score
                row[f"{c.aspect}_weight"] = c.weight
                row[f"{c.aspect}_contribution"] = c.contribution
            rank_rows.append(row)
        written["rank"] = str(write("08_sample_ranking.csv", rank_rows))
    except Exception:
        written["rank"] = str(write("08_sample_ranking.csv", []))

    brands = brand_distribution(db)
    written["brands"] = str(write("09_brand_distribution.csv", brands))

    funnel = funnel_counts(db)
    written["funnel"] = str(
        write("10_corpus_funnel.csv", [{"stage": k, "count": v} for k, v in funnel.items()])
    )

    return written


def walkthrough_example(db: Session, settings: Settings | None = None) -> dict[str, Any] | None:
    """One real review traced through every pipeline stage (for on-screen demo)."""
    settings = settings or get_settings()

    # Prefer a kept review that has aspect labels.
    row = db.execute(
        select(AspectSentiment.review_id, func.count())
        .group_by(AspectSentiment.review_id)
        .order_by(func.count().desc())
        .limit(1)
    ).first()
    if not row:
        return None
    review_id = int(row[0])

    review = db.get(Review, review_id)
    if review is None:
        return None
    phone = db.get(Smartphone, review.smartphone_id)
    sentences = (
        db.execute(
            select(Sentence)
            .where(Sentence.review_id == review_id)
            .order_by(Sentence.position)
        )
        .scalars()
        .all()
    )
    aspects = (
        db.execute(
            select(AspectSentiment)
            .where(AspectSentiment.review_id == review_id)
            .order_by(AspectSentiment.id)
        )
        .scalars()
        .all()
    )
    scores = (
        db.execute(
            select(AspectScore)
            .where(AspectScore.smartphone_id == review.smartphone_id)
            .order_by(AspectScore.aspect)
        )
        .scalars()
        .all()
    )

    # Tiny ranking contribution for this phone under equal weights
    contrib = []
    if scores:
        w = 1.0 / max(len(scores), 1)
        for s in scores:
            contrib.append(
                {
                    "aspect": s.aspect,
                    "score": round(float(s.score or 0), 4),
                    "weight": round(w, 4),
                    "contribution": round(float(s.score or 0) * w, 4),
                }
            )

    return {
        "phone": {
            "id": phone.id if phone else review.smartphone_id,
            "name": (phone.canonical_name or phone.raw_title) if phone else None,
            "brand": phone.brand if phone else None,
        },
        "raw_review": {
            "review_id": review.id,
            "rating": review.rating,
            "language": review.language,
            "title": review.title,
            "body": (review.body or "")[:600],
        },
        "preprocess": {
            "decision": _review_decision(review),
            "is_spam": bool(review.is_spam),
            "excluded_reason": review.excluded_reason,
            "cleaned_body": (review.cleaned_body or review.body or "")[:600],
        },
        "sentences": [
            {"sentence_id": s.id, "position": s.position, "text": s.text}
            for s in sentences[:8]
        ],
        "aspect_sentiments": [
            {
                "sentence_id": a.sentence_id,
                "aspect": a.aspect,
                "sentiment": a.sentiment,
                "opinion_term": a.opinion_term,
                "method": a.method,
            }
            for a in aspects[:12]
        ],
        "feature_scores": [
            {
                "aspect": s.aspect,
                "score": round(float(s.score or 0), 4),
                "mentions": s.mention_count,
                "positive": s.positive_count,
                "neutral": s.neutral_count,
                "negative": s.negative_count,
            }
            for s in scores
        ],
        "user_weighted_contribution": contrib,
        "explanation_hint": (
            f"This phone’s feature scores come from {len(aspects)} aspect labels "
            f"in review #{review.id}. Ranking multiplies each score by the user’s "
            f"priority weight for that feature."
        ),
    }


def stage_samples(db: Session, stage_id: str, limit: int = 8) -> list[dict[str, Any]]:
    """Small preview rows for the Evidence UI table."""
    if stage_id == "brands":
        return brand_distribution(db)

    if stage_id == "raw":
        rows = []
        for rev in db.execute(select(Review).order_by(Review.id).limit(limit)).scalars().all():
            phone = db.get(Smartphone, rev.smartphone_id)
            rows.append(
                {
                    "review_id": rev.id,
                    "phone": (phone.canonical_name if phone else rev.smartphone_id),
                    "rating": rev.rating,
                    "title": (rev.title or "")[:80],
                    "body": (rev.body or "")[:120],
                }
            )
        return rows

    if stage_id == "preprocess":
        # Mix kept + excluded
        kept = (
            db.execute(
                select(Review)
                .where(
                    Review.is_spam.is_(False),
                    Review.excluded_reason.is_(None),
                )
                .limit(max(limit // 2, 3))
            )
            .scalars()
            .all()
        )
        dropped = (
            db.execute(
                select(Review)
                .where(Review.excluded_reason.isnot(None))
                .limit(max(limit // 2, 3))
            )
            .scalars()
            .all()
        )
        rows = []
        for rev in list(kept) + list(dropped):
            rows.append(
                {
                    "review_id": rev.id,
                    "language": rev.language,
                    "decision": _review_decision(rev),
                    "excluded_reason": rev.excluded_reason or "",
                    "preview": ((rev.cleaned_body or rev.body or "")[:100]),
                }
            )
        return rows[:limit]

    if stage_id == "kept":
        rows = []
        for rev in (
            db.execute(
                select(Review)
                .where(
                    Review.is_spam.is_(False),
                    Review.excluded_reason.is_(None),
                )
                .limit(limit)
            )
            .scalars()
            .all()
        ):
            phone = db.get(Smartphone, rev.smartphone_id)
            rows.append(
                {
                    "review_id": rev.id,
                    "phone": phone.canonical_name if phone else rev.smartphone_id,
                    "language": rev.language,
                    "cleaned_body": (rev.cleaned_body or rev.body or "")[:140],
                }
            )
        return rows

    if stage_id == "sentences":
        return [
            {
                "sentence_id": s.id,
                "review_id": s.review_id,
                "position": s.position,
                "text": s.text[:160],
            }
            for s in db.execute(select(Sentence).order_by(Sentence.id).limit(limit))
            .scalars()
            .all()
        ]

    if stage_id == "aspects":
        return [
            {
                "sentence_id": a.sentence_id,
                "aspect": a.aspect,
                "sentiment": a.sentiment,
                "opinion_term": a.opinion_term,
            }
            for a in db.execute(
                select(AspectSentiment).order_by(AspectSentiment.id).limit(limit)
            )
            .scalars()
            .all()
        ]

    if stage_id == "scores":
        return [
            {
                "smartphone_id": s.smartphone_id,
                "aspect": s.aspect,
                "score": round(float(s.score or 0), 4),
                "mentions": s.mention_count,
            }
            for s in db.execute(
                select(AspectScore).order_by(AspectScore.smartphone_id).limit(limit)
            )
            .scalars()
            .all()
        ]

    if stage_id == "matrix":
        vectors = build_feature_vectors(db, settings=get_settings())[:limit]
        return [
            {
                "smartphone_id": v.smartphone_id,
                "name": v.name,
                "brand": v.brand,
                **{a: round(float(sc), 3) for a, sc in (v.scores or {}).items() if sc is not None},
            }
            for v in vectors
        ]

    if stage_id == "rank":
        path = evidence_dir() / "08_sample_ranking.csv"
        if path.is_file():
            df = pd.read_csv(path).head(limit)
            return df.to_dict(orient="records")
        return []

    return []


def evidence_summary(db: Session, settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    out = evidence_dir(settings)
    files = []
    for stage in PIPELINE_STAGES:
        path = out / stage["file"]
        files.append(
            {
                **stage,
                "exists": path.is_file(),
                "download_url": f"/evidence/download/{stage['file']}",
                "rows_preview_url": f"/evidence/stage/{stage['id']}",
            }
        )
    brand_csv = out / "09_brand_distribution.csv"
    funnel_csv = out / "10_corpus_funnel.csv"
    return {
        "funnel": funnel_counts(db),
        "brands": brand_distribution(db),
        "brand_chart_url": "/ui/charts/01_phone_brand_distribution.png",
        "stages": files,
        "extra_downloads": [
            {
                "title": "Brand distribution CSV",
                "file": "09_brand_distribution.csv",
                "exists": brand_csv.is_file(),
                "download_url": "/evidence/download/09_brand_distribution.csv",
            },
            {
                "title": "Corpus funnel CSV",
                "file": "10_corpus_funnel.csv",
                "exists": funnel_csv.is_file(),
                "download_url": "/evidence/download/10_corpus_funnel.csv",
            },
        ],
        "folder": str(out),
        "note": (
            "Click each pipeline step to see sample rows. Download CSVs to prove "
            "the full dataset to examiners. Regenerating refreshes files from the live DB."
        ),
    }
