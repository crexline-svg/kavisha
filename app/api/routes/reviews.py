"""Review browsing endpoints (raw corpus inspection)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.entities import Review, Sentence, Smartphone
from app.models.schemas import ReviewListResponse, ReviewOut

router = APIRouter(prefix="/reviews", tags=["reviews"])


@router.get("", response_model=ReviewListResponse, summary="List reviews")
def list_reviews(
    db: Session = Depends(get_db),
    phone_id: int | None = Query(None),
    only_usable: bool = Query(True, description="Hide spam / filtered-out reviews."),
    analysed: bool | None = Query(None, description="Filter on whether ABSA has run."),
    min_rating: float | None = Query(None, ge=1, le=5),
    max_rating: float | None = Query(None, ge=1, le=5),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> ReviewListResponse:
    stmt = select(Review)

    if phone_id is not None:
        if db.get(Smartphone, phone_id) is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"No phone with id {phone_id}.")
        stmt = stmt.where(Review.smartphone_id == phone_id)
    if only_usable:
        stmt = stmt.where(Review.excluded_reason.is_(None), Review.is_spam.is_(False))
    if analysed is True:
        stmt = stmt.where(Review.processed_at.is_not(None))
    elif analysed is False:
        stmt = stmt.where(Review.processed_at.is_(None))
    if min_rating is not None:
        stmt = stmt.where(Review.rating >= min_rating)
    if max_rating is not None:
        stmt = stmt.where(Review.rating <= max_rating)

    total = db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    rows = db.scalars(stmt.order_by(Review.id).offset(offset).limit(limit)).all()

    return ReviewListResponse(
        total=total,
        limit=limit,
        offset=offset,
        items=[ReviewOut.model_validate(row) for row in rows],
    )


@router.get("/{review_id}", response_model=ReviewOut, summary="A single review")
def get_review(review_id: int, db: Session = Depends(get_db)) -> Review:
    review = db.get(Review, review_id)
    if review is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No review with id {review_id}.")
    return review


@router.get(
    "/{review_id}/sentences",
    summary="Segmented sentences of a review with their extracted aspects",
)
def get_review_sentences(review_id: int, db: Session = Depends(get_db)) -> list[dict]:
    """Shows Steps 2-4 for one review, which is handy for qualitative examples."""
    review = db.get(Review, review_id)
    if review is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No review with id {review_id}.")

    sentences = db.scalars(
        select(Sentence).where(Sentence.review_id == review_id).order_by(Sentence.position)
    ).all()

    return [
        {
            "position": sentence.position,
            "text": sentence.text,
            "word_count": sentence.word_count,
            "aspects": [
                {
                    "aspect": record.aspect,
                    "sentiment": record.sentiment,
                    "confidence": record.confidence,
                    "opinion_term": record.opinion_term,
                    "method": record.method,
                }
                for record in sentence.aspect_sentiments
            ],
        }
        for sentence in sentences
    ]
