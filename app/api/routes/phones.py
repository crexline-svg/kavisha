"""Phone catalogue endpoints."""

from __future__ import annotations

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import Response
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import get_db
from app.models.entities import AspectScore, AspectSentiment, PriceObservation, Review, Sentence, Smartphone
from app.models.schemas import (
    AspectScoreOut,
    AspectSentimentOut,
    PhoneDetail,
    PhoneListResponse,
    PhoneSummary,
    PriceOut,
)
from app.nlp.aggregate import latest_prices

router = APIRouter(prefix="/phones", tags=["phones"])

_IMAGE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; SmartphoneRecommender/1.0)",
    "Referer": "https://www.amazon.com/",
}


def _review_counts(db: Session, phone_ids: list[int]) -> tuple[dict[int, int], dict[int, int]]:
    if not phone_ids:
        return {}, {}

    totals = {
        phone_id: count
        for phone_id, count in db.execute(
            select(Review.smartphone_id, func.count(Review.id))
            .where(Review.smartphone_id.in_(phone_ids))
            .group_by(Review.smartphone_id)
        ).all()
    }
    analysed = {
        phone_id: count
        for phone_id, count in db.execute(
            select(Review.smartphone_id, func.count(Review.id))
            .where(
                Review.smartphone_id.in_(phone_ids),
                Review.processed_at.is_not(None),
                Review.excluded_reason.is_(None),
            )
            .group_by(Review.smartphone_id)
        ).all()
    }
    return totals, analysed


def _phone_description(phone: Smartphone) -> str | None:
    """Prefer specs_raw description, then first feature bullet."""
    specs = phone.specs_raw or {}
    desc = specs.get("description") if isinstance(specs, dict) else None
    if isinstance(desc, str) and desc.strip():
        return desc.strip()
    bullets = phone.feature_bullets or []
    if bullets:
        first = str(bullets[0]).strip()
        if first:
            return first
    return None


def _to_summary(phone: Smartphone, prices, totals, analysed) -> PhoneSummary:  # noqa: ANN001
    price_value, currency = prices.get(phone.id, (None, None))
    summary = PhoneSummary.model_validate(phone)
    summary.latest_price = price_value
    summary.currency = currency
    summary.review_count = totals.get(phone.id, 0)
    summary.analyzed_review_count = analysed.get(phone.id, 0)
    summary.description = _phone_description(phone)
    return summary


@router.get("", response_model=PhoneListResponse, summary="List scraped phones")
def list_phones(
    db: Session = Depends(get_db),
    q: str | None = Query(None, description="Substring match on title, brand or model."),
    brand: str | None = Query(None),
    has_scores: bool | None = Query(None, description="Only phones with aspect scores."),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> PhoneListResponse:
    stmt = select(Smartphone)

    if q:
        pattern = f"%{q.strip()}%"
        stmt = stmt.where(
            or_(
                Smartphone.canonical_name.ilike(pattern),
                Smartphone.raw_title.ilike(pattern),
                Smartphone.brand.ilike(pattern),
                Smartphone.model.ilike(pattern),
            )
        )
    if brand:
        stmt = stmt.where(Smartphone.brand.ilike(brand.strip()))
    if has_scores:
        stmt = stmt.where(
            Smartphone.id.in_(select(AspectScore.smartphone_id).distinct())
        )

    total = db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    phones = list(
        db.scalars(stmt.order_by(Smartphone.id).offset(offset).limit(limit)).all()
    )

    ids = [phone.id for phone in phones]
    prices = latest_prices(db, ids)
    totals, analysed = _review_counts(db, ids)

    return PhoneListResponse(
        total=total,
        limit=limit,
        offset=offset,
        items=[_to_summary(phone, prices, totals, analysed) for phone in phones],
    )


@router.get("/{phone_id}/image", summary="Product image (proxied from Amazon CDN)")
def phone_image(phone_id: int, db: Session = Depends(get_db)) -> Response:
    """Serve product photos through the API so the browser is not blocked by hotlink rules."""
    phone = db.get(Smartphone, phone_id)
    if phone is None or not phone.image_url:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No image for this phone.")

    try:
        with httpx.Client(timeout=12.0, follow_redirects=True) as client:
            upstream = client.get(phone.image_url.strip(), headers=_IMAGE_HEADERS)
            upstream.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            f"Could not fetch product image: {exc}",
        ) from exc

    media_type = upstream.headers.get("content-type", "image/jpeg").split(";")[0]
    return Response(
        content=upstream.content,
        media_type=media_type,
        headers={"Cache-Control": "public, max-age=86400"},
    )


@router.get("/{phone_id}", response_model=PhoneDetail, summary="Phone details with scores")
def get_phone(phone_id: int, db: Session = Depends(get_db)) -> PhoneDetail:
    phone = db.get(Smartphone, phone_id)
    if phone is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No phone with id {phone_id}.")

    prices = latest_prices(db, [phone_id])
    totals, analysed = _review_counts(db, [phone_id])

    detail = PhoneDetail.model_validate(phone)
    price_value, currency = prices.get(phone_id, (None, None))
    detail.latest_price = price_value
    detail.currency = currency
    detail.review_count = totals.get(phone_id, 0)
    detail.analyzed_review_count = analysed.get(phone_id, 0)
    detail.description = _phone_description(phone)

    history = db.scalars(
        select(PriceObservation)
        .where(PriceObservation.smartphone_id == phone_id)
        .order_by(PriceObservation.observed_at.desc())
        .limit(50)
    ).all()
    detail.price_history = [PriceOut.model_validate(row) for row in history]

    scores = db.scalars(
        select(AspectScore)
        .where(AspectScore.smartphone_id == phone_id)
        .order_by(AspectScore.aspect)
    ).all()
    detail.aspect_scores = [AspectScoreOut.model_validate(row) for row in scores]

    return detail


@router.get(
    "/{phone_id}/scores",
    response_model=list[AspectScoreOut],
    summary="Aggregated aspect scores for one phone",
)
def get_phone_scores(phone_id: int, db: Session = Depends(get_db)) -> list[AspectScore]:
    if db.get(Smartphone, phone_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No phone with id {phone_id}.")
    return list(
        db.scalars(
            select(AspectScore)
            .where(AspectScore.smartphone_id == phone_id)
            .order_by(AspectScore.aspect)
        ).all()
    )


@router.get(
    "/{phone_id}/opinions",
    response_model=list[AspectSentimentOut],
    summary="Individual aspect-sentiment records with their source sentence",
)
def get_phone_opinions(
    phone_id: int,
    db: Session = Depends(get_db),
    aspect: str | None = Query(None),
    sentiment: str | None = Query(None, pattern="^(positive|negative|neutral)$"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> list[AspectSentimentOut]:
    """Evidence behind a score: the exact sentences that produced each label."""
    if db.get(Smartphone, phone_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No phone with id {phone_id}.")

    stmt = (
        select(AspectSentiment, Sentence.text)
        .join(Sentence, Sentence.id == AspectSentiment.sentence_id)
        .where(AspectSentiment.smartphone_id == phone_id)
    )
    if aspect:
        stmt = stmt.where(AspectSentiment.aspect == aspect.strip().lower())
    if sentiment:
        stmt = stmt.where(AspectSentiment.sentiment == sentiment)

    rows = db.execute(
        stmt.order_by(AspectSentiment.id).offset(offset).limit(limit)
    ).all()

    output: list[AspectSentimentOut] = []
    for record, sentence_text in rows:
        item = AspectSentimentOut.model_validate(record)
        item.sentence_text = sentence_text
        output.append(item)
    return output


@router.delete("/{phone_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete a phone")
def delete_phone(phone_id: int, db: Session = Depends(get_db)) -> None:
    phone = db.get(Smartphone, phone_id)
    if phone is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No phone with id {phone_id}.")
    db.delete(phone)
    db.commit()


@router.get("/meta/brands", response_model=list[dict], summary="Brands present in the corpus")
def list_brands(db: Session = Depends(get_db)) -> list[dict]:
    rows = db.execute(
        select(Smartphone.brand, func.count(Smartphone.id))
        .where(Smartphone.brand.is_not(None))
        .group_by(Smartphone.brand)
        .order_by(func.count(Smartphone.id).desc())
    ).all()
    return [{"brand": brand, "phones": count} for brand, count in rows]
