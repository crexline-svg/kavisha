"""Recommendation endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import get_db
from app.models.schemas import RecommendRequest, RecommendResponse
from app.services.recommender import recommend

router = APIRouter(prefix="/recommend", tags=["recommendation"])


@router.post("", response_model=RecommendResponse, summary="Rank phones by weighted aspects")
def get_recommendations(
    request: RecommendRequest, db: Session = Depends(get_db)
) -> RecommendResponse:
    """Rank phones against a user's aspect priorities.

    `weights` maps aspects to importance and is normalised internally, so
    `{"battery": 2, "camera": 1}` and `{"battery": 0.67, "camera": 0.33}` are
    equivalent. Core aspects include design and review-based price. Use budget
    min/max for list-price filters. Omit `weights` entirely to weight all
    aspects equally.

    Each result carries a `breakdown` showing how much every aspect contributed and
    whether the score was imputed, plus a `coverage` figure giving the share of the
    requested weight that is backed by real review mentions.
    """
    return recommend(db, request, get_settings())
