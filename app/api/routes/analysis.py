"""ABSA pipeline endpoints: run the analysis, inspect the output layer, export."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import get_db
from app.models.schemas import (
    AnalyzeRequest,
    CorpusStats,
    FeatureVector,
    JobAccepted,
    ValidationReport,
)
from app.nlp.aggregate import corpus_stats, recompute_aspect_scores, validation_report
from app.nlp.aspects import ASPECT_DEFINITIONS, ASPECT_LABELS, aspects_for
from app.services.analysis import run_analysis
from app.services.exports import export_dataset
from app.services.jobs import JobContext, job_manager
from app.services.recommender import build_feature_vectors

router = APIRouter(tags=["analysis"])


@router.post(
    "/analyze",
    response_model=JobAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Run the ABSA pipeline (Steps 1-6) over stored reviews",
)
def start_analysis(request: AnalyzeRequest) -> JobAccepted:
    """Queue the ABSA pipeline.

    Leave `phone_ids` empty to analyse every phone. Reviews already annotated are
    skipped unless `force` is true. Aspect scores are recomputed at the end.
    """
    settings = get_settings()

    def worker(ctx: JobContext) -> dict[str, Any]:
        ctx.note("Starting ABSA pipeline")
        return run_analysis(
            phone_ids=request.phone_ids or None,
            force=request.force,
            max_reviews_per_phone=request.max_reviews_per_phone,
            engine_name=request.engine,
            settings=settings,
            progress=ctx.progress,
        )

    job_id = job_manager.submit("analyze", request.model_dump(), worker)
    engine = request.engine or settings.resolved_absa_engine()

    return JobAccepted(
        job_id=job_id,
        status="pending",
        poll_url=f"/jobs/{job_id}",
        message=f"Analysis queued using the '{engine}' engine.",
    )


@router.post("/analyze/recompute-scores", summary="Re-aggregate scores without re-annotating")
def recompute_scores(db: Session = Depends(get_db)) -> dict[str, Any]:
    """Rebuild Step 6 only. Use after changing NEUTRAL_WEIGHT or SHRINKAGE_STRENGTH."""
    settings = get_settings()
    scores = recompute_aspect_scores(db, None, settings)
    db.commit()
    return {
        "phones_scored": len(scores),
        "neutral_weight": settings.neutral_weight,
        "shrinkage_strength": settings.shrinkage_strength,
    }


@router.get(
    "/features",
    response_model=list[FeatureVector],
    summary="The smartphone feature score database (output layer)",
)
def get_features(
    db: Session = Depends(get_db),
    phone_ids: list[int] | None = Query(None),
    raw: bool = Query(False, description="Return unshrunk raw scores instead."),
) -> list[FeatureVector]:
    return build_feature_vectors(
        db, phone_ids or None, apply_shrinkage=not raw, settings=get_settings()
    )


@router.get("/stats", response_model=CorpusStats, summary="Corpus statistics")
def get_stats(db: Session = Depends(get_db)) -> dict[str, Any]:
    return corpus_stats(db, get_settings())


@router.get(
    "/validation",
    response_model=ValidationReport,
    summary="Agreement between star ratings and ABSA output",
)
def get_validation(db: Session = Depends(get_db)) -> dict[str, Any]:
    return validation_report(db, get_settings())


@router.get("/aspects", summary="The active aspect taxonomy")
def get_aspects() -> list[dict[str, str]]:
    settings = get_settings()
    return [
        {
            "aspect": aspect,
            "label": ASPECT_LABELS.get(aspect, aspect),
            "definition": ASPECT_DEFINITIONS.get(aspect, ""),
        }
        for aspect in aspects_for(settings.aspect_set)
    ]


@router.post("/export", summary="Export every pipeline stage to CSV")
def export_csv(db: Session = Depends(get_db)) -> dict[str, Any]:
    files = export_dataset(db, settings=get_settings())
    return {"files": files, "count": len(files)}
