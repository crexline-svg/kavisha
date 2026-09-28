"""Evidence endpoints for viva demos: pipeline walkthrough + CSV downloads."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import get_db
from app.services.pipeline_evidence import (
    PIPELINE_STAGES,
    evidence_dir,
    evidence_summary,
    export_evidence_csvs,
    stage_samples,
    walkthrough_example,
)

router = APIRouter(prefix="/evidence", tags=["evidence"])

_ALLOWED_FILES = {s["file"] for s in PIPELINE_STAGES} | {
    "09_brand_distribution.csv",
    "10_corpus_funnel.csv",
}


@router.get("/summary", summary="Pipeline funnel, brands, and CSV catalogue")
def get_evidence_summary(db: Session = Depends(get_db)) -> dict:
    return evidence_summary(db, get_settings())


@router.get("/walkthrough", summary="One real review traced through every stage")
def get_walkthrough(db: Session = Depends(get_db)) -> dict:
    example = walkthrough_example(db, get_settings())
    if example is None:
        raise HTTPException(status_code=404, detail="No analysed reviews available yet.")
    return example


@router.get("/stage/{stage_id}", summary="Sample rows for one pipeline stage")
def get_stage_samples(
    stage_id: str,
    db: Session = Depends(get_db),
    limit: int = Query(8, ge=1, le=50),
) -> dict:
    known = {s["id"] for s in PIPELINE_STAGES} | {"brands"}
    if stage_id not in known:
        raise HTTPException(status_code=404, detail=f"Unknown stage: {stage_id}")
    rows = stage_samples(db, stage_id, limit=limit)
    meta = next((s for s in PIPELINE_STAGES if s["id"] == stage_id), None)
    return {"stage": stage_id, "meta": meta, "count": len(rows), "rows": rows}


@router.post("/export", summary="Regenerate evidence CSV pack from the live database")
def post_export_evidence(db: Session = Depends(get_db)) -> dict:
    written = export_evidence_csvs(db, get_settings())
    return {
        "ok": True,
        "folder": str(evidence_dir(get_settings())),
        "files": written,
        "count": len(written),
    }


@router.get("/download/{filename}", summary="Download one evidence CSV")
def download_evidence_file(filename: str) -> FileResponse:
    if filename not in _ALLOWED_FILES or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=404, detail="Unknown evidence file")
    path = evidence_dir(get_settings()) / filename
    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail="File not generated yet. Click Regenerate CSVs on the Evidence tab.",
        )
    return FileResponse(
        path,
        media_type="text/csv",
        filename=filename,
        headers={"Cache-Control": "no-cache"},
    )
