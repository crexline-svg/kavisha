"""Job status endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.entities import Job
from app.models.schemas import JobOut
from app.services.jobs import job_manager

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("", response_model=list[JobOut], summary="List recent jobs")
def list_jobs(
    db: Session = Depends(get_db),
    job_type: str | None = Query(None, description="Filter by 'scrape' or 'analyze'."),
    job_status: str | None = Query(None, alias="status"),
    limit: int = Query(20, ge=1, le=200),
) -> list[Job]:
    stmt = select(Job).order_by(Job.created_at.desc()).limit(limit)
    if job_type:
        stmt = stmt.where(Job.job_type == job_type)
    if job_status:
        stmt = stmt.where(Job.status == job_status)
    return list(db.scalars(stmt).all())


@router.get("/{job_id}", response_model=JobOut, summary="Poll a job")
def get_job(job_id: str, db: Session = Depends(get_db)) -> Job:
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No job with id {job_id}.")
    return job


@router.post("/{job_id}/cancel", summary="Request cancellation of a running job")
def cancel_job(job_id: str, db: Session = Depends(get_db)) -> dict[str, str]:
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No job with id {job_id}.")
    if job.status in ("completed", "failed"):
        return {"job_id": job_id, "status": job.status, "detail": "Job already finished."}

    requested = job_manager.cancel(job_id)
    return {
        "job_id": job_id,
        "status": job.status,
        "detail": (
            "Cancellation requested; the job stops after the current phone."
            if requested
            else "Job is not tracked by this process, so it cannot be cancelled."
        ),
    }
