"""Scraping endpoints. Long-running work is delegated to background jobs."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.schemas import JobAccepted, ScrapeRequest
from app.scrapers.pipeline import scrape_to_db
from app.services.analysis import run_analysis
from app.services.jobs import JobContext, job_manager, run_coroutine_blocking

router = APIRouter(prefix="/scrape", tags=["scraping"])
logger = get_logger(__name__)


@router.post(
    "",
    response_model=JobAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Scrape phone details, prices and reviews",
)
def start_scrape(request: ScrapeRequest) -> JobAccepted:
    """Queue a scrape job.

    Give search terms in `queries`, known ASINs in `product_ids`, or both. The
    response returns immediately with a job id; poll `GET /jobs/{job_id}` for
    progress. When `analyze_after_scrape` is true the ABSA pipeline runs on the
    newly scraped reviews as soon as scraping finishes.
    """
    try:
        request.validate_targets()
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc

    settings = get_settings()

    def worker(ctx: JobContext) -> dict[str, Any]:
        ctx.note("Starting scrape")
        summary = run_coroutine_blocking(
            scrape_to_db(
                queries=request.queries,
                product_ids=request.product_ids,
                max_phones_per_query=request.max_phones_per_query,
                max_reviews_per_phone=request.max_reviews_per_phone,
                max_review_pages=request.max_review_pages,
                review_sort=request.review_sort,
                headless=request.headless,
                settings=settings,
                progress=ctx.progress,
            )
        )

        if request.analyze_after_scrape and summary.get("phone_ids"):
            ctx.note("Scrape done; running the ABSA pipeline")
            summary["analysis"] = run_analysis(
                phone_ids=summary["phone_ids"],
                settings=settings,
                progress=ctx.progress,
            )

        return summary

    job_id = job_manager.submit("scrape", request.model_dump(), worker)

    return JobAccepted(
        job_id=job_id,
        status="pending",
        poll_url=f"/jobs/{job_id}",
        message=(
            "Scrape queued. Poll the job for progress. Expect roughly "
            f"{settings.min_delay_s:.0f}-{settings.max_delay_s:.0f}s per page "
            "because of the politeness delay."
        ),
    )
