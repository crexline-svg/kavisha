"""Background job runner.

Scraping a few hundred reviews takes minutes, far longer than an HTTP request
should last, so `POST /scrape` and `POST /analyze` return a job id immediately and
the client polls `GET /jobs/{id}`.

Jobs execute in a worker thread rather than on the event loop. That keeps the
synchronous SQLAlchemy calls off the API's loop, and it lets each job own a fresh
event loop for Playwright (which needs a subprocess-capable loop on Windows).
"""

from __future__ import annotations

import asyncio
import sys
import threading
import traceback
import uuid
from collections.abc import Callable, Coroutine
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

from app.core.database import session_scope
from app.core.logging import get_logger
from app.models.entities import Job

logger = get_logger(__name__)

PENDING, RUNNING, COMPLETED, FAILED = "pending", "running", "completed", "failed"


class JobContext:
    """Handed to a worker so it can publish progress."""

    def __init__(self, job_id: str) -> None:
        self.job_id = job_id
        self._cancelled = threading.Event()

    def progress(self, done: int, total: int, message: str | None = None) -> None:
        _patch_job(self.job_id, progress=done, total=total, message=message)

    def note(self, message: str) -> None:
        logger.info("[job %s] %s", self.job_id[:8], message)
        _patch_job(self.job_id, message=message)

    def cancel(self) -> None:
        self._cancelled.set()

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()


def _patch_job(job_id: str, **fields: Any) -> None:
    clean = {key: value for key, value in fields.items() if value is not None}
    if not clean:
        return
    try:
        with session_scope() as db:
            job = db.get(Job, job_id)
            if job is None:
                return
            for key, value in clean.items():
                setattr(job, key, value)
    except Exception as exc:  # a failed progress write must not kill the job
        logger.debug("Could not update job %s: %s", job_id, exc)


def run_coroutine_blocking(coro: Coroutine[Any, Any, Any]) -> Any:
    """Run a coroutine on a dedicated loop suitable for Playwright subprocesses."""
    if sys.platform == "win32":
        loop = asyncio.ProactorEventLoop()
    else:
        loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    finally:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception:
            pass
        asyncio.set_event_loop(None)
        loop.close()


class JobManager:
    def __init__(self, max_workers: int = 2) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="job")
        self._contexts: dict[str, JobContext] = {}
        self._lock = threading.Lock()

    def submit(
        self,
        job_type: str,
        params: dict[str, Any],
        worker: Callable[[JobContext], dict[str, Any]],
    ) -> str:
        job_id = str(uuid.uuid4())

        with session_scope() as db:
            db.add(
                Job(
                    id=job_id,
                    job_type=job_type,
                    status=PENDING,
                    params=params,
                    message="Queued",
                )
            )

        context = JobContext(job_id)
        with self._lock:
            self._contexts[job_id] = context

        self._executor.submit(self._run, context, worker)
        logger.info("Queued %s job %s", job_type, job_id[:8])
        return job_id

    def _run(self, context: JobContext, worker: Callable[[JobContext], dict[str, Any]]) -> None:
        job_id = context.job_id
        _patch_job(
            job_id,
            status=RUNNING,
            started_at=datetime.now(timezone.utc),
            message="Running",
        )
        try:
            result = worker(context) or {}
            _patch_job(
                job_id,
                status=COMPLETED,
                result=result,
                finished_at=datetime.now(timezone.utc),
                message="Completed",
            )
            logger.info("Job %s completed.", job_id[:8])
        except Exception as exc:  # noqa: BLE001
            detail = f"{type(exc).__name__}: {exc}"
            logger.error("Job %s failed: %s", job_id[:8], detail)
            logger.debug(traceback.format_exc())
            _patch_job(
                job_id,
                status=FAILED,
                error=detail,
                finished_at=datetime.now(timezone.utc),
                message="Failed",
            )
        finally:
            with self._lock:
                self._contexts.pop(job_id, None)

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            context = self._contexts.get(job_id)
        if context is None:
            return False
        context.cancel()
        _patch_job(job_id, message="Cancellation requested")
        return True

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


job_manager = JobManager()
