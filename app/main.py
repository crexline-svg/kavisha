"""FastAPI application wiring.

Endpoint map, mirroring the methodology:

    POST /scrape                    input layer      (phone details, prices, reviews)
    POST /analyze                   Steps 1-6        (clean, segment, ABSA, aggregate)
    GET  /features                  output layer     (feature score database)
    POST /recommend                 recommendation engine
    GET  /stats, /validation        dataset description + internal validity check
    POST /export                    CSV artefacts for the dissertation appendix
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response as StarletteResponse

from app import __version__
from app.api.routes import analysis, evidence, feedback, jobs, phones, recommend, reviews, scrape
from app.core.config import get_settings
from app.core.database import init_db
from app.core.logging import get_logger, setup_logging
from app.services.jobs import job_manager

setup_logging()
logger = get_logger(__name__)
settings = get_settings()

DESCRIPTION = """
Backend for an aspect-based-sentiment-analysis smartphone recommender.

**Typical research flow (offline corpus)**

1. `python run.py ingest-hf` loads McAuley-Lab/Amazon-Reviews-2023 into SQLite
   (phones + reviews, with Step 1 preprocessing).
2. `POST /analyze` (or the CLI) segments reviews, extracts aspects, classifies
   sentiment and aggregates per-phone scores.
3. `GET /features` returns the feature score database.
4. `POST /recommend` ranks phones against user-supplied aspect weights.

The website at `/ui/` only **reads** the database and collects user requirements.
Live Amazon scraping remains available via CLI (`python run.py scrape`) if needed,
but is not part of the UI.
"""


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    settings.ensure_dirs()
    init_db()
    from app.services.demo import ensure_demo_if_empty

    ensure_demo_if_empty(settings)
    logger.info(
        "%s v%s ready | db=%s | ABSA engine=%s | aspects=%s",
        settings.app_name,
        __version__,
        settings.sqlalchemy_url,
        settings.resolved_absa_engine(),
        settings.aspect_set,
    )
    if not settings.llm_available():
        logger.warning(
            "LLM_API_KEY is not set, so the rule-based lexicon baseline will be used. "
            "Set it in .env to run the LLM annotator described in the methodology."
        )
    try:
        yield
    finally:
        job_manager.shutdown()
        logger.info("Shutting down.")


app = FastAPI(
    title=settings.app_name,
    version=__version__,
    description=DESCRIPTION,
    lifespan=lifespan,
    contact={"name": "Research backend"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class _NoCacheUIMiddleware(BaseHTTPMiddleware):
    """Prevent stale app.js / app.css after UI updates during development."""

    async def dispatch(self, request: Request, call_next) -> StarletteResponse:
        response = await call_next(request)
        path = request.url.path
        if path.startswith("/ui/") and (
            path.endswith((".js", ".css", ".html")) or path in {"/ui", "/ui/"}
        ):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
        return response


app.add_middleware(_NoCacheUIMiddleware)

app.include_router(scrape.router)
app.include_router(analysis.router)
app.include_router(phones.router)
app.include_router(reviews.router)
app.include_router(recommend.router)
app.include_router(feedback.router)
app.include_router(evidence.router)
app.include_router(jobs.router)

# The dashboard is a dependency-free static bundle, mounted last so it cannot
# shadow any API route.
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.is_dir():

    @app.get("/ui/", include_in_schema=False)
    @app.get("/ui", include_in_schema=False)
    def ui_index() -> HTMLResponse:
        html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        html = html.replace("{{ASSET_VERSION}}", __version__)
        return HTMLResponse(
            html,
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )

    app.mount("/ui", StaticFiles(directory=str(STATIC_DIR), html=False), name="ui")


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse("/ui/" if STATIC_DIR.is_dir() else "/docs")


@app.get("/health", tags=["meta"], summary="Health and configuration snapshot")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "version": __version__,
        "pipeline_version": settings.pipeline_version,
        "marketplace": settings.marketplace,
        "absa_engine": settings.resolved_absa_engine(),
        "llm_model": settings.llm_model if settings.llm_available() else None,
        "aspect_set": settings.aspect_set,
        "headless": settings.headless,
        "logged_in_session": settings.storage_state_path.exists(),
    }
