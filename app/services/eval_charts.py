"""Research analysis charts shown on the Evaluation tab."""

from __future__ import annotations

import shutil
from pathlib import Path

from app.core.config import Settings, get_settings

# Canonical chart catalogue (filename → UI copy)
CHART_CATALOG: list[dict[str, str]] = [
    {
        "id": "brand",
        "file": "01_phone_brand_distribution.png",
        "title": "Phone distribution by brand",
        "caption": "How many smartphones from each brand are in the analysis corpus.",
    },
    {
        "id": "reviews",
        "file": "02_reviews_per_phone_hist.png",
        "title": "Reviews per phone",
        "caption": "How many kept English reviews each phone has after preprocessing.",
    },
    {
        "id": "aspects",
        "file": "03_aspect_sentiment_distribution.png",
        "title": "Aspect & sentiment distribution",
        "caption": "Which features appear in reviews and the positive / neutral / negative mix.",
    },
    {
        "id": "sentiment",
        "file": "04_overall_sentiment_pie.png",
        "title": "Overall ABSA sentiment mix",
        "caption": "Share of all aspect mentions labelled positive, neutral, or negative.",
    },
    {
        "id": "scores",
        "file": "05_sample_phone_feature_scores.png",
        "title": "Explainable feature-score profiles",
        "caption": "Six aspect scores for sample phones. Faded bars mean fewer review mentions (lower confidence).",
    },
    {
        "id": "agreement",
        "file": "06_star_text_agreement_by_rating.png",
        "title": "Star-text agreement by rating",
        "caption": "How often review-text polarity matches Amazon star ratings (ABSA sanity check).",
    },
    {
        "id": "priorities",
        "file": "07_ranking_priority_top1.png",
        "title": "Priorities change Top-1",
        "caption": "Different user priority profiles select different Top-1 phones.",
    },
    {
        "id": "baseline",
        "file": "08_proposed_vs_baseline_table.png",
        "title": "Proposed ranking vs Amazon stars",
        "caption": "Side-by-side Top-5: feature-weighted method versus Amazon star baseline.",
    },
]


def analysis_charts_dir(settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    return settings.data_dir / "exports" / "analysis" / "charts"


def static_charts_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "static" / "charts"


def sync_charts_to_static(settings: Settings | None = None) -> int:
    """Copy generated PNGs into app/static/charts so /ui/charts/ can serve them."""
    src = analysis_charts_dir(settings)
    dst = static_charts_dir()
    dst.mkdir(parents=True, exist_ok=True)
    if not src.is_dir():
        return 0
    copied = 0
    for item in CHART_CATALOG:
        source = src / item["file"]
        if source.is_file():
            shutil.copy2(source, dst / item["file"])
            copied += 1
    return copied


def list_evaluation_charts(settings: Settings | None = None) -> dict:
    """Return chart metadata for the Evaluation UI (only files that exist)."""
    settings = settings or get_settings()
    # Prefer synced static copies; fall back to analysis exports.
    static_dir = static_charts_dir()
    export_dir = analysis_charts_dir(settings)
    sync_charts_to_static(settings)

    charts: list[dict[str, str]] = []
    for item in CHART_CATALOG:
        static_path = static_dir / item["file"]
        export_path = export_dir / item["file"]
        if not static_path.is_file() and export_path.is_file():
            static_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(export_path, static_path)
        if static_path.is_file():
            charts.append(
                {
                    "id": item["id"],
                    "title": item["title"],
                    "caption": item["caption"],
                    "url": f"/ui/charts/{item['file']}",
                }
            )

    return {
        "count": len(charts),
        "charts": charts,
        "note": (
            "Explainable charts from the research analysis pipeline "
            "(corpus, ABSA, scores, ranking). Click a chart to enlarge."
            if charts
            else "No charts found yet. Run: python data/artifacts/generate_research_charts.py"
        ),
    }
