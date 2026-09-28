"""Command-line entry point.

    python run.py init-db                       create the database
    python run.py seed-demo                     insert synthetic data (offline testing)
    python run.py purge-demo                    delete the synthetic data again
    python run.py prepare-csv                   Phase 1: full_reviews.csv + preprocess + ABSA (recommended)
    python run.py ingest-csv                    load data/full_reviews.csv only (English Step 1)
    python run.py prepare-corpus                Phase 1: load Amazon-Reviews-2023 + preprocess + ABSA (legacy)
    python run.py reset-corpus                  wipe ALL phones/reviews (clean slate)
    python run.py import-corpus PATH            import a pre-built HF folder + optional ABSA
    python run.py ingest-hf                     stream HF directly into the DB (same data, less portable)
    python run.py login                         save a signed-in browser session (optional live scrape)
    python run.py scrape -q "samsung galaxy s24" -q "iphone 15"
    python run.py analyze                       run Steps 1-6 over stored reviews
    python run.py pipeline -q "pixel 8"         scrape then analyse in one go
    python run.py website                       staged research website (dataset -> ABSA -> recommend)
    python run.py serve                         start the FastAPI server
    python run.py stats                         corpus statistics
    python run.py features                      print the feature score table
    python run.py recommend -w battery=0.5 -w camera=0.5
    python run.py sample-gold                   CSV template for manual ABSA labels
    python run.py evaluate-absa --gold PATH     score gold labels; show on Evaluation tab
    python run.py export                        write CSVs for the appendix
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from app.core.config import get_settings
from app.core.database import init_db, session_scope
from app.core.logging import setup_logging

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Smartphone ABSA recommendation research toolkit.",
)
console = Console()


def _bootstrap():
    setup_logging()
    settings = get_settings()
    settings.ensure_dirs()
    init_db()
    return settings


# --------------------------------------------------------------------------- #
@app.command("init-db")
def cmd_init_db() -> None:
    """Create the database schema."""
    settings = _bootstrap()
    console.print(f"[green]Database ready:[/green] {settings.sqlalchemy_url}")


@app.command("seed-demo")
def cmd_seed_demo(
    reviews: Annotated[int, typer.Option("--reviews", "-n", help="Reviews per phone.")] = 60,
    seed: Annotated[int, typer.Option(help="RNG seed for reproducibility.")] = 42,
) -> None:
    """Insert synthetic phones and reviews so the pipeline can be tested offline."""
    settings = _bootstrap()
    from app.services.demo import seed_demo_data

    with session_scope() as db:
        result = seed_demo_data(db, reviews_per_phone=reviews, seed=seed, settings=settings)

    console.print(
        f"[green]Seeded[/green] {result['reviews_inserted']} review(s) "
        f"across {result['phones_total']} demo phone(s)."
    )
    console.print("Next: [cyan]python run.py analyze[/cyan]")


@app.command("purge-demo")
def cmd_purge_demo(
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip the confirmation prompt.")] = False,
) -> None:
    """Delete the synthetic demo phones so only real scraped data remains."""
    settings = _bootstrap()
    from app.nlp.aggregate import recompute_aspect_scores
    from app.services.demo import purge_demo_data

    if not yes:
        typer.confirm(
            "Delete all synthetic demo phones, reviews and scores? Scraped data is untouched.",
            abort=True,
        )

    with session_scope() as db:
        result = purge_demo_data(db)
        recompute_aspect_scores(db, None, settings)

    if result["phones_deleted"]:
        console.print(
            f"[green]Purged[/green] {result['phones_deleted']} demo phone(s) and "
            f"{result['reviews_deleted']} review(s); aspect scores recomputed."
        )
    else:
        console.print("[yellow]No demo data found — nothing to purge.[/yellow]")


@app.command("reset-corpus")
def cmd_reset_corpus(
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip the confirmation prompt.")] = False,
) -> None:
    """Delete ALL phones/reviews/scores (demo + scrape + HF) so you can load a clean real corpus."""
    _bootstrap()
    from app.services.corpus_import import reset_corpus

    if not yes:
        typer.confirm(
            "Delete the ENTIRE corpus (every phone, review and score)? This cannot be undone.",
            abort=True,
        )

    with session_scope() as db:
        result = reset_corpus(db)

    console.print(f"[green]Reset[/green] — deleted {result['phones_deleted']} phone(s).")


@app.command("prepare-corpus")
def cmd_prepare_corpus(
    max_phones: Annotated[int, typer.Option("--max-phones", help="How many real phones to keep. 0 = all matching smartphones.")] = 0,
    max_reviews: Annotated[int, typer.Option("--max-reviews", help="Reviews per phone.")] = 60,
    min_ratings: Annotated[int, typer.Option("--min-ratings", help="Minimum Amazon rating count. 0 = no popularity cutoff.")] = 0,
    brand: Annotated[list[str] | None, typer.Option("--brand", "-b", help="Brand filter (repeatable).")] = None,
    max_review_scan: Annotated[int, typer.Option("--max-review-scan", help="Max review rows to scan. 0 = until phones are filled or the JSONL ends.")] = 0,
    out: Annotated[str, typer.Option("--out", help="Folder to write phones.jsonl / reviews.jsonl.")] = "data/exports/hf_corpus",
    analyze: Annotated[bool, typer.Option("--analyze/--no-analyze", help="Run Steps 2–6 after import.")] = True,
) -> None:
    """Phase 1 (local, no Colab): load McAuley-Lab/Amazon-Reviews-2023, preprocess, import, ABSA.

    Same work as the notebook, run on this machine. Then start the website with serve.
    """
    settings = _bootstrap()
    from scripts.build_hf_corpus import build_corpus
    from app.services.analysis import run_analysis
    from app.services.corpus_import import import_corpus_folder

    brands = brand or []

    def progress(done: int, total: int, message: str) -> None:
        console.print(f"({done}/{total}) {message}")

    console.print("[cyan]Phase 1a — download + filter + Step 1 preprocess (exact HF reviews)…[/cyan]")
    manifest = build_corpus(
        max_phones=max_phones,
        max_reviews_per_phone=max_reviews,
        min_rating_count=min_ratings,
        max_review_scan=max_review_scan,
        brand_filter=brands,
        out_dir=Path(out),
    )
    console.print_json(json.dumps(manifest.get("counts", manifest), default=str))

    console.print("[cyan]Phase 1b — import into SQLite (replace existing / demo data)…[/cyan]")
    with session_scope() as db:
        result = import_corpus_folder(
            db,
            Path(out),
            settings=settings,
            replace=True,
            progress=progress,
        )
    console.print_json(json.dumps(result, default=str))

    if not result.get("phones_upserted"):
        console.print("[yellow]No phones imported. Try lowering --min-ratings.[/yellow]")
        raise typer.Exit(1)

    if analyze:
        console.print("[cyan]Phase 1c — methodology Steps 2–6 (segment → ABSA → scores)…[/cyan]")
        analysis = run_analysis(
            phone_ids=result["phone_ids"],
            settings=settings,
            progress=progress,
        )
        console.print_json(json.dumps(analysis, default=str))

    console.print("[green]Phase 1 done.[/green] Phase 2: [cyan]python run.py serve[/cyan] → http://127.0.0.1:8000/ui/")


@app.command("import-corpus")
def cmd_import_corpus(
    folder: Annotated[str, typer.Argument(help="Folder with phones.jsonl + reviews.jsonl")],
    replace: Annotated[bool, typer.Option("--replace/--no-replace", help="Wipe DB first.")] = True,
    analyze: Annotated[bool, typer.Option("--analyze/--no-analyze", help="Run Steps 2-6 after import.")] = True,
) -> None:
    """Import a real HF corpus built in Colab or by scripts/build_hf_corpus.py."""
    settings = _bootstrap()
    from app.services.analysis import run_analysis
    from app.services.corpus_import import import_corpus_folder

    def progress(done: int, total: int, message: str) -> None:
        console.print(f"({done}/{total}) {message}")

    with session_scope() as db:
        result = import_corpus_folder(
            db,
            Path(folder),
            settings=settings,
            replace=replace,
            progress=progress,
        )

    console.print_json(json.dumps(result, default=str))
    if not result["phones_upserted"]:
        console.print("[yellow]No phones imported.[/yellow]")
        raise typer.Exit(1)

    if analyze:
        console.print("[cyan]Running methodology Steps 2–6 (segment → ABSA → aggregate)…[/cyan]")
        analysis = run_analysis(
            phone_ids=result["phone_ids"],
            settings=settings,
            progress=progress,
        )
        console.print_json(json.dumps(analysis, default=str))

    console.print("[green]Done.[/green] Open the UI: [cyan]python run.py serve[/cyan]")


@app.command("ingest-csv")
def cmd_ingest_csv(
    csv_path: Annotated[
        str,
        typer.Option("--csv", help="Path to full_reviews.csv (or compatible scrape)."),
    ] = "data/full_reviews.csv",
    replace: Annotated[bool, typer.Option("--replace/--no-replace", help="Wipe DB first.")] = True,
    max_phones: Annotated[int, typer.Option("--max-phones", help="Phones to keep (0 = all).")] = 0,
    max_reviews: Annotated[int, typer.Option("--max-reviews", help="Reviews per phone (0 = all).")] = 0,
    min_reviews: Annotated[int, typer.Option("--min-reviews", help="Min reviews present in CSV for a phone.")] = 0,
    brand: Annotated[list[str] | None, typer.Option("--brand", "-b", help="Brand filter (repeatable).")] = None,
) -> None:
    """Load local scraped Amazon reviews CSV (``data/full_reviews.csv``).

    Wipes the previous corpus by default, drops user-manual listings, runs
    methodology Step 1 (English-only) on insert.
    """
    settings = _bootstrap()
    from app.services.scraped_csv_ingest import ingest_full_reviews_csv

    def progress(done: int, total: int, message: str) -> None:
        console.print(f"({done}/{total}) {message}")

    with session_scope() as db:
        result = ingest_full_reviews_csv(
            db,
            csv_path=Path(csv_path),
            replace=replace,
            max_phones=max_phones,
            max_reviews_per_phone=max_reviews,
            min_reviews=min_reviews,
            brand_filter=brand,
            settings=settings,
            progress=progress,
        )

    console.print_json(json.dumps(result, default=str))
    if not result["phones_upserted"]:
        console.print("[yellow]No phones imported.[/yellow]")
        raise typer.Exit(1)
    console.print(
        "[green]CSV ingest done.[/green] Next: [cyan]python run.py analyze[/cyan] "
        "or [cyan]python run.py prepare-csv[/cyan]"
    )


@app.command("prepare-csv")
def cmd_prepare_csv(
    csv_path: Annotated[
        str,
        typer.Option("--csv", help="Path to full_reviews.csv."),
    ] = "data/full_reviews.csv",
    max_phones: Annotated[int, typer.Option("--max-phones", help="Phones to keep (0 = all).")] = 0,
    max_reviews: Annotated[int, typer.Option("--max-reviews", help="Reviews per phone (0 = all).")] = 0,
    min_reviews: Annotated[int, typer.Option("--min-reviews", help="Min reviews in CSV for a phone.")] = 0,
    brand: Annotated[list[str] | None, typer.Option("--brand", "-b", help="Brand filter.")] = None,
    analyze: Annotated[bool, typer.Option("--analyze/--no-analyze", help="Run Steps 2-6 after import.")] = True,
) -> None:
    """Recommended path: wipe old corpus, load full_reviews.csv, English preprocess, ABSA."""
    settings = _bootstrap()
    from app.services.analysis import run_analysis
    from app.services.scraped_csv_ingest import ingest_full_reviews_csv

    def progress(done: int, total: int, message: str) -> None:
        console.print(f"({done}/{total}) {message}")

    console.print(
        "[cyan]Phase 1 — scraped full_reviews.csv "
        "(phones + reviews by ASIN, English-only Step 1)…[/cyan]"
    )
    with session_scope() as db:
        result = ingest_full_reviews_csv(
            db,
            csv_path=Path(csv_path),
            replace=True,
            max_phones=max_phones,
            max_reviews_per_phone=max_reviews,
            min_reviews=min_reviews,
            brand_filter=brand,
            settings=settings,
            progress=progress,
        )

    console.print_json(json.dumps(result, default=str))
    if not result["phones_upserted"]:
        console.print("[yellow]No phones imported.[/yellow]")
        raise typer.Exit(1)

    if analyze:
        console.print("[cyan]Phase 2 — methodology Steps 2–6 (segment → ABSA → scores)…[/cyan]")
        analysis = run_analysis(
            phone_ids=result["phone_ids"],
            settings=settings,
            progress=progress,
        )
        console.print_json(json.dumps(analysis, default=str))

    console.print(
        "[green]Done.[/green] Start the app: [cyan]python run.py serve[/cyan] "
        "or [cyan]python run.py website[/cyan]"
    )


@app.command("ingest-hf")
def cmd_ingest_hf(
    max_phones: Annotated[int, typer.Option("--max-phones", help="How many phones to keep.")] = 30,
    max_reviews: Annotated[int, typer.Option("--max-reviews", help="Reviews per phone.")] = 60,
    min_ratings: Annotated[int, typer.Option("--min-ratings", help="Minimum Amazon rating count.")] = 80,
    brand: Annotated[list[str] | None, typer.Option("--brand", "-b", help="Brand filter (repeatable).")] = None,
    max_review_scan: Annotated[int, typer.Option("--max-review-scan", help="Max review rows to scan. 0 = until phones are filled or the JSONL ends.")] = 0,
    analyze: Annotated[bool, typer.Option("--analyze/--no-analyze", help="Run ABSA after ingest.")] = True,
) -> None:
    """Load McAuley-Lab/Amazon-Reviews-2023 (Cell Phones) into the existing DB.

    Streams product metadata + reviews from Hugging Face, keeps real handsets
    (accessories filtered out), runs Step 1 preprocessing on insert, then
    optionally runs the ABSA pipeline so the website can recommend immediately.
    """
    settings = _bootstrap()
    from app.services.analysis import run_analysis
    from app.services.hf_ingest import ingest_amazon_reviews_2023

    def progress(done: int, total: int, message: str) -> None:
        console.print(f"({done}/{total}) {message}")

    with session_scope() as db:
        result = ingest_amazon_reviews_2023(
            db,
            max_phones=max_phones,
            max_reviews_per_phone=max_reviews,
            min_rating_count=min_ratings,
            max_review_scan=max_review_scan,
            brand_filter=brand,
            settings=settings,
            progress=progress,
        )

    console.print_json(json.dumps(result, default=str))
    if not result["phones_upserted"]:
        console.print("[yellow]No phones ingested. Try lowering --min-ratings or removing --brand.[/yellow]")
        raise typer.Exit(1)

    if analyze:
        console.print("[cyan]Running ABSA pipeline…[/cyan]")
        analysis = run_analysis(
            phone_ids=result["phone_ids"],
            settings=settings,
            progress=progress,
        )
        console.print_json(json.dumps(analysis, default=str))

    console.print("[green]Done.[/green] Start the site with: [cyan]python run.py serve[/cyan]")


@app.command("login")
def cmd_login() -> None:
    """Open a browser to sign in once; the session is reused by later scrapes."""
    settings = _bootstrap()
    from app.scrapers.pipeline import save_login_session
    from app.services.jobs import run_coroutine_blocking

    run_coroutine_blocking(save_login_session(settings))


@app.command("scrape")
def cmd_scrape(
    query: Annotated[list[str] | None, typer.Option("--query", "-q", help="Search term (repeatable).")] = None,
    asin: Annotated[list[str] | None, typer.Option("--asin", "-a", help="Explicit product id (repeatable).")] = None,
    max_phones: Annotated[int, typer.Option("--max-phones", help="Phones per query.")] = 5,
    max_reviews: Annotated[int, typer.Option("--max-reviews", help="Reviews per phone.")] = 100,
    max_pages: Annotated[int, typer.Option("--max-pages", help="Review pages per phone.")] = 10,
    sort: Annotated[str, typer.Option(help="Review order: recent or helpful.")] = "recent",
    show_browser: Annotated[bool, typer.Option("--show-browser", help="Run with a visible window.")] = False,
) -> None:
    """Scrape phone details, prices and reviews."""
    settings = _bootstrap()
    if not query and not asin:
        console.print("[red]Provide at least one --query or --asin.[/red]")
        raise typer.Exit(1)

    from app.scrapers.pipeline import scrape_to_db
    from app.services.jobs import run_coroutine_blocking

    def progress(done: int, total: int, message: str) -> None:
        console.print(f"[dim]({done}/{total})[/dim] {message}")

    summary = run_coroutine_blocking(
        scrape_to_db(
            queries=list(query or []),
            product_ids=list(asin or []),
            max_phones_per_query=max_phones,
            max_reviews_per_phone=max_reviews,
            max_review_pages=max_pages,
            review_sort=sort,
            headless=not show_browser,
            settings=settings,
            progress=progress,
        )
    )

    console.print_json(json.dumps(summary, default=str))
    if summary["errors"]:
        console.print(f"[yellow]{len(summary['errors'])} error(s) occurred.[/yellow]")
    if summary["phones_scraped"]:
        console.print("Next: [cyan]python run.py analyze[/cyan]")


@app.command("analyze")
def cmd_analyze(
    phone_id: Annotated[list[int] | None, typer.Option("--phone-id", "-p", help="Limit to phone id (repeatable).")] = None,
    force: Annotated[bool, typer.Option("--force", help="Re-annotate already processed reviews.")] = False,
    max_reviews: Annotated[int | None, typer.Option("--max-reviews", help="Cap reviews per phone.")] = None,
    engine: Annotated[str | None, typer.Option("--engine", help="auto, llm or lexicon.")] = None,
) -> None:
    """Run Steps 1-6: clean, segment, extract aspects, classify sentiment, aggregate."""
    settings = _bootstrap()
    from app.services.analysis import run_analysis

    def progress(done: int, total: int, message: str) -> None:
        console.print(f"[dim]({done}/{total})[/dim] {message}")

    result = run_analysis(
        phone_ids=list(phone_id) if phone_id else None,
        force=force,
        max_reviews_per_phone=max_reviews,
        engine_name=engine,
        settings=settings,
        progress=progress,
    )
    console.print_json(json.dumps(result, default=str))


@app.command("pipeline")
def cmd_pipeline(
    query: Annotated[list[str] | None, typer.Option("--query", "-q")] = None,
    asin: Annotated[list[str] | None, typer.Option("--asin", "-a")] = None,
    max_phones: Annotated[int, typer.Option("--max-phones")] = 5,
    max_reviews: Annotated[int, typer.Option("--max-reviews")] = 100,
    max_pages: Annotated[int, typer.Option("--max-pages")] = 10,
) -> None:
    """Scrape and then analyse, end to end."""
    settings = _bootstrap()
    if not query and not asin:
        console.print("[red]Provide at least one --query or --asin.[/red]")
        raise typer.Exit(1)

    from app.scrapers.pipeline import scrape_to_db
    from app.services.analysis import run_analysis
    from app.services.jobs import run_coroutine_blocking

    def progress(done: int, total: int, message: str) -> None:
        console.print(f"[dim]({done}/{total})[/dim] {message}")

    summary = run_coroutine_blocking(
        scrape_to_db(
            queries=list(query or []),
            product_ids=list(asin or []),
            max_phones_per_query=max_phones,
            max_reviews_per_phone=max_reviews,
            max_review_pages=max_pages,
            settings=settings,
            progress=progress,
        )
    )
    console.print_json(json.dumps(summary, default=str))

    if summary["phone_ids"]:
        result = run_analysis(
            phone_ids=summary["phone_ids"], settings=settings, progress=progress
        )
        console.print_json(json.dumps(result, default=str))


@app.command("stats")
def cmd_stats() -> None:
    """Print corpus statistics and the rating-agreement check."""
    settings = _bootstrap()
    from app.nlp.aggregate import corpus_stats, validation_report

    with session_scope() as db:
        stats = corpus_stats(db, settings)
        validation = validation_report(db, settings)

    table = Table(title="Corpus statistics", show_header=False)
    for key, value in stats.items():
        table.add_row(str(key), json.dumps(value) if isinstance(value, dict) else str(value))
    console.print(table)

    console.print(
        f"\n[bold]Rating agreement:[/bold] {validation['agreement_rate']:.1%} "
        f"over {validation['reviews_compared']} review(s), "
        f"MAE {validation['mean_absolute_error']:.3f}"
    )


@app.command("features")
def cmd_features() -> None:
    """Print the smartphone feature score table (the output layer)."""
    settings = _bootstrap()
    from app.nlp.aspects import aspects_for
    from app.services.recommender import build_feature_vectors

    allowed = aspects_for(settings.aspect_set)

    with session_scope() as db:
        vectors = build_feature_vectors(db, settings=settings)

    if not vectors:
        console.print("[yellow]No phones yet. Run scrape/seed-demo then analyze.[/yellow]")
        return

    table = Table(title="Smartphone feature scores")
    table.add_column("Phone", overflow="fold", max_width=34)
    table.add_column("Price", justify="right")
    for aspect in allowed:
        table.add_column(aspect.title(), justify="right")
    table.add_column("Reviews", justify="right")

    for vector in vectors:
        row = [vector.name, f"{vector.price:.0f}" if vector.price else "-"]
        for aspect in allowed:
            score = vector.scores.get(aspect)
            row.append(f"{score:.2f}" if score is not None else "-")
        row.append(str(vector.review_count))
        table.add_row(*row)

    console.print(table)


@app.command("recommend")
def cmd_recommend(
    weight: Annotated[list[str] | None, typer.Option("--weight", "-w", help="aspect=value (repeatable).")] = None,
    top_k: Annotated[int, typer.Option("--top", help="How many results.")] = 5,
    budget_max: Annotated[float | None, typer.Option("--budget-max")] = None,
    min_reviews: Annotated[int, typer.Option("--min-reviews")] = 0,
) -> None:
    """Rank phones against aspect weights, e.g. -w battery=0.6 -w camera=0.4."""
    settings = _bootstrap()
    from app.models.schemas import RecommendRequest
    from app.services.recommender import recommend

    weights: dict[str, float] = {}
    for item in weight or []:
        if "=" not in item:
            console.print(f"[red]Bad --weight '{item}'. Use aspect=value.[/red]")
            raise typer.Exit(1)
        key, _, value = item.partition("=")
        try:
            weights[key.strip().lower()] = float(value)
        except ValueError:
            console.print(f"[red]'{value}' is not a number.[/red]")
            raise typer.Exit(1) from None

    request = RecommendRequest(
        weights=weights, top_k=top_k, budget_max=budget_max, min_reviews=min_reviews
    )

    with session_scope() as db:
        response = recommend(db, request, settings)

    if not response.results:
        console.print("[yellow]No candidates. Scrape and analyse some phones first.[/yellow]")
        return

    console.print(f"[dim]Weights: {response.weights_used}[/dim]")

    table = Table(title=f"Top {len(response.results)} recommendations")
    table.add_column("#", justify="right")
    table.add_column("Phone", overflow="fold", max_width=32)
    table.add_column("Score", justify="right")
    table.add_column("Coverage", justify="right")
    table.add_column("Price", justify="right")
    table.add_column("Strengths", overflow="fold", max_width=26)
    table.add_column("Weaknesses", overflow="fold", max_width=26)

    for item in response.results:
        table.add_row(
            str(item.rank),
            item.name,
            f"{item.final_score:.3f}",
            f"{item.coverage:.0%}",
            f"{item.price:.0f}" if item.price else "-",
            ", ".join(item.strengths) or "-",
            ", ".join(item.weaknesses) or "-",
        )

    console.print(table)


@app.command("sample-gold")
def cmd_sample_gold(
    n: Annotated[int, typer.Option("--n", help="Number of rows to sample.")] = 200,
) -> None:
    """Write a CSV template for manual ABSA gold labels (proposal Stage 7)."""
    settings = _bootstrap()
    from app.services.absa_eval import sample_gold_template

    with session_scope() as db:
        path = sample_gold_template(db, n=n, settings=settings)
    console.print(
        f"[green]Gold template:[/green] {path}\n"
        "Fill [cyan]gold_sentiment[/cyan] with positive / negative / neutral, then run:\n"
        f"  [cyan]python run.py evaluate-absa --gold \"{path}\"[/cyan]"
    )


@app.command("evaluate-absa")
def cmd_evaluate_absa(
    gold: Annotated[Path, typer.Option("--gold", help="CSV with gold_sentiment filled.")],
) -> None:
    """Score ABSA predictions against a filled gold CSV; save metrics for Evaluation tab."""
    _bootstrap()
    from app.services.absa_eval import evaluate_gold_csv

    report = evaluate_gold_csv(gold)
    console.print_json(json.dumps(report, default=str))
    console.print(f"[green]Saved for Evaluation tab:[/green] {report.get('saved_to')}")


@app.command("export")
def cmd_export() -> None:
    """Export every pipeline stage to CSV."""
    settings = _bootstrap()
    from app.services.exports import export_dataset

    with session_scope() as db:
        files = export_dataset(db, settings=settings)

    for name, path in files.items():
        console.print(f"[green]{name:<18}[/green] {path}")


@app.command("serve")
def cmd_serve(
    host: Annotated[str | None, typer.Option(help="Bind address.")] = None,
    port: Annotated[int | None, typer.Option(help="Port.")] = None,
    reload: Annotated[bool, typer.Option("--reload", help="Auto-reload on code changes.")] = False,
) -> None:
    """Start the FastAPI server."""
    import os

    _bootstrap()
    import uvicorn

    bind_host = host or ("0.0.0.0" if os.environ.get("PORT") else "127.0.0.1")
    bind_port = port or int(os.environ.get("PORT", "8000"))

    console.print(
        f"\n[bold green]Dashboard:[/bold green] http://{bind_host}:{bind_port}/ui/   [dim](open this one)[/dim]"
    )
    console.print(
        f"[green]API docs: [/green] http://{bind_host}:{bind_port}/docs [dim](OpenAPI reference)[/dim]\n"
    )
    uvicorn.run("app.main:app", host=bind_host, port=bind_port, reload=reload)


@app.command("website")
def cmd_website(
    port: Annotated[int, typer.Option(help="Port.")] = 8501,
) -> None:
    """Start the staged research website (one page per methodology stage)."""
    _bootstrap()
    import subprocess
    import sys

    script = Path(__file__).with_name("streamlit_app.py")
    console.print(f"\n[bold green]Website:[/bold green] http://localhost:{port}\n")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(script), "--server.port", str(port)],
        check=False,
    )


if __name__ == "__main__":
    app()
