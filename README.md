# Smartphone Recommendation via Aspect-Based Sentiment Analysis

Research prototype (v1.2.0) for **explainable, feature-based smartphone recommendations** from Amazon review text.

The system:

1. Loads smartphone reviews (scraped CSV, optional live scrape, or Hugging Face corpus)
2. Runs a full **ABSA pipeline** (preprocess → segment → aspect + sentiment → scores)
3. Lets users set **feature priorities** and ranks phones with plain-language explanations
4. Compares the proposed ranking against an **Amazon star-rating baseline**
5. Collects user ratings and reports **NDCG@3**, **Spearman**, and review–star agreement

**Student:** IM/2021/049  
**UI:** http://127.0.0.1:8000/ui/  
**API docs:** http://127.0.0.1:8000/docs

---

## Table of contents

1. [Research aim](#1-research-aim)
2. [Technologies used](#2-technologies-used)
3. [Architecture](#3-architecture)
4. [Methodology steps](#4-methodology-steps)
5. [Features (aspects)](#5-features-aspects)
6. [Quick start](#6-quick-start)
7. [Full workflow (recommended)](#7-full-workflow-recommended)
8. [Other data paths](#8-other-data-paths)
9. [Web UI](#9-web-ui)
10. [Evaluation](#10-evaluation)
11. [Scoring formulas](#11-scoring-formulas)
12. [CLI reference](#12-cli-reference)
13. [HTTP API](#13-http-api)
14. [Configuration](#14-configuration)
15. [Project structure](#15-project-structure)
16. [Thesis notes (as-built)](#16-thesis-notes-as-built)

---

## 1. Research aim

> Develop an explainable feature-based smartphone recommendation system using Aspect-Based Sentiment Analysis (ABSA) of user reviews.

**Objectives covered by this prototype**

| # | Objective | Implementation |
|---|-----------|----------------|
| 1 | Collect & prepare smartphone review data | CSV / scrape / HF → SQLite |
| 2 | Identify features and classify sentiment (+ / − / neutral) | Lexicon ABSA (optional LLM) |
| 3 | Compute per-phone feature performance scores | Aggregation + shrinkage |
| 4 | Rank phones by user-defined feature weights | Recommend UI + API |
| 5 | Produce plain-language explanations | “Why recommended” text per result |

**Out of scope (by design):** collaborative filtering, live scrape at recommend time, gold-label Macro-F1 (optional / skipped).

---

## 2. Technologies used

### Languages & runtime

| Technology | Role |
|---|---|
| **Python 3.12+** | Backend, NLP pipeline, CLI |
| **HTML / CSS / JavaScript** | Research web UI (`app/static/`) |
| **SQL** (via SQLAlchemy) | Persistence |

### Backend & API

| Technology | Role |
|---|---|
| **FastAPI** | REST API and static UI hosting |
| **Uvicorn** | ASGI server |
| **Pydantic / pydantic-settings** | Request schemas and `.env` config |
| **Typer + Rich** | CLI (`run.py`) |
| **python-dotenv** | Environment loading |

### Database & data

| Technology | Role |
|---|---|
| **SQLite** | Default research DB (`data/smartphones.db`) |
| **SQLAlchemy 2.0** | ORM / sessions |
| **Pandas** | CSV ingest, exports, corpus helpers |
| **Hugging Face `datasets`** | Optional Amazon Reviews 2023 path |

### Scraping (optional / legacy live path)

| Technology | Role |
|---|---|
| **Playwright** | Browser automation for Amazon |
| **BeautifulSoup4 + lxml** | HTML parsing |
| **Tenacity** | Retries / backoff |

### NLP / ABSA

| Technology | Role |
|---|---|
| **pysbd** | Sentence segmentation |
| **langdetect** | English-only filter |
| **Domain lexicon engine** | Offline aspect + sentiment (default practical engine) |
| **httpx** | OpenAI-compatible LLM client (optional) |
| **OpenAI / OpenRouter / Groq / Ollama** | Optional LLM ABSA via chat API |

### Frontends

| Technology | Role |
|---|---|
| **Custom static UI** | Main research site: Recommend / Phones / Evaluation |
| **Outfit + IBM Plex Mono** | UI typography (Google Fonts) |
| **Streamlit** | Optional staged lab UI (`python run.py website`) |

### Evaluation metrics (implemented)

| Metric | Purpose |
|---|---|
| **NDCG@3** | Ranking quality from user 1–5 ratings |
| **Spearman ρ** | Rank correlation vs user ratings |
| **Star–text agreement** | ABSA polarity vs Amazon stars (sanity check) |
| **Proposed vs star baseline** | Method comparison in Evaluation tab |

---

## 3. Architecture

```
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────┐
│ Reviews corpus  │────▶│ ABSA pipeline        │────▶│ SQLite          │
│ CSV / scrape/HF │     │ Steps 1–6            │     │ scores + text   │
└─────────────────┘     └──────────────────────┘     └────────┬────────┘
                                                              │
                    ┌─────────────────────────────────────────┘
                    ▼
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────┐
│ FastAPI         │◀───▶│ Recommendation       │────▶│ Static UI       │
│ /recommend etc. │     │ weights + baseline   │     │ /ui/            │
└─────────────────┘     └──────────────────────┘     └─────────────────┘
                    │
                    ▼
            Feedback + Evaluation
            (NDCG, Spearman, compare methods)
```

**Primary research path:** offline `data/full_reviews.csv` → `prepare-csv` / `analyze` → `serve` → `/ui/`.

---

## 4. Methodology steps

| Step | Description | Module |
|---|---|---|
| Input | Phones, prices, reviews | `scraped_csv_ingest` / scrapers / HF ingest |
| **1** | Clean, normalise, spam filter, de-dupe, English filter | `app/nlp/preprocess.py` |
| **2** | Sentence segmentation | `app/nlp/segment.py` |
| **3–4** | Aspect extraction + sentiment classification | `app/nlp/absa.py` |
| **5** | Structured aspect–sentiment records | table `aspect_sentiments` |
| **6** | Aspect score aggregation | `app/nlp/aggregate.py` |
| Output | Feature score database | `GET /features` |
| Rec. | Weighted ranking + explanations | `app/services/recommender.py` |
| Eval. | Feedback, NDCG, baseline compare | `app/services/feedback.py` |

### ABSA engines

| Engine | Needs | Use |
|---|---|---|
| `lexicon` | Nothing | Offline domain lexicon (recommended for reproducible runs) |
| `llm` | `LLM_API_KEY` | Optional OpenAI-compatible annotator |
| `auto` | — | LLM if key present, else lexicon |

Force lexicon:

```powershell
python run.py analyze --force --engine lexicon
```

---

## 5. Features (aspects)

**Core set (`ASPECT_SET=core`) — six features matching the study:**

| Aspect | Meaning |
|---|---|
| `battery` | Battery life, charging, endurance |
| `camera` | Photo / video quality |
| `display` | Screen quality, brightness, refresh |
| `performance` | Speed, gaming, heating, multitasking |
| `design` | Build, look, materials, ergonomics |
| `price` | Value for money **from review text** |

**Extended set** adds: `software`, `connectivity`, `audio`, `durability`.

**UI note:** There is **one** Price priority slider (review-based). List price is filtered with Min/Max USD, not a second “affordability” slider.

---

## 6. Quick start

```powershell
cd "d:\4TH YEAR\RESEARCH SETUP\smartphone recommendation"

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Optional: copy and edit secrets
copy .env.example .env

# Recommended corpus path (stop serve first if it is running)
python run.py prepare-csv
python run.py serve
```

Open **http://127.0.0.1:8000/ui/**

If Design was added after an older analysis, re-run:

```powershell
# Stop serve (Ctrl+C) first — SQLite needs a free writer
python run.py analyze --force --engine lexicon
python run.py serve
```

Hard-refresh the browser: `Ctrl+F5`.

---

## 7. Full workflow (recommended)

### Step A — Environment

1. Create venv and install `requirements.txt`
2. Copy `.env.example` → `.env`
3. Leave `ABSA_ENGINE=auto` or set `lexicon` for fully offline ABSA
4. Optional: set `LLM_API_KEY` only if you want LLM annotations

### Step B — Load reviews

Place scraped file at `data/full_reviews.csv` (or pass `--csv PATH`).

```powershell
python run.py prepare-csv
```

This:

1. Drops non-phone listings
2. Replaces previous corpus in SQLite
3. Runs Step 1 (English-only cleaning)
4. Runs Steps 2–6 (segment → ABSA → scores) unless `--no-analyze`

Useful flags:

| Flag | Default | Meaning |
|---|---|---|
| `--csv PATH` | `data/full_reviews.csv` | Input file |
| `--max-phones` | `0` | Cap phones (`0` = all) |
| `--max-reviews` | `0` | Cap reviews per phone |
| `--brand Apple` | — | Optional brand filter (repeatable) |
| `--no-analyze` | off | Import + Step 1 only |

### Step C — Serve the research UI

```powershell
python run.py serve
```

| URL | Purpose |
|---|---|
| http://127.0.0.1:8000/ui/ | Main app |
| http://127.0.0.1:8000/docs | OpenAPI |

### Step D — Use the recommender

1. **Recommend** → set priorities (1–10) for the six aspects  
2. Optional budget min/max  
3. Choose ranking method: **Your priorities** or **Amazon stars (baseline)**  
4. **Rank phones** → read Top-10 + “Why recommended”  
5. Rate each phone **1–5**  
6. Open **Evaluation** for cumulative metrics  

Repeat with the other ranking method so Evaluation can compare them.

### Step E — Export (optional)

```powershell
python run.py export
python run.py stats
python run.py features
```

---

## 8. Other data paths

### Load CSV without ABSA (analyze later)

```powershell
python run.py ingest-csv --csv data/full_reviews.csv
python run.py analyze --force --engine lexicon
```

### Live Amazon scrape (optional)

```powershell
python run.py login          # once — saves cookies
python run.py scrape -q "samsung galaxy s24"
python run.py analyze
```

Requires Playwright browsers:

```powershell
playwright install chromium
```

### Hugging Face Amazon Reviews 2023 (legacy / large)

```powershell
python run.py prepare-corpus --max-phones 0 --max-reviews 60
# or Colab zip → import-corpus
python run.py import-corpus .\hf_corpus --replace --analyze
```

### Staged Streamlit website

```powershell
python run.py website
# -> http://localhost:8501
```

Pages: Dataset → Preprocessing → ABSA → Aspect Scores → Recommendations.

### Demo / reset

```powershell
python run.py seed-demo      # synthetic only — do not report in thesis
python run.py purge-demo -y
python run.py reset-corpus -y
```

---

## 9. Web UI

Three main views:

| View | Purpose |
|---|---|
| **Recommend** | Priorities, rank (weighted or star baseline), rate phones, explanations |
| **Phones** | Browse phones and per-aspect score bars (incl. Design) |
| **Evaluation** | Satisfaction, NDCG@3, Spearman, method comparison, star–text agreement |

Theme: light / dark (top bar toggle). Brand palette: teal / slate. Fonts: Outfit + IBM Plex Mono.

---

## 10. Evaluation

What the Evaluation tab uses **without** gold labels:

| Signal | How it is collected |
|---|---|
| User satisfaction / relevance | Same 1–5 ratings after ranking |
| Ranking quality | Mean **NDCG@3** and **Spearman** across sessions |
| Method comparison | Sessions tagged `weighted` vs `star_rating` |
| ABSA sanity check | Agreement between review polarity and Amazon stars |

Optional gold path (not required):

```powershell
python run.py sample-gold
# fill gold_sentiment in the CSV
python run.py evaluate-absa --gold data/artifacts/gold_absa_template.csv
```

---

## 11. Scoring formulas

### Aspect score (Step 6)

```
raw_score = (positive + 0.5 × neutral) / mentions     ∈ [0, 1]
```

Stored `score` applies Bayesian shrinkage toward the corpus mean:

```
score = (mentions × raw_score + k × corpus_mean) / (mentions + k)
```

`k = SHRINKAGE_STRENGTH` (default `5`).

### Recommendation (proposed)

```
final = Σ(wₐ × scoreₐ) / Σ(wₐ)
```

- Missing aspects → imputed with candidate-set mean (flagged `imputed`, not zero)
- `coverage` = share of requested weight backed by real mentions
- Baseline method sorts by Amazon `site_rating` instead of weighted aspect scores

---

## 12. CLI reference

```text
python run.py init-db
python run.py prepare-csv [--csv PATH] [--max-phones N] [--no-analyze]
python run.py ingest-csv
python run.py analyze [--force] [--engine lexicon|llm|auto] [--phone-id ID]
python run.py serve
python run.py website
python run.py stats
python run.py features
python run.py recommend -w battery=8 -w camera=5 -w design=6
python run.py export
python run.py sample-gold
python run.py evaluate-absa --gold PATH
python run.py login
python run.py scrape -q "query" [--asin ASIN]
python run.py pipeline -q "query"
python run.py prepare-corpus | import-corpus | ingest-hf | reset-corpus
python run.py seed-demo | purge-demo
```

---

## 13. HTTP API

Interactive docs: `/docs`. Long jobs return a job id; poll `GET /jobs/{job_id}`.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Version, engine, aspect set |
| `GET` | `/stats` | Corpus counts |
| `GET` | `/aspects` | Active aspect taxonomy |
| `GET` | `/phones` | Phone list |
| `GET` | `/phones/{id}` | Phone detail + aspect scores |
| `GET` | `/features` | Feature score vectors |
| `POST` | `/recommend` | Weighted or star-baseline ranking |
| `POST` | `/feedback` | Save ratings / session feedback |
| `GET` | `/feedback/evaluation` | Aggregate evaluation report |
| `POST` | `/scrape` | Background scrape job |
| `POST` | `/analyze` | Background analysis job |
| `GET` | `/jobs/{id}` | Job status |
| `POST` | `/export` | Export pipeline CSVs |

Static UI is served at `/ui/`.

---

## 14. Configuration

Copy `.env.example` to `.env`. Important keys:

| Variable | Default | Meaning |
|---|---|---|
| `ABSA_ENGINE` | `auto` | `auto` / `llm` / `lexicon` |
| `ASPECT_SET` | `core` | `core` (6 features) or `extended` |
| `LLM_BASE_URL` | OpenAI | Any OpenAI-compatible base URL |
| `LLM_API_KEY` | empty | Required only for LLM engine |
| `LLM_MODEL` | `gpt-4o-mini` | Chat model id |
| `LANGUAGE_FILTER` | `en` | Keep English reviews |
| `NEUTRAL_WEIGHT` | `0.5` | Neutral contribution in score |
| `MIN_MENTIONS_FOR_SCORE` | `3` | Low-confidence threshold |
| `SHRINKAGE_STRENGTH` | `5.0` | Shrinkage `k` |
| `MARKETPLACE` | amazon.com | Live scrape locale |
| `HEADLESS` | `true` | Playwright headless mode |

Database default: `sqlite:///data/smartphones.db`.

---

## 15. Project structure

```text
smartphone recommendation/
├── app/
│   ├── api/routes/          # FastAPI routers
│   ├── core/                # config, database, logging
│   ├── models/              # SQLAlchemy entities + Pydantic schemas
│   ├── nlp/                 # preprocess, segment, ABSA, aggregate, aspects
│   ├── scrapers/            # Playwright Amazon scrapers
│   ├── services/            # analysis, recommend, feedback, ingest, metrics
│   ├── static/              # UI: index.html, app.css, app.js
│   └── main.py              # FastAPI app
├── pages/                   # Streamlit staged website
├── data/
│   ├── full_reviews.csv     # primary scraped corpus
│   ├── smartphones.db       # SQLite research DB
│   ├── artifacts/           # gold templates, scrape debug files
│   └── exports/             # CSV exports
├── notebooks/               # optional Colab HF corpus notebook
├── scripts/                 # helpers
├── tests/
├── run.py                   # CLI entry point
├── streamlit_app.py
├── requirements.txt
├── .env.example
└── README.md
```

---

## 16. Thesis notes (as-built)

Document the system **as implemented**, not only as originally proposed:

| Proposal wording | As-built |
|---|---|
| Transformer / BERT training | **Lexicon ABSA** (+ optional LLM API); no local BERT fine-tune |
| Six features incl. design | **Yes** — core aspects include design |
| Manual gold Macro-F1 | **Optional / skipped**; use star agreement + user ratings |
| Rating baseline | **Yes** — Amazon star ranking + Evaluation compare |
| Explanations | **Yes** — plain-language feature-score explanations |

**Typical demo script for viva**

1. Show corpus + Phones tab (six aspect bars)  
2. Set priorities → Rank (proposed) → explain Top-1  
3. Switch to Amazon stars → Rank again → rate both lists  
4. Open Evaluation → NDCG / method comparison  

---

## License / ethics

Amazon reviews are third-party content. Respect site terms, institutional ethics rules, and privacy (do not publish reviewer names). Prefer the offline CSV / public dataset path for reproducibility.
