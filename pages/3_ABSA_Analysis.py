"""Stage 3: aspect extraction and sentiment classification (Steps 3-4)."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from sqlalchemy import func, select

from app.core.config import get_settings
from app.core.database import session_scope
from app.models.entities import AspectSentiment, Sentence, Smartphone
from app.services.analysis import run_analysis
from app.services.workflow import pipeline_status
from app.ui_support import bootstrap, sidebar_status, stage_progress

st.set_page_config(page_title="ABSA Analysis", page_icon="ðŸ§ ", layout="wide")
bootstrap()

st.title("3. ABSA Analysis")
st.caption("Step 2 segmentation Â· Step 3 aspect extraction Â· Step 4 sentiment classification")

settings = get_settings()
status = pipeline_status()
sidebar_status(st, status)

if not status["db_reviews"]:
    st.info("Import a corpus on **1 Dataset** before running ABSA.")
    st.stop()

engine_name = settings.resolved_absa_engine()
if engine_name == "lexicon":
    st.warning(
        "Running the **lexicon baseline** because no `LLM_API_KEY` is set. "
        "Add one to `.env` to run the LLM annotator from the methodology. "
        "Keeping both lets you report the LLM against a baseline."
    )
else:
    st.success(f"Using the **LLM** engine (`{settings.llm_model}`).")

st.markdown(
    """
Sentences for one phone are annotated in a single call, so the LLM engine can batch
them. Aspect labels are restricted to the closed taxonomy â€” the model cannot invent
new aspect names.
"""
)

with session_scope() as db:
    phones = db.execute(
        select(Smartphone.id, Smartphone.canonical_name, Smartphone.raw_title).order_by(
            Smartphone.id
        )
    ).all()

labels = {f"{name or title} (id {pid})": pid for pid, name, title in phones}

scope = st.radio(
    "What to analyse",
    ("All phones in the database", "Selected phones only"),
    index=0,
)
selected_ids: list[int] | None = None
if scope.startswith("Selected"):
    picked = st.multiselect("Phones", list(labels))
    selected_ids = [labels[label] for label in picked]

controls = st.columns(3)
with controls[0]:
    max_reviews = st.number_input(
        "Max reviews per phone (0 = all pending)", min_value=0, max_value=5000, value=0, step=10
    )
with controls[1]:
    force = st.checkbox(
        "Re-analyse everything",
        value=False,
        help="Deletes existing sentences and re-runs Step 1 with current settings.",
    )
with controls[2]:
    engine_override = st.selectbox("Engine", ("auto (configured)", "lexicon", "llm"), index=0)

if st.button("Run ABSA", type="primary"):
    bar = st.progress(0.0)
    line = st.empty()
    try:
        with st.spinner("Segmenting sentences and extracting aspect sentimentâ€¦"):
            result = run_analysis(
                phone_ids=selected_ids or None,
                force=force,
                max_reviews_per_phone=int(max_reviews) or None,
                engine_name=None if engine_override.startswith("auto") else engine_override,
                settings=settings,
                progress=stage_progress(line, bar),
            )
        st.success(
            f"Analysed {result['phones_analyzed']} phone(s): "
            f"{result['sentences_created']} sentence(s), "
            f"{result['aspects_extracted']} aspect mention(s)."
        )
        st.json(result)
        if result.get("errors"):
            st.error("Some phones failed. See the errors above.")
        else:
            st.info("Next: build the score table on **4 Aspect Scores**.")
    except Exception as exc:  # noqa: BLE001
        st.error(f"ABSA run failed: {exc}")

# --------------------------------------------------------------------------- #
st.header("Structured aspect-sentiment records (Step 5)")

with session_scope() as db:
    total_rows = db.scalar(select(func.count(AspectSentiment.id))) or 0
    by_aspect = db.execute(
        select(AspectSentiment.aspect, AspectSentiment.sentiment, func.count(AspectSentiment.id))
        .group_by(AspectSentiment.aspect, AspectSentiment.sentiment)
    ).all()
    sample = db.execute(
        select(
            Smartphone.canonical_name,
            AspectSentiment.aspect,
            AspectSentiment.sentiment,
            AspectSentiment.opinion_term,
            Sentence.text,
        )
        .join(Sentence, Sentence.id == AspectSentiment.sentence_id)
        .join(Smartphone, Smartphone.id == AspectSentiment.smartphone_id)
        .limit(50)
    ).all()

if not total_rows:
    st.info("No aspect-sentiment records yet.")
else:
    st.metric("Aspect mentions stored", total_rows)
    pivot = (
        pd.DataFrame(
            [
                {"aspect": aspect, "sentiment": sentiment, "count": count}
                for aspect, sentiment, count in by_aspect
            ]
        )
        .pivot(index="aspect", columns="sentiment", values="count")
        .fillna(0)
        .astype(int)
    )
    st.dataframe(pivot, width="stretch")

    st.subheader("Sample records")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "phone": phone,
                    "aspect": aspect,
                    "sentiment": sentiment,
                    "opinion_term": term,
                    "sentence": text,
                }
                for phone, aspect, sentiment, term, text in sample
            ]
        ),
        width="stretch",
        hide_index=True,
    )
