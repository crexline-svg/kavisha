"""Home page of the staged research website.

Run with::

    streamlit run streamlit_app.py

Each page in ``pages/`` is one methodology stage, so the pipeline can be
explained and inspected step by step instead of as a single command.
"""

from __future__ import annotations

import streamlit as st

from app import __version__
from app.core.config import get_settings
from app.nlp.aspects import ASPECT_LABELS, aspects_for
from app.services.workflow import pipeline_status
from app.ui_support import bootstrap, sidebar_status

st.set_page_config(page_title="Smartphone Recommendation System", page_icon="ðŸ“±", layout="wide")

bootstrap()

st.title("Smartphone Recommendation System")
st.caption("Aspect-Based Sentiment Analysis of Amazon smartphone reviews")

status = pipeline_status()
sidebar_status(st, status)

st.markdown(
    """
Work through the pages in order. Nothing downloads or overwrites data on its own â€”
every stage waits for a button click.

| Page | Methodology stage |
| --- | --- |
| **1 Dataset** | Scan Amazon metadata, pick real smartphones, collect their reviews (Step 1 preprocessing) |
| **2 Preprocessing** | Inspect cleaning results and sentence segmentation (Steps 1-2) |
| **3 ABSA Analysis** | Aspect extraction and sentiment classification (Steps 3-4) |
| **4 Aspect Scores** | Aggregate aspect-sentiment records into feature scores (Steps 5-6) |
| **5 Recommendations** | Weighted ranking from your requirements |
"""
)

st.subheader("Where the pipeline stands")

first = st.columns(4)
first[0].metric("Candidate phones", status["candidates"])
first[1].metric("Selected phones", status["selected"])
first[2].metric("Corpus phones", status["corpus_phones"])
first[3].metric("Corpus reviews", status["corpus_reviews"])

second = st.columns(4)
second[0].metric("Phones in database", status["db_phones"])
second[1].metric("Reviews analysed", status["db_reviews_analysed"])
second[2].metric("Sentences", status["db_sentences"])
second[3].metric("Phones scored", status["db_scored_phones"])

if not status["candidates"]:
    st.info("Start on **1 Dataset**: scan a small slice of product metadata.")
elif not status["selected"]:
    st.info("Candidates are ready. On **1 Dataset**, tick the phones you want to study.")
elif not status["corpus_phones"]:
    st.info("Selection saved. On **1 Dataset**, collect reviews for the selected phones.")
elif not status["db_phones"]:
    st.info("Corpus files are built. On **1 Dataset**, import them into the database.")
elif not status["db_reviews_analysed"]:
    st.info("Reviews are imported. Run **3 ABSA Analysis** next.")
elif not status["db_scored_phones"]:
    st.info("ABSA is done. Build the score table on **4 Aspect Scores**.")
else:
    st.success("The pipeline is complete. Rank phones on **5 Recommendations**.")

with st.expander("Configured aspects and data source"):
    aspects = aspects_for(get_settings().aspect_set)
    st.write(
        {
            "app_version": __version__,
            "dataset": "McAuley-Lab/Amazon-Reviews-2023 (Cell Phones and Accessories)",
            "absa_engine": status["engine"],
            "aspects": [ASPECT_LABELS.get(a, a) for a in aspects],
        }
    )
    st.caption(
        "The LLM (when configured) is used only for aspect extraction and sentiment. "
        "Ranking is deterministic Python arithmetic."
    )
