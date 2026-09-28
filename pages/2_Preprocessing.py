"""Stage 2: inspect Step 1 cleaning and Step 2 sentence segmentation."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from sqlalchemy import func, select

from app.core.database import session_scope
from app.models.entities import Review, Sentence, Smartphone
from app.nlp.segment import segment_sentences
from app.services.workflow import pipeline_status
from app.ui_support import bootstrap, sidebar_status

st.set_page_config(page_title="Preprocessing", page_icon="ðŸ§¹", layout="wide")
bootstrap()

st.title("2. Preprocessing")
st.caption("Step 1 cleaning Â· Step 2 sentence segmentation")

status = pipeline_status()
sidebar_status(st, status)

if not status["db_reviews"]:
    st.info("No reviews in the database yet. Import a corpus on **1 Dataset** first.")
    st.stop()

st.markdown(
    """
Step 1 runs when reviews are inserted: HTML stripping, whitespace and unicode
normalisation, spam and length checks, language filtering, and content-hash
de-duplication. Rejected reviews stay in the database with a reason so the
exclusions are auditable.
"""
)

metrics = st.columns(4)
metrics[0].metric("Reviews stored", status["db_reviews"])
metrics[1].metric("Usable after Step 1", status["db_reviews_usable"])
metrics[2].metric("Analysed", status["db_reviews_analysed"])
metrics[3].metric("Sentences", status["db_sentences"])

with session_scope() as db:
    exclusion_rows = db.execute(
        select(Review.excluded_reason, func.count(Review.id))
        .where(Review.excluded_reason.is_not(None))
        .group_by(Review.excluded_reason)
        .order_by(func.count(Review.id).desc())
    ).all()
    spam_count = db.scalar(select(func.count(Review.id)).where(Review.is_spam.is_(True))) or 0
    per_phone = db.execute(
        select(
            Smartphone.id,
            Smartphone.canonical_name,
            Smartphone.raw_title,
            func.count(Review.id),
        )
        .join(Review, Review.smartphone_id == Smartphone.id)
        .group_by(Smartphone.id)
        .order_by(func.count(Review.id).desc())
    ).all()

st.subheader("Why reviews were excluded")
if exclusion_rows:
    st.dataframe(
        pd.DataFrame(
            [{"reason": reason, "reviews": count} for reason, count in exclusion_rows]
        ),
        width="stretch",
        hide_index=True,
    )
else:
    st.write("No reviews were excluded by Step 1.")
st.caption(f"Reviews flagged as spam: {spam_count}")

st.subheader("Reviews per phone")
st.dataframe(
    pd.DataFrame(
        [
            {"phone_id": pid, "phone": name or title, "reviews": count}
            for pid, name, title, count in per_phone
        ]
    ),
    width="stretch",
    hide_index=True,
)

# --------------------------------------------------------------------------- #
st.header("Before and after cleaning")

phone_choices = {f"{name or title} (id {pid})": pid for pid, name, title, _c in per_phone}
if phone_choices:
    chosen_label = st.selectbox("Phone", list(phone_choices))
    chosen_id = phone_choices[chosen_label]

    with session_scope() as db:
        reviews = list(
            db.scalars(
                select(Review).where(Review.smartphone_id == chosen_id).limit(25)
            ).all()
        )
        sample = [
            {
                "review_id": review.id,
                "rating": review.rating,
                "raw": (review.body or "")[:300],
                "cleaned": (review.cleaned_body or "")[:300],
                "language": review.language,
                "excluded_reason": review.excluded_reason,
            }
            for review in reviews
        ]
        review_texts = {
            review.id: review.cleaned_body or review.body or "" for review in reviews
        }

    st.dataframe(pd.DataFrame(sample), width="stretch", hide_index=True)

    st.header("Sentence segmentation (Step 2)")
    st.caption("Segmentation is applied to the cleaned text before aspect extraction.")

    if review_texts:
        review_id = st.selectbox("Review", list(review_texts))
        text = review_texts[review_id]
        sentences = segment_sentences(text)
        st.write(f"**{len(sentences)}** sentence(s) from review {review_id}:")
        for index, sentence in enumerate(sentences, start=1):
            st.write(f"{index}. {sentence}")
        if not sentences:
            st.warning("This review produced no sentences, so it cannot feed ABSA.")

with session_scope() as db:
    stored_sentences = db.scalar(select(func.count(Sentence.id))) or 0

if stored_sentences:
    st.success(
        f"{stored_sentences} sentence(s) are already stored from a previous ABSA run."
    )
else:
    st.info("Sentences are written to the database when you run **3 ABSA Analysis**.")
