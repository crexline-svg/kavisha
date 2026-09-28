"""Stage 5: weighted recommendation from user requirements."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.core.config import get_settings
from app.core.database import session_scope
from app.models.schemas import RecommendRequest
from app.nlp.aspects import ASPECT_LABELS, aspects_for
from app.services.recommender import AFFORDABILITY_KEY, recommend
from app.services.workflow import pipeline_status
from app.ui_support import bootstrap, format_price, render_phone_tile, sidebar_status

st.set_page_config(page_title="Recommendations", page_icon="â­", layout="wide")
bootstrap()

st.title("5. Recommendations")
st.caption("Your requirements, ranked against the aspect score database")

settings = get_settings()
status = pipeline_status()
sidebar_status(st, status)

if not status["db_scored_phones"]:
    st.info("No aspect scores yet. Build them on **4 Aspect Scores** first.")
    st.stop()

aspects = aspects_for(settings.aspect_set)

PRESETS: dict[str, dict[str, int]] = {
    "Balanced": {},
    "Photography": {"camera": 5, "display": 4, "performance": 3, "battery": 3},
    "Battery life": {"battery": 5, "performance": 3, "price": 3},
    "Gaming": {"performance": 5, "display": 4, "battery": 4},
    "Best value": {AFFORDABILITY_KEY: 5, "price": 4, "battery": 3, "performance": 3},
}

left, right = st.columns([1, 2])

with left:
    st.subheader("Your requirements")
    preset_name = st.selectbox("Preset", list(PRESETS))
    preset = PRESETS[preset_name]

    weights: dict[str, float] = {}
    for aspect in aspects:
        weights[aspect] = st.slider(
            ASPECT_LABELS.get(aspect, aspect),
            min_value=0,
            max_value=5,
            value=int(preset.get(aspect, 3)),
            help="0 means this aspect is ignored.",
        )
    weights[AFFORDABILITY_KEY] = st.slider(
        "Affordability (numeric price)",
        min_value=0,
        max_value=5,
        value=int(preset.get(AFFORDABILITY_KEY, 2)),
        help="Cheapest phone in the candidate set scores 1.0. Separate from review talk about price.",
    )

    st.divider()
    budget_cols = st.columns(2)
    with budget_cols[0]:
        budget_min = st.number_input("Min price", min_value=0.0, value=0.0, step=25.0)
    with budget_cols[1]:
        budget_max = st.number_input("Max price (0 = any)", min_value=0.0, value=0.0, step=25.0)

    min_reviews = st.number_input(
        "Minimum analysed reviews",
        min_value=0,
        value=0,
        step=1,
        help="Keep at 0 while the corpus is small.",
    )
    top_k = st.number_input("How many phones to show", min_value=1, max_value=50, value=10)
    shrinkage = st.checkbox("Apply shrinkage to sparse scores", value=True)
    go = st.button("Rank phones", type="primary")

with right:
    if not go:
        st.info("Set your priorities on the left, then rank phones.")
        st.stop()

    request = RecommendRequest(
        weights={key: float(value) for key, value in weights.items() if value > 0},
        budget_min=budget_min or None,
        budget_max=budget_max or None,
        min_reviews=int(min_reviews),
        top_k=int(top_k),
        apply_shrinkage=shrinkage,
    )

    with session_scope() as db:
        response = recommend(db, request, settings)

    if not response.results:
        st.warning(
            f"No phone matched. {response.candidates_considered} candidate(s) passed the "
            "filters. Try lowering the minimum reviews or widening the price range."
        )
        st.stop()

    st.caption(
        f"Weights used (normalised): "
        + ", ".join(f"{ASPECT_LABELS.get(k, k)} {v:.2f}" for k, v in response.weights_used.items())
    )

    summary = pd.DataFrame(
        [
            {
                "rank": item.rank,
                "phone": item.name,
                "brand": item.brand or "Unknown",
                "price": item.price,
                "score": round(item.final_score, 3),
                "evidence coverage": round(item.coverage, 2),
                "reviews": item.review_count,
            }
            for item in response.results
        ]
    )
    st.subheader("Top matches")

    cols_per_row = 3
    for row_start in range(0, len(response.results), cols_per_row):
        row_items = response.results[row_start : row_start + cols_per_row]
        cols = st.columns(len(row_items))
        for col, item in zip(cols, row_items):
            with col:
                render_phone_tile(
                    st,
                    rank=item.rank,
                    name=item.name,
                    brand=item.brand,
                    price=item.price,
                    currency=item.currency,
                    image_url=item.image_url,
                    site_rating=item.site_rating,
                    site_rating_count=item.site_rating_count,
                    score=item.final_score,
                    review_count=item.review_count,
                )

    with st.expander("Summary table"):
        st.dataframe(summary, width="stretch", hide_index=True)

    for item in response.results:
        with st.expander(f"#{item.rank} - {item.name} - score {item.final_score:.3f}"):
            detail_cols = st.columns([1, 2])
            with detail_cols[0]:
                if item.image_url:
                    st.image(item.image_url, use_container_width=True)
            with detail_cols[1]:
                info = st.columns(3)
                info[0].metric("Price", format_price(item.price, item.currency))
                info[1].metric("Analysed reviews", item.review_count)
                info[2].metric("Evidence coverage", f"{item.coverage:.0%}")

            if item.strengths:
                st.write("**Strengths:** " + ", ".join(item.strengths))
            if item.weaknesses:
                st.write("**Weaknesses:** " + ", ".join(item.weaknesses))

            st.write("How the score was built:")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "aspect": ASPECT_LABELS.get(part.aspect, part.aspect),
                            "score": part.score,
                            "weight": part.weight,
                            "contribution": part.contribution,
                            "mentions": part.mention_count,
                            "imputed": part.imputed,
                        }
                        for part in item.breakdown
                    ]
                ),
                width="stretch",
                hide_index=True,
            )
            if any(part.imputed for part in item.breakdown):
                st.caption(
                    "Imputed aspects were never mentioned for this phone and were filled "
                    "with the candidate-set mean, which lowers evidence coverage."
                )
