"""Stage 4: aggregate aspect-sentiment records into the feature score table."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.core.config import get_settings
from app.core.database import session_scope
from app.nlp.aggregate import recompute_aspect_scores
from app.nlp.aspects import ASPECT_LABELS, aspects_for
from app.services.recommender import build_feature_vectors
from app.services.workflow import pipeline_status
from app.ui_support import bootstrap, sidebar_status

st.set_page_config(page_title="Aspect Scores", page_icon="ðŸ“Š", layout="wide")
bootstrap()

st.title("4. Aspect Scores")
st.caption("Step 6 aggregation Â· the smartphone feature score database")

settings = get_settings()
status = pipeline_status()
sidebar_status(st, status)

if not status["db_aspect_rows"]:
    st.info("No aspect-sentiment records yet. Run **3 ABSA Analysis** first.")
    st.stop()

st.markdown(
    f"""
Each aspect score is the share of favourable opinion:

```text
score = (positive + {settings.neutral_weight} x neutral) / mentions
```

Scores are then pulled toward the corpus mean (shrinkage strength
`{settings.shrinkage_strength}`) so a phone with two mentions cannot outrank one with
two hundred on noise alone. Aspects with fewer than
`{settings.min_mentions_for_score}` mentions are treated as weak evidence.
"""
)

if st.button("Recompute aspect scores", type="primary"):
    try:
        with st.spinner("Aggregating aspect sentiment into scoresâ€¦"):
            with session_scope() as db:
                scores = recompute_aspect_scores(db, None, settings)
        st.success(f"Recomputed scores for {len(scores)} phone(s).")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Aggregation failed: {exc}")

apply_shrinkage = st.checkbox(
    "Apply shrinkage", value=True, help="Untick to see the raw ratio per aspect."
)

with session_scope() as db:
    vectors = build_feature_vectors(db, None, apply_shrinkage=apply_shrinkage, settings=settings)

if not vectors:
    st.info("No feature vectors yet. Recompute the scores above.")
    st.stop()

aspects = aspects_for(settings.aspect_set)

rows = []
for vector in vectors:
    row = {
        "phone": vector.name,
        "brand": vector.brand or "Unknown",
        "price": vector.price,
        "reviews": vector.review_count,
        "mentions": vector.mention_count,
    }
    for aspect in aspects:
        row[ASPECT_LABELS.get(aspect, aspect)] = vector.scores.get(aspect)
    rows.append(row)

table = pd.DataFrame(rows).sort_values("mentions", ascending=False).reset_index(drop=True)

st.subheader("Feature score matrix")
score_columns = [ASPECT_LABELS.get(aspect, aspect) for aspect in aspects]
st.dataframe(
    table,
    width="stretch",
    hide_index=True,
    column_config={
        "price": st.column_config.NumberColumn("price", format="%.2f"),
        **{
            column: st.column_config.ProgressColumn(
                column, format="%.2f", min_value=0.0, max_value=1.0
            )
            for column in score_columns
        },
    },
)
st.caption("Blank cells mean the aspect was never mentioned for that phone.")

st.subheader("Evidence per aspect")
coverage = []
for aspect in aspects:
    label = ASPECT_LABELS.get(aspect, aspect)
    mentions = sum(vector.mentions.get(aspect, 0) for vector in vectors)
    phones_with = sum(1 for vector in vectors if vector.scores.get(aspect) is not None)
    observed = [
        vector.scores[aspect] for vector in vectors if vector.scores.get(aspect) is not None
    ]
    coverage.append(
        {
            "aspect": label,
            "phones with a score": phones_with,
            "total mentions": mentions,
            "mean score": round(sum(observed) / len(observed), 3) if observed else None,
        }
    )
st.dataframe(pd.DataFrame(coverage), width="stretch", hide_index=True)

st.download_button(
    "Download the score table (CSV)",
    table.to_csv(index=False).encode("utf-8"),
    file_name="smartphone_aspect_scores.csv",
    mime="text/csv",
)

st.info("Next: rank phones against your requirements on **5 Recommendations**.")
