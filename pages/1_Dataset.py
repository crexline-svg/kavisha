"""Stage 1: Amazon metadata, smartphone selection, and review collection."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.services.workflow import (
    CANDIDATES_CSV_PATH,
    CORPUS_DIR,
    SELECTION_CSV_PATH,
    collect_reviews,
    filter_candidates,
    import_corpus,
    load_candidates,
    load_manifest,
    load_selection,
    pipeline_status,
    save_selection,
    scan_metadata,
)
from app.ui_support import bootstrap, sidebar_status, stage_progress

st.set_page_config(page_title="Dataset", page_icon="ðŸ“¦", layout="wide")
bootstrap()

st.title("1. Dataset")
st.caption("McAuley-Lab/Amazon-Reviews-2023 Â· Cell Phones and Accessories")
sidebar_status(st, pipeline_status())

st.markdown(
    """
Load **product metadata first**. The category is mostly cases, chargers and cables,
so candidates are filtered by keyword and then **you** confirm which listings are
real phones. Reviews are only collected for the phones you select.
"""
)

# --------------------------------------------------------------------------- #
st.header("1. Scan product metadata")

col_a, col_b = st.columns(2)
with col_a:
    metadata_limit = st.number_input(
        "Metadata rows to scan",
        min_value=500,
        max_value=2_000_000,
        value=20_000,
        step=500,
        help="This bounds the runtime. Only a fraction of these rows are phones.",
    )
with col_b:
    min_rating_count = st.number_input(
        "Minimum Amazon rating count",
        min_value=0,
        max_value=1_000_000,
        value=0,
        step=10,
        help="0 keeps unpopular phones too.",
    )

brands = st.multiselect(
    "Restrict to brands (optional)",
    ["samsung", "apple", "google", "xiaomi", "oneplus", "motorola", "nokia", "oppo", "vivo"],
    default=[],
)

if st.button("Scan metadata", type="primary"):
    bar = st.progress(0.0)
    line = st.empty()
    try:
        with st.spinner("Streaming product metadata from Hugging Faceâ€¦"):
            candidates = scan_metadata(
                metadata_limit=int(metadata_limit),
                min_rating_count=int(min_rating_count),
                brands=brands,
                progress=stage_progress(line, bar),
            )
        st.success(f"Saved {len(candidates)} candidate phone(s) to {CANDIDATES_CSV_PATH.name}.")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Metadata scan failed: {exc}")

# --------------------------------------------------------------------------- #
st.header("2. Review the candidates")

candidates = load_candidates()
if candidates.empty:
    st.info("No candidates yet. Run a metadata scan above.")
else:
    brand_options = ["All"] + sorted(candidates["brand"].dropna().astype(str).unique().tolist())
    filter_cols = st.columns(4)
    with filter_cols[0]:
        brand_choice = st.selectbox("Brand", brand_options)
    with filter_cols[1]:
        name_query = st.text_input("Search name")
    with filter_cols[2]:
        min_ratings_view = st.number_input("Min rating count", min_value=0, value=0, step=10)
    with filter_cols[3]:
        min_avg_view = st.slider("Min average rating", 0.0, 5.0, 0.0, 0.1)

    visible = filter_candidates(
        candidates,
        brand=brand_choice,
        name_query=name_query or None,
        min_rating_count=int(min_ratings_view),
        min_average_rating=float(min_avg_view),
    )
    st.write(f"Showing **{len(visible)}** of {len(candidates)} candidates.")
    st.dataframe(visible, width="stretch", hide_index=True)

    # ----------------------------------------------------------------------- #
    st.header("3. Select the phones to study")
    st.caption(
        "Keyword filtering is imperfect. Untick anything that is an accessory or a "
        "duplicate listing before collecting reviews."
    )

    preselect_count = st.number_input(
        "Pre-tick the most reviewed candidates",
        min_value=0,
        max_value=max(len(visible), 1),
        value=min(30, len(visible)),
        step=5,
    )
    labels = (
        visible["parent_asin"].astype(str) + " â€” " + visible["product_name"].astype(str)
    ).tolist()
    chosen = st.multiselect(
        "Smartphones to keep",
        options=labels,
        default=labels[: int(preselect_count)],
    )
    chosen_asins = [label.split(" â€” ", 1)[0] for label in chosen]
    chosen_rows = visible[visible["parent_asin"].astype(str).isin(chosen_asins)]

    if st.button(f"Save selection ({len(chosen_rows)} phone(s))"):
        try:
            save_selection(chosen_rows)
            st.success(f"Saved {len(chosen_rows)} phone(s) to {SELECTION_CSV_PATH.name}.")
        except ValueError as exc:
            st.error(str(exc))

# --------------------------------------------------------------------------- #
selection = load_selection()
if not selection.empty:
    st.header("4. Collect reviews for the selected phones")
    st.write(f"Current selection: **{len(selection)}** phone(s).")
    st.dataframe(selection, width="stretch", hide_index=True)

    st.warning(
        "This scans the Amazon review file. The first run downloads several GB into the "
        "Hugging Face cache and can take a long time. More selected phones means a longer scan."
    )

    review_cols = st.columns(2)
    with review_cols[0]:
        max_reviews = st.number_input(
            "Maximum reviews per phone", min_value=5, max_value=1000, value=60, step=5
        )
    with review_cols[1]:
        max_scan = st.number_input(
            "Stop after scanning this many review rows (0 = no limit)",
            min_value=0,
            max_value=200_000_000,
            value=0,
            step=100_000,
        )

    confirm = st.checkbox("I understand this can run for a long time")
    if st.button("Collect reviews", disabled=not confirm, type="primary"):
        bar = st.progress(0.0)
        line = st.empty()
        try:
            with st.spinner("Matching reviews to the selected ASINsâ€¦"):
                manifest = collect_reviews(
                    asins=selection["parent_asin"].astype(str).tolist(),
                    max_reviews_per_phone=int(max_reviews),
                    max_review_scan=int(max_scan),
                    progress=stage_progress(line, bar),
                )
            counts = manifest["counts"]
            st.success(
                f"Wrote {counts['phones_written']} phone(s) and "
                f"{counts['reviews_written']} review(s) to {CORPUS_DIR.name}."
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Review collection failed: {exc}")

manifest = load_manifest()
if manifest:
    st.subheader("Corpus built (Step 1 applied)")
    counts = manifest["counts"]
    metric_cols = st.columns(4)
    metric_cols[0].metric("Phones written", counts["phones_written"])
    metric_cols[1].metric("Reviews written", counts["reviews_written"])
    metric_cols[2].metric("Usable after Step 1", counts["reviews_usable"])
    metric_cols[3].metric("Review rows scanned", f"{counts['reviews_scanned']:,}")

    if counts.get("exclusion_breakdown"):
        st.write("Excluded by Step 1:", counts["exclusion_breakdown"])
    if counts.get("phones_without_reviews"):
        st.caption(
            f"{len(counts['phones_without_reviews'])} selected phone(s) had no matching "
            "reviews and were dropped. Raise the scan limit or pick better-known models."
        )
    st.dataframe(pd.DataFrame(manifest["phones"]), width="stretch", hide_index=True)

    # ----------------------------------------------------------------------- #
    st.header("5. Import into the database")
    st.caption(
        "Stop `python run.py serve` before importing â€” SQLite allows one writer. "
        "Replacing clears any previous corpus."
    )
    replace = st.checkbox("Replace the existing corpus", value=True)
    if st.button("Import corpus into database"):
        bar = st.progress(0.0)
        line = st.empty()
        try:
            with st.spinner("Writing phones and reviews into SQLiteâ€¦"):
                result = import_corpus(replace=replace, progress=stage_progress(line, bar))
            st.success(
                f"Imported {result['phones_upserted']} phone(s) and "
                f"{result['reviews_inserted']} review(s)."
            )
            st.json(result)
            st.info("Next: run **3 ABSA Analysis**.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Import failed: {exc}")
