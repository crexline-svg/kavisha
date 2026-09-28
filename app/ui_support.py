"""Small helpers shared by the Streamlit pages.

Kept out of ``pages/`` so Streamlit does not treat it as a page.
"""

from __future__ import annotations

from typing import Any

from app.core.database import init_db
from app.core.logging import setup_logging

_BOOTSTRAPPED = False


def bootstrap() -> None:
    """Create tables and configure logging once per Streamlit session."""
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    setup_logging()
    init_db()
    _BOOTSTRAPPED = True


def stage_progress(placeholder: Any, bar: Any | None = None):
    """Adapt the ``(done, total, message)`` callback used across the services."""

    def report(done: int, total: int, message: str) -> None:
        placeholder.write(message)
        if bar is not None and total:
            bar.progress(min(max(done / total, 0.0), 1.0))

    return report


def sidebar_status(st: Any, status: dict[str, Any]) -> None:
    """Compact stage status so every page shows where the pipeline stands."""
    with st.sidebar:
        st.markdown("### Pipeline status")
        st.caption(f"ABSA engine: `{status['engine']}`")
        st.write(
            f"Candidates: **{status['candidates']}**  \n"
            f"Selected: **{status['selected']}**  \n"
            f"Corpus phones: **{status['corpus_phones']}**  \n"
            f"Phones in database: **{status['db_phones']}**  \n"
            f"Reviews analysed: **{status['db_reviews_analysed']}**  \n"
            f"Phones scored: **{status['db_scored_phones']}**"
        )


def format_price(price: float | None, currency: str | None = None) -> str:
    if price is None:
        return "n/a"
    symbols = {"USD": "$", "EUR": "€", "GBP": "£", "INR": "₹"}
    prefix = symbols.get((currency or "").upper(), "")
    if prefix:
        return f"{prefix}{price:,.0f}"
    suffix = f" {currency}" if currency else ""
    return f"{price:,.2f}{suffix}"


def star_line(rating: float | None, count: int | None = None) -> str:
    if rating is None:
        return ""
    full = int(rating)
    stars = "★" * full + "☆" * (5 - full)
    suffix = f" ({count:,})" if count else ""
    return f"{stars} {rating:.1f}{suffix}"


def render_phone_tile(
    st: Any,
    *,
    rank: int | None = None,
    name: str,
    brand: str | None = None,
    price: float | None = None,
    currency: str | None = None,
    image_url: str | None = None,
    site_rating: float | None = None,
    site_rating_count: int | None = None,
    score: float | None = None,
    review_count: int | None = None,
    subtitle: str | None = None,
) -> None:
    """Amazon-style product tile with image, title, rating, and price."""
    with st.container(border=True):
        if image_url:
            st.image(image_url, use_container_width=True)
        else:
            initial = (brand or name or "?")[:1].upper()
            st.markdown(
                f"<div style='text-align:center;padding:2.5rem 0;background:#1a2331;"
                f"border-radius:8px;font-size:2rem;font-weight:700;color:#6f8098'>{initial}</div>",
                unsafe_allow_html=True,
            )

        heading = f"#{rank} · {name}" if rank is not None else name
        st.markdown(f"**{heading}**")
        if brand:
            st.caption(brand)
        rating_text = star_line(site_rating, site_rating_count)
        if rating_text:
            st.markdown(rating_text)
        st.markdown(f"**{format_price(price, currency)}**")
        meta_parts = []
        if review_count is not None:
            meta_parts.append(f"{review_count:,} analysed reviews")
        if score is not None:
            meta_parts.append(f"match score **{score:.3f}**")
        if subtitle:
            meta_parts.append(subtitle)
        if meta_parts:
            st.caption(" · ".join(meta_parts))
