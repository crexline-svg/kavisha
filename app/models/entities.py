"""Database tables, one per stage of the methodology.

    Smartphone         -> scraped product identity + specs
    PriceObservation   -> price history (a phone can be re-scraped over time)
    Review             -> raw + cleaned review text (Step 1)
    Sentence           -> sentence segmentation output (Step 2)
    AspectSentiment    -> aspect extraction + sentiment classification (Steps 3-4, Step 5 record)
    AspectScore        -> aggregated per-phone aspect scores (Step 6, output layer)
    Job                -> background scrape/analysis run bookkeeping
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Smartphone(Base):
    __tablename__ = "smartphones"
    __table_args__ = (
        UniqueConstraint("source", "source_product_id", name="uq_phone_source_product"),
        Index("ix_phone_brand_model", "brand", "model"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)

    source: Mapped[str] = mapped_column(String(32), default="amazon", index=True)
    source_product_id: Mapped[str] = mapped_column(String(64), index=True)
    product_url: Mapped[str | None] = mapped_column(Text)

    raw_title: Mapped[str | None] = mapped_column(Text)
    brand: Mapped[str | None] = mapped_column(String(64), index=True)
    model: Mapped[str | None] = mapped_column(String(128), index=True)
    canonical_name: Mapped[str | None] = mapped_column(String(200), index=True)
    image_url: Mapped[str | None] = mapped_column(Text)

    # Marketplace-reported aggregate rating, useful as a validation baseline.
    site_rating: Mapped[float | None] = mapped_column(Float)
    site_rating_count: Mapped[int | None] = mapped_column(Integer)

    # Normalised specs (best-effort parse); full table kept in `specs_raw`.
    ram_gb: Mapped[float | None] = mapped_column(Float)
    storage_gb: Mapped[float | None] = mapped_column(Float)
    display_inches: Mapped[float | None] = mapped_column(Float)
    battery_mah: Mapped[int | None] = mapped_column(Integer)
    rear_camera_mp: Mapped[float | None] = mapped_column(Float)
    refresh_rate_hz: Mapped[int | None] = mapped_column(Integer)
    chipset: Mapped[str | None] = mapped_column(String(128))
    operating_system: Mapped[str | None] = mapped_column(String(64))
    release_year: Mapped[int | None] = mapped_column(Integer)

    specs_raw: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    feature_bullets: Mapped[list[str] | None] = mapped_column(JSON)

    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    prices: Mapped[list["PriceObservation"]] = relationship(
        back_populates="smartphone", cascade="all, delete-orphan", passive_deletes=True
    )
    reviews: Mapped[list["Review"]] = relationship(
        back_populates="smartphone", cascade="all, delete-orphan", passive_deletes=True
    )
    aspect_scores: Mapped[list["AspectScore"]] = relationship(
        back_populates="smartphone", cascade="all, delete-orphan", passive_deletes=True
    )

    @property
    def display_name(self) -> str:
        return self.canonical_name or self.raw_title or f"{self.source}:{self.source_product_id}"


class PriceObservation(Base):
    __tablename__ = "price_observations"
    __table_args__ = (Index("ix_price_phone_time", "smartphone_id", "observed_at"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    smartphone_id: Mapped[int] = mapped_column(
        ForeignKey("smartphones.id", ondelete="CASCADE"), index=True
    )

    price: Mapped[float | None] = mapped_column(Float)
    list_price: Mapped[float | None] = mapped_column(Float)
    currency: Mapped[str | None] = mapped_column(String(8))
    discount_pct: Mapped[float | None] = mapped_column(Float)
    availability: Mapped[str | None] = mapped_column(String(128))
    source_url: Mapped[str | None] = mapped_column(Text)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    smartphone: Mapped[Smartphone] = relationship(back_populates="prices")


class Review(Base):
    __tablename__ = "reviews"
    __table_args__ = (
        # Deterministic de-duplication (Step 1): identical text for the same phone
        # can only be stored once, regardless of how often we re-scrape.
        UniqueConstraint("smartphone_id", "content_hash", name="uq_review_phone_hash"),
        Index("ix_review_phone_processed", "smartphone_id", "processed_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    smartphone_id: Mapped[int] = mapped_column(
        ForeignKey("smartphones.id", ondelete="CASCADE"), index=True
    )

    source: Mapped[str] = mapped_column(String(32), default="amazon")
    source_review_id: Mapped[str | None] = mapped_column(String(128), index=True)
    source_url: Mapped[str | None] = mapped_column(Text)

    title: Mapped[str | None] = mapped_column(Text)
    body: Mapped[str] = mapped_column(Text)
    cleaned_body: Mapped[str | None] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(64), index=True)

    rating: Mapped[float | None] = mapped_column(Float)
    review_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    reviewer_name: Mapped[str | None] = mapped_column(String(200))
    verified_purchase: Mapped[bool | None] = mapped_column(Boolean)
    helpful_votes: Mapped[int | None] = mapped_column(Integer)
    variant: Mapped[str | None] = mapped_column(String(200))
    country: Mapped[str | None] = mapped_column(String(64))

    language: Mapped[str | None] = mapped_column(String(8), index=True)
    # Step 1 exclusion flags, kept rather than deleted so the dissertation can
    # report exactly how many reviews were filtered and why.
    is_duplicate: Mapped[bool] = mapped_column(Boolean, default=False)
    is_spam: Mapped[bool] = mapped_column(Boolean, default=False)
    excluded_reason: Mapped[str | None] = mapped_column(String(64))

    scraped_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    pipeline_version: Mapped[str | None] = mapped_column(String(32))

    smartphone: Mapped[Smartphone] = relationship(back_populates="reviews")
    sentences: Mapped[list["Sentence"]] = relationship(
        back_populates="review", cascade="all, delete-orphan", passive_deletes=True
    )

    @property
    def is_usable(self) -> bool:
        return not (self.is_duplicate or self.is_spam) and self.excluded_reason is None


class Sentence(Base):
    __tablename__ = "sentences"
    __table_args__ = (
        UniqueConstraint("review_id", "position", name="uq_sentence_review_position"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    review_id: Mapped[int] = mapped_column(
        ForeignKey("reviews.id", ondelete="CASCADE"), index=True
    )
    smartphone_id: Mapped[int] = mapped_column(
        ForeignKey("smartphones.id", ondelete="CASCADE"), index=True
    )

    position: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    char_count: Mapped[int] = mapped_column(Integer, default=0)
    word_count: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    review: Mapped[Review] = relationship(back_populates="sentences")
    aspect_sentiments: Mapped[list["AspectSentiment"]] = relationship(
        back_populates="sentence", cascade="all, delete-orphan", passive_deletes=True
    )


class AspectSentiment(Base):
    """One (aspect, sentiment) pair extracted from one sentence — Step 5 record."""

    __tablename__ = "aspect_sentiments"
    __table_args__ = (
        Index("ix_as_phone_aspect", "smartphone_id", "aspect"),
        Index("ix_as_aspect_sentiment", "aspect", "sentiment"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    sentence_id: Mapped[int] = mapped_column(
        ForeignKey("sentences.id", ondelete="CASCADE"), index=True
    )
    review_id: Mapped[int] = mapped_column(
        ForeignKey("reviews.id", ondelete="CASCADE"), index=True
    )
    smartphone_id: Mapped[int] = mapped_column(
        ForeignKey("smartphones.id", ondelete="CASCADE"), index=True
    )

    aspect: Mapped[str] = mapped_column(String(32), index=True)
    sentiment: Mapped[str] = mapped_column(String(16), index=True)
    confidence: Mapped[float | None] = mapped_column(Float)
    opinion_term: Mapped[str | None] = mapped_column(String(200))

    method: Mapped[str] = mapped_column(String(16), default="llm")
    model_name: Mapped[str | None] = mapped_column(String(96))
    pipeline_version: Mapped[str | None] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    sentence: Mapped[Sentence] = relationship(back_populates="aspect_sentiments")


class AspectScore(Base):
    """Aggregated score per (phone, aspect) — Step 6 / output layer."""

    __tablename__ = "aspect_scores"
    __table_args__ = (
        UniqueConstraint("smartphone_id", "aspect", name="uq_score_phone_aspect"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    smartphone_id: Mapped[int] = mapped_column(
        ForeignKey("smartphones.id", ondelete="CASCADE"), index=True
    )

    aspect: Mapped[str] = mapped_column(String(32), index=True)

    positive_count: Mapped[int] = mapped_column(Integer, default=0)
    negative_count: Mapped[int] = mapped_column(Integer, default=0)
    neutral_count: Mapped[int] = mapped_column(Integer, default=0)
    mention_count: Mapped[int] = mapped_column(Integer, default=0)

    raw_score: Mapped[float | None] = mapped_column(Float)
    score: Mapped[float | None] = mapped_column(Float)
    confidence: Mapped[float | None] = mapped_column(Float)

    method: Mapped[str | None] = mapped_column(String(16))
    pipeline_version: Mapped[str | None] = mapped_column(String(32))
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    smartphone: Mapped[Smartphone] = relationship(back_populates="aspect_scores")


class Job(Base):
    """Background scrape / analyse run, so long operations are pollable over HTTP."""

    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    job_type: Mapped[str] = mapped_column(String(32), index=True)
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)

    params: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON)

    progress: Mapped[int] = mapped_column(Integer, default=0)
    total: Mapped[int] = mapped_column(Integer, default=0)
    message: Mapped[str | None] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class RecommendationFeedback(Base):
    """Step 13 — user satisfaction after seeing Top-N recommendations."""

    __tablename__ = "recommendation_feedback"

    id: Mapped[int] = mapped_column(primary_key=True)
    satisfaction: Mapped[int] = mapped_column(Integer)  # 1–5 overall (mean of phone ratings)
    comment: Mapped[str | None] = mapped_column(Text)

    # Snapshot of the recommendation request / result for evaluation (Step 14).
    weights_used: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    top_phone_ids: Mapped[list[Any] | None] = mapped_column(JSON)
    top_phone_names: Mapped[list[Any] | None] = mapped_column(JSON)
    # Per recommended phone: [{smartphone_id, name, rank, satisfaction, final_score}]
    phone_ratings: Mapped[list[Any] | None] = mapped_column(JSON)
    candidates_considered: Mapped[int | None] = mapped_column(Integer)
    session_id: Mapped[str | None] = mapped_column(String(64), index=True)
    ranking_method: Mapped[str | None] = mapped_column(String(32), default="weighted")

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
