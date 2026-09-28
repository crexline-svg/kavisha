"""Request/response models for the HTTP API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Sentiment = Literal["positive", "negative", "neutral"]


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# --------------------------------------------------------------------------- #
# Jobs
# --------------------------------------------------------------------------- #
class JobOut(ORMModel):
    id: str
    job_type: str
    status: str
    progress: int
    total: int
    message: str | None = None
    error: str | None = None
    params: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @property
    def percent(self) -> float:
        return round(100 * self.progress / self.total, 1) if self.total else 0.0


class JobAccepted(BaseModel):
    job_id: str
    status: str
    poll_url: str
    message: str


# --------------------------------------------------------------------------- #
# Scraping
# --------------------------------------------------------------------------- #
class ScrapeRequest(BaseModel):
    """Start a scrape. Provide `queries`, `product_ids`, or both."""

    queries: list[str] = Field(
        default_factory=list,
        description="Search terms, e.g. ['samsung galaxy s24', 'iphone 15'].",
        examples=[["samsung galaxy s24", "google pixel 8"]],
    )
    product_ids: list[str] = Field(
        default_factory=list,
        description="Known marketplace product IDs (Amazon ASINs) to scrape directly.",
    )
    max_phones_per_query: int = Field(5, ge=1, le=50)
    max_reviews_per_phone: int = Field(100, ge=0, le=2000)
    max_review_pages: int = Field(10, ge=0, le=100)
    review_sort: Literal["recent", "helpful"] = "recent"
    headless: bool | None = Field(None, description="Overrides the HEADLESS setting.")
    analyze_after_scrape: bool = Field(
        True, description="Automatically run the ABSA pipeline when scraping finishes."
    )

    @field_validator("queries", "product_ids")
    @classmethod
    def _strip(cls, values: list[str]) -> list[str]:
        return [v.strip() for v in values if v and v.strip()]

    def validate_targets(self) -> None:
        if not self.queries and not self.product_ids:
            raise ValueError("Provide at least one entry in 'queries' or 'product_ids'.")


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
class AnalyzeRequest(BaseModel):
    phone_ids: list[int] = Field(
        default_factory=list, description="Empty means every phone in the database."
    )
    max_reviews_per_phone: int | None = Field(
        None, ge=1, description="Cap reviews per phone (useful to limit LLM cost)."
    )
    force: bool = Field(
        False, description="Re-process reviews that already have ABSA results."
    )
    engine: Literal["auto", "llm", "lexicon"] | None = None


class AspectScoreOut(ORMModel):
    aspect: str
    positive_count: int
    negative_count: int
    neutral_count: int
    mention_count: int
    raw_score: float | None
    score: float | None
    confidence: float | None
    method: str | None
    computed_at: datetime


class AspectSentimentOut(ORMModel):
    id: int
    aspect: str
    sentiment: str
    confidence: float | None
    opinion_term: str | None
    method: str
    sentence_text: str | None = None


# --------------------------------------------------------------------------- #
# Phones
# --------------------------------------------------------------------------- #
class PriceOut(ORMModel):
    price: float | None
    list_price: float | None
    currency: str | None
    discount_pct: float | None
    availability: str | None
    observed_at: datetime


class PhoneSummary(ORMModel):
    id: int
    source: str
    source_product_id: str
    brand: str | None
    model: str | None
    canonical_name: str | None
    raw_title: str | None
    image_url: str | None
    product_url: str | None
    site_rating: float | None
    site_rating_count: int | None
    description: str | None = None
    latest_price: float | None = None
    currency: str | None = None
    review_count: int = 0
    analyzed_review_count: int = 0


class PhoneDetail(PhoneSummary):
    ram_gb: float | None = None
    storage_gb: float | None = None
    display_inches: float | None = None
    battery_mah: int | None = None
    rear_camera_mp: float | None = None
    refresh_rate_hz: int | None = None
    chipset: str | None = None
    operating_system: str | None = None
    release_year: int | None = None
    specs_raw: dict[str, Any] | None = None
    feature_bullets: list[str] | None = None
    first_seen_at: datetime
    last_seen_at: datetime
    price_history: list[PriceOut] = Field(default_factory=list)
    aspect_scores: list[AspectScoreOut] = Field(default_factory=list)


class ReviewOut(ORMModel):
    id: int
    smartphone_id: int
    title: str | None
    body: str
    cleaned_body: str | None
    rating: float | None
    review_date: datetime | None
    reviewer_name: str | None
    verified_purchase: bool | None
    helpful_votes: int | None
    language: str | None
    is_spam: bool
    is_duplicate: bool
    excluded_reason: str | None
    source_url: str | None
    scraped_at: datetime
    processed_at: datetime | None


class Paginated(BaseModel):
    total: int
    limit: int
    offset: int


class PhoneListResponse(Paginated):
    items: list[PhoneSummary]


class ReviewListResponse(Paginated):
    items: list[ReviewOut]


# --------------------------------------------------------------------------- #
# Feature vector (the "Smartphone Feature Score Database" output layer)
# --------------------------------------------------------------------------- #
class FeatureVector(BaseModel):
    smartphone_id: int
    name: str
    brand: str | None = None
    price: float | None = None
    currency: str | None = None
    review_count: int = 0
    mention_count: int = 0
    image_url: str | None = None
    product_url: str | None = None
    site_rating: float | None = None
    site_rating_count: int | None = None
    scores: dict[str, float | None] = Field(
        description="Aspect -> score in [0,1]. Null when the aspect was never mentioned."
    )
    mentions: dict[str, int] = Field(default_factory=dict)
    confidence: dict[str, float] = Field(default_factory=dict)
    affordability: float | None = Field(
        None, description="Numeric price mapped to [0,1] across the corpus; 1 = cheapest."
    )


# --------------------------------------------------------------------------- #
# Recommendation
# --------------------------------------------------------------------------- #
class AspectContribution(BaseModel):
    aspect: str
    score: float | None
    weight: float
    contribution: float
    mention_count: int
    imputed: bool = False


class Recommendation(BaseModel):
    rank: int
    smartphone_id: int
    name: str
    brand: str | None
    price: float | None
    currency: str | None
    image_url: str | None = None
    product_url: str | None = None
    site_rating: float | None = None
    site_rating_count: int | None = None
    final_score: float
    review_count: int
    coverage: float = Field(description="Share of requested weight backed by real mentions.")
    breakdown: list[AspectContribution]
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    explanation: str = Field(
        default="",
        description="Plain-language reason this phone was recommended.",
    )


class RecommendRequest(BaseModel):
    weights: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Aspect -> importance. Unlisted aspects get weight 0. "
            "Weights are normalised internally. Core set: battery, camera, "
            "display, performance, design, price."
        ),
        examples=[{"battery": 0.3, "camera": 0.25, "design": 0.2, "price": 0.25}],
    )
    phone_ids: list[int] = Field(default_factory=list)
    brands: list[str] = Field(default_factory=list)
    budget_min: float | None = Field(None, ge=0)
    budget_max: float | None = Field(None, ge=0)
    min_reviews: int = Field(0, ge=0, description="Exclude phones with fewer analysed reviews.")
    top_k: int = Field(10, ge=1, le=100)
    apply_shrinkage: bool = Field(
        True, description="Pull sparse aspect scores toward the corpus mean (reduces small-sample bias)."
    )
    method: Literal["weighted", "star_rating"] = Field(
        "weighted",
        description=(
            "weighted = user aspect priorities (proposed method). "
            "star_rating = Amazon average stars baseline for comparison."
        ),
    )


class RecommendResponse(BaseModel):
    weights_used: dict[str, float]
    candidates_considered: int
    results: list[Recommendation]
    method: str = "weighted"


class PhoneSatisfactionRating(BaseModel):
    """Per-phone graded fit for one recommended smartphone (1–5).

    Used both as user feedback and as graded relevance for NDCG@3 / Spearman.
    """

    smartphone_id: int
    name: str
    rank: int = Field(..., ge=1)
    satisfaction: int = Field(
        ...,
        ge=1,
        le=5,
        description="How well this phone fits the user's needs (1–5). Also used as graded relevance for ranking evaluation.",
    )
    final_score: float | None = None


class FeedbackRequest(BaseModel):
    satisfaction: int | None = Field(
        None,
        ge=1,
        le=5,
        description="Overall list satisfaction 1–5. If omitted, mean of phone_ratings is used.",
    )
    comment: str | None = Field(None, max_length=2000)
    weights_used: dict[str, float] = Field(default_factory=dict)
    top_phone_ids: list[int] = Field(default_factory=list)
    top_phone_names: list[str] = Field(default_factory=list)
    phone_ratings: list[PhoneSatisfactionRating] = Field(
        default_factory=list,
        description="Satisfaction 1–5 for each recommended smartphone.",
    )
    candidates_considered: int | None = None
    session_id: str | None = Field(None, max_length=64)
    ranking_method: str | None = Field(
        "weighted",
        description="weighted (proposed) or star_rating (Amazon baseline).",
    )


class FeedbackOut(ORMModel):
    id: int
    satisfaction: int
    comment: str | None
    weights_used: dict[str, Any] | None = None
    top_phone_ids: list[Any] | None = None
    top_phone_names: list[Any] | None = None
    phone_ratings: list[Any] | None = None
    candidates_considered: int | None = None
    session_id: str | None = None
    ranking_method: str | None = None
    created_at: datetime


# --------------------------------------------------------------------------- #
# Corpus statistics (for the dissertation's dataset description)
# --------------------------------------------------------------------------- #
class CorpusStats(BaseModel):
    phones: int
    reviews_total: int
    reviews_usable: int
    reviews_excluded: int
    exclusion_breakdown: dict[str, int]
    sentences: int
    aspect_sentiments: int
    sentiment_distribution: dict[str, int]
    aspect_distribution: dict[str, int]
    mean_sentences_per_review: float
    mean_aspects_per_sentence: float
    pipeline_version: str
    absa_engine: str
    currency_breakdown: dict[str, int] = Field(
        default_factory=dict,
        description="Phones per price currency. More than one entry means numeric "
        "prices are not directly comparable across the corpus.",
    )
    sources: list[str] = Field(default_factory=list)
    has_demo_data: bool = Field(
        default=False,
        description="True while synthetic seed-demo phones remain in the database.",
    )


class ValidationReport(BaseModel):
    """Agreement between star ratings and ABSA output — an internal validity check."""

    reviews_compared: int
    agreement_rate: float
    mean_absolute_error: float
    by_rating: dict[str, dict[str, float]]
    note: str


class EvaluationReport(BaseModel):
    """Step 14 — ABSA validity + recommendation satisfaction + ranking quality."""

    absa_engine: str
    absa_method_breakdown: dict[str, int]
    absa_validation: ValidationReport
    feedback_count: int
    phone_rating_count: int = 0
    mean_satisfaction: float | None
    satisfaction_distribution: dict[str, int]
    mean_satisfaction_by_rank: dict[str, float] = Field(default_factory=dict)
    top1_mean_satisfaction: float | None = None
    high_satisfaction_rate: float | None = Field(
        None,
        description="Share of phone ratings that are 4 or 5.",
    )
    evaluation_verdict: str | None = Field(
        None,
        description="Short quality label derived from mean phone satisfaction.",
    )
    # Ranking quality (per-phone 1–5 used as graded relevance)
    mean_ndcg_at_3: float | None = Field(
        None,
        description="Mean NDCG@3 across sessions that have per-phone ratings with ranks.",
    )
    ndcg_session_count: int = 0
    mean_spearman: float | None = Field(
        None,
        description="Mean Spearman correlation (system order vs user rating order).",
    )
    spearman_session_count: int = 0
    ranking_quality_note: str | None = None
    # Proposed vs Amazon-star baseline (from feedback sessions tagged by ranking_method)
    proposed_mean_satisfaction: float | None = None
    proposed_mean_ndcg_at_3: float | None = None
    proposed_session_count: int = 0
    baseline_mean_satisfaction: float | None = None
    baseline_mean_ndcg_at_3: float | None = None
    baseline_session_count: int = 0
    baseline_comparison_note: str | None = None
    # Gold-label ABSA technical metrics (from data/artifacts/absa_eval_latest.json)
    absa_gold_metrics: dict[str, Any] | None = None
    recent_feedback: list[FeedbackOut] = Field(default_factory=list)
