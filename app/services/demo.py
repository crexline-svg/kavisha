"""Synthetic corpus generator for offline development and testing.

Lets you exercise Steps 1-6, the API and the recommender without touching a
marketplace. Reviews are assembled from templates using a seeded RNG, so the
dataset is reproducible. Each phone has a deliberately different aspect profile
so rankings are meaningful, and the star rating is derived from the sampled
sentence polarities, which gives the /validation endpoint something to check.

This is test scaffolding, not research data. Never report results from it.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.models.entities import PriceObservation, Review, Smartphone
from app.nlp.preprocess import preprocess_review

logger = get_logger(__name__)

# aspect -> probability that a mention is positive
DEMO_PHONES: list[dict[str, object]] = [
    {
        "asin": "DEMOSAMS01",
        "brand": "Samsung",
        "model": "Galaxy S25",
        "title": "Samsung Galaxy S25 5G Smartphone, 256GB, Titanium Grey, Unlocked",
        "price": 899.0,
        "specs": {"ram_gb": 12.0, "storage_gb": 256.0, "display_inches": 6.2,
                  "battery_mah": 4000, "rear_camera_mp": 50.0, "refresh_rate_hz": 120,
                  "chipset": "Snapdragon 8 Elite", "operating_system": "Android 15"},
        "profile": {"battery": 0.85, "camera": 0.65, "display": 0.90,
                    "performance": 0.91, "price": 0.72},
    },
    {
        "asin": "DEMOAPPL02",
        "brand": "Apple",
        "model": "iPhone 17",
        "title": "Apple iPhone 17 128GB Black Unlocked Smartphone",
        "price": 1099.0,
        "specs": {"ram_gb": 8.0, "storage_gb": 128.0, "display_inches": 6.1,
                  "battery_mah": 3600, "rear_camera_mp": 48.0, "refresh_rate_hz": 120,
                  "chipset": "A19 Bionic", "operating_system": "iOS 19"},
        "profile": {"battery": 0.80, "camera": 0.94, "display": 0.91,
                    "performance": 0.95, "price": 0.58},
    },
    {
        "asin": "DEMOXIAO03",
        "brand": "Xiaomi",
        "model": "Xiaomi 15",
        "title": "Xiaomi 15 5G 512GB Dual SIM Smartphone, Green",
        "price": 649.0,
        "specs": {"ram_gb": 16.0, "storage_gb": 512.0, "display_inches": 6.36,
                  "battery_mah": 5400, "rear_camera_mp": 50.0, "refresh_rate_hz": 120,
                  "chipset": "Snapdragon 8 Gen 4", "operating_system": "Android 15"},
        "profile": {"battery": 0.90, "camera": 0.76, "display": 0.82,
                    "performance": 0.84, "price": 0.91},
    },
    {
        "asin": "DEMOGOOG04",
        "brand": "Google",
        "model": "Pixel 10",
        "title": "Google Pixel 10 128GB Obsidian Unlocked Android Smartphone",
        "price": 799.0,
        "specs": {"ram_gb": 12.0, "storage_gb": 128.0, "display_inches": 6.3,
                  "battery_mah": 4700, "rear_camera_mp": 50.0, "refresh_rate_hz": 120,
                  "chipset": "Tensor G5", "operating_system": "Android 16"},
        "profile": {"battery": 0.68, "camera": 0.93, "display": 0.84,
                    "performance": 0.74, "price": 0.77},
    },
]

TEMPLATES: dict[str, dict[str, list[str]]] = {
    "battery": {
        "positive": [
            "The battery easily lasts two full days on a single charge.",
            "Battery life is excellent, I end the day at 45 percent.",
            "Fast charging is brilliant, it tops up in under an hour.",
            "Battery backup is amazing even with heavy use.",
        ],
        "negative": [
            "The battery drains far too quickly during video calls.",
            "Battery life is poor, it dies by early evening.",
            "Charging is painfully slow compared to my old phone.",
            "The battery barely lasts a working day.",
        ],
        "neutral": [
            "It has a large battery and charges over USB-C.",
            "Battery is okay, nothing special either way.",
        ],
    },
    "camera": {
        "positive": [
            "The camera captures stunning detail even in low light.",
            "Photos are crisp and the colours look natural.",
            "Video stabilisation is superb for handheld shots.",
            "Portrait mode produces gorgeous results.",
        ],
        "negative": [
            "The camera performs poorly in low light and photos come out grainy.",
            "Pictures look washed out and the zoom is blurry.",
            "Night shots are disappointing and full of noise.",
            "The selfie camera is mediocre at best.",
        ],
        "neutral": [
            "It has a triple camera setup on the back.",
            "The camera app has the usual modes.",
        ],
    },
    "display": {
        "positive": [
            "The display is amazing, bright and vivid outdoors.",
            "Colours are punchy and the 120Hz refresh rate feels smooth.",
            "Screen quality is excellent with deep blacks.",
            "The panel is sharp and the touch response is instant.",
        ],
        "negative": [
            "The display is dim under sunlight and colours look dull.",
            "Screen has a noticeable green tint which is annoying.",
            "Touch response is unresponsive at the edges.",
        ],
        "neutral": [
            "It uses an AMOLED panel with a centred punch hole.",
        ],
    },
    "performance": {
        "positive": [
            "Performance is buttery smooth, no lag at all.",
            "The processor handles heavy games without heating up.",
            "Multitasking is seamless with plenty of RAM.",
            "It feels incredibly fast and responsive.",
        ],
        "negative": [
            "It lags badly when switching between apps.",
            "The phone overheats during gaming and throttles hard.",
            "Performance is sluggish and the UI stutters.",
            "Apps crash frequently which is frustrating.",
        ],
        "neutral": [
            "It ships with a flagship chipset and 12GB of RAM.",
        ],
    },
    "price": {
        "positive": [
            "Great value for money at this price point.",
            "Worth every penny considering the specifications.",
            "Very reasonably priced compared to the competition.",
            "An absolute bargain during the sale.",
        ],
        "negative": [
            "Far too expensive for what you actually get.",
            "It is overpriced and not worth the money.",
            "The price is hard to justify against cheaper rivals.",
        ],
        "neutral": [
            "The price sits in the usual flagship bracket.",
        ],
    },
}

FILLER = [
    "Delivery arrived a day early and the packaging was neat.",
    "I switched from an older model after three years.",
    "Overall I am happy with this purchase so far.",
    "Setup took about fifteen minutes.",
]

TITLES = [
    "Solid upgrade", "Mixed feelings", "Great phone with caveats",
    "Exactly what I wanted", "Would buy again", "Decent but flawed",
    "Impressive hardware", "Not for everyone",
]


def _sample_review(
    rng: random.Random, profile: dict[str, float]
) -> tuple[str, float]:
    """Build one review body and the star rating implied by its polarities."""
    aspects = list(profile)
    rng.shuffle(aspects)
    chosen = aspects[: rng.randint(2, min(4, len(aspects)))]

    sentences: list[str] = []
    polarities: list[float] = []

    for aspect in chosen:
        roll = rng.random()
        positive_probability = profile[aspect]
        if roll < positive_probability * 0.9:
            sentiment, value = "positive", 1.0
        elif roll < positive_probability * 0.9 + 0.12:
            sentiment, value = "neutral", 0.5
        else:
            sentiment, value = "negative", 0.0

        options = TEMPLATES[aspect][sentiment]
        sentences.append(rng.choice(options))
        polarities.append(value)

    if rng.random() < 0.4:
        sentences.insert(rng.randint(0, len(sentences)), rng.choice(FILLER))

    mean_polarity = sum(polarities) / len(polarities)
    # Map [0,1] onto 1..5 stars with a little noise, as real raters are noisy.
    rating = round(min(5, max(1, 1 + mean_polarity * 4 + rng.uniform(-0.5, 0.5))))

    return " ".join(sentences), float(rating)


def seed_demo_data(
    db: Session,
    reviews_per_phone: int = 60,
    seed: int = 42,
    settings: Settings | None = None,
) -> dict[str, int]:
    """Insert synthetic phones, prices and reviews. Idempotent per (asin, hash)."""
    settings = settings or get_settings()
    rng = random.Random(seed)

    created_phones = 0
    inserted_reviews = 0
    now = datetime.now(timezone.utc)

    for spec in DEMO_PHONES:
        asin = str(spec["asin"])
        phone = db.scalar(
            select(Smartphone).where(
                Smartphone.source == "demo", Smartphone.source_product_id == asin
            )
        )
        if phone is None:
            phone = Smartphone(source="demo", source_product_id=asin)
            db.add(phone)
            created_phones += 1

        phone.brand = str(spec["brand"])
        phone.model = str(spec["model"])
        phone.canonical_name = f"{spec['brand']} {spec['model']}"
        phone.raw_title = str(spec["title"])
        phone.product_url = f"https://example.invalid/dp/{asin}"
        phone.site_rating = 4.3
        phone.site_rating_count = reviews_per_phone * 30
        for key, value in dict(spec["specs"]).items():  # type: ignore[arg-type]
            setattr(phone, key, value)
        phone.last_seen_at = now
        db.flush()

        db.add(
            PriceObservation(
                smartphone_id=phone.id,
                price=float(spec["price"]),  # type: ignore[arg-type]
                list_price=float(spec["price"]) * 1.1,  # type: ignore[arg-type]
                currency="USD",
                discount_pct=9.09,
                availability="In Stock",
                source_url=phone.product_url,
            )
        )

        existing = set(
            db.scalars(select(Review.content_hash).where(Review.smartphone_id == phone.id)).all()
        )

        profile = dict(spec["profile"])  # type: ignore[arg-type]
        for index in range(reviews_per_phone):
            body, rating = _sample_review(rng, profile)
            title = rng.choice(TITLES)

            cleaned = preprocess_review(
                body,
                title=title,
                min_chars=settings.min_review_chars,
                language_filter=settings.language_filter or None,
            )
            if cleaned.content_hash in existing:
                continue
            existing.add(cleaned.content_hash)

            db.add(
                Review(
                    smartphone_id=phone.id,
                    source="demo",
                    source_review_id=f"{asin}-R{index:04d}",
                    source_url=phone.product_url,
                    title=title,
                    body=body,
                    cleaned_body=cleaned.cleaned_text or None,
                    content_hash=cleaned.content_hash,
                    rating=rating,
                    review_date=now - timedelta(days=rng.randint(1, 400)),
                    reviewer_name=f"Demo Reviewer {index}",
                    verified_purchase=rng.random() < 0.8,
                    helpful_votes=rng.randint(0, 40),
                    country="Demo",
                    language=cleaned.language,
                    is_spam=cleaned.is_spam,
                    excluded_reason=cleaned.excluded_reason,
                    pipeline_version=settings.pipeline_version,
                )
            )
            inserted_reviews += 1

    db.flush()
    logger.info(
        "Seeded %s new phone(s) and %s review(s) of synthetic data.",
        created_phones,
        inserted_reviews,
    )
    return {
        "phones_created": created_phones,
        "reviews_inserted": inserted_reviews,
        "phones_total": len(DEMO_PHONES),
    }


def ensure_demo_if_empty(
    settings: Settings | None = None,
    reviews_per_phone: int = 24,
) -> bool:
    """Seed and analyse the synthetic corpus when the database has no phones.

    Returns True when a new corpus was created. No-op unless ``SEED_DEMO`` is on.
    """
    settings = settings or get_settings()
    if not settings.seed_demo:
        return False

    from app.core.database import session_scope
    from app.services.analysis import run_analysis

    with session_scope() as db:
        existing = db.scalar(select(func.count()).select_from(Smartphone)) or 0
    if existing:
        return False

    logger.info("Empty database with SEED_DEMO enabled; loading sample phones.")
    with session_scope() as db:
        seed_demo_data(
            db,
            reviews_per_phone=reviews_per_phone,
            settings=settings,
        )
    run_analysis(settings=settings, engine_name="lexicon")
    return True


def purge_demo_data(db: Session) -> dict[str, int]:
    """Delete every synthetic phone and everything derived from it.

    Demo rows carry ``source == "demo"``, so this cannot touch scraped data.
    Prices, reviews, sentences and aspect records go with the phone via the
    ON DELETE CASCADE foreign keys.
    """
    phones = db.scalars(select(Smartphone).where(Smartphone.source == "demo")).all()
    if not phones:
        return {"phones_deleted": 0, "reviews_deleted": 0}

    phone_ids = [phone.id for phone in phones]
    reviews = db.scalar(
        select(func.count(Review.id)).where(Review.smartphone_id.in_(phone_ids))
    ) or 0

    for phone in phones:
        db.delete(phone)
    db.flush()

    logger.info("Purged %s demo phone(s) and %s review(s).", len(phones), reviews)
    return {"phones_deleted": len(phones), "reviews_deleted": int(reviews)}
