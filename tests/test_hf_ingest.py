"""Offline tests for Hugging Face meta → phone filtering (no network)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.hf_ingest import is_smartphone_meta, meta_to_product, review_to_scraped


def test_filters_accessories_and_keeps_phones() -> None:
    phone = {
        "parent_asin": "B0TESTPHONE",
        "title": "Samsung Galaxy S23 5G Unlocked Android Smartphone 128GB",
        "average_rating": 4.4,
        "rating_number": 1200,
        "price": "699.99",
        "features": ["6.1-inch display", "50MP camera"],
        "categories": ["Cell Phones & Accessories", "Cell Phones", "Unlocked Cell Phones"],
        "details": '{"Brand": "Samsung", "Model Name": "Galaxy S23", "RAM": "8 GB", "Memory Storage Capacity": "128 GB"}',
        "images": {"hi_res": ["https://example.com/p.jpg"], "large": [], "thumb": [], "variant": ["MAIN"]},
    }
    case = {
        "parent_asin": "B0TESTCASE",
        "title": "Case for Samsung Galaxy S23 5G, Clear Cover",
        "average_rating": 4.8,
        "rating_number": 9000,
        "price": "12.99",
        "features": [],
        "categories": ["Cell Phones & Accessories", "Cases, Holsters & Clips"],
        "details": '{"Brand": "Spigen"}',
        "images": {},
    }
    band = {
        "parent_asin": "B0TESTBAND",
        "title": "Samsung LEOMARON Bands for Active 2 Watch Bands 44mm 40mm",
        "average_rating": 4.5,
        "rating_number": 2000,
        "price": "15.99",
        "features": [],
        "categories": ["Cell Phones & Accessories", "Accessories"],
        "details": '{"Brand": "Samsung"}',
        "images": {},
    }

    assert is_smartphone_meta(phone) is True
    assert is_smartphone_meta(case) is False
    assert is_smartphone_meta(band) is False

    product = meta_to_product(phone)
    assert product is not None
    assert product.source == "amazon_hf"
    assert product.brand == "Samsung"
    assert product.price is not None
    assert product.price.price == 699.99
    assert product.price.currency == "USD"
    assert meta_to_product(case) is None
    assert meta_to_product(band) is None


def test_review_row_maps() -> None:
    row = {
        "parent_asin": "B0TESTPHONE",
        "asin": "B0TESTPHONE",
        "user_id": "U123",
        "timestamp": 1_700_000_000_000,
        "title": "Great battery",
        "text": "The battery lasts all day and the camera is sharp.",
        "rating": 5.0,
        "verified_purchase": True,
        "helpful_vote": 3,
    }
    scraped = review_to_scraped(row)
    assert scraped is not None
    assert scraped.body.startswith("The battery")
    assert scraped.rating == 5.0
    assert scraped.verified_purchase is True
