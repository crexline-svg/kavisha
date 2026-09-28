"""Offline tests for the scraper's parsing layer and the NLP pipeline.

The HTML fixtures reproduce the structure of Amazon's search, product and review
templates. They let the extraction logic be verified without any network traffic,
which also means these tests keep working when the live site changes.

Run with:  python -m pytest tests -q      (or: python tests/test_parsers.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.nlp.absa import LexiconAbsaEngine
from app.nlp.aggregate import affordability_index
from app.nlp.aspects import detect_aspects
from app.nlp.preprocess import compute_content_hash, preprocess_review
from app.nlp.segment import segment_sentences
from app.scrapers.amazon import (
    looks_like_phone,
    parse_product_page,
    parse_reviews,
    parse_search_results,
)
from app.scrapers.parsers import (
    extract_brand,
    extract_model,
    normalize_specs,
    parse_helpful_votes,
    parse_number,
    parse_price,
    parse_rating,
    parse_rating_count,
    parse_review_date,
)

BASE = "https://www.amazon.com"

SEARCH_HTML = """
<div data-component-type="s-search-result" data-asin="B0CMDRCZBJ">
  <h2><a href="/dp/B0CMDRCZBJ"><span>SAMSUNG Galaxy S24 Cell Phone, 128GB Unlocked Android Smartphone</span></a></h2>
  <img class="s-image" src="https://m.media-amazon.com/images/I/61abc.jpg"/>
  <span aria-label="4.3 out of 5 stars"></span>
  <span class="a-price"><span class="a-offscreen">$799.99</span></span>
</div>
<div data-component-type="s-search-result" data-asin="B0CACCESS01">
  <h2><span>Spigen Tough Armor Case for Galaxy S24, Military Grade Cover</span></h2>
  <span class="a-price"><span class="a-offscreen">$18.99</span></span>
</div>
<div data-component-type="s-search-result" data-asin="B0CMDRZZZZ">
  <h2><span>Google Pixel 8 Pro - 256GB 5G Smartphone, Obsidian</span></h2>
  <span class="a-price"><span class="a-offscreen">$899.00</span></span>
  <span class="puis-sponsored-label-text">Sponsored</span>
</div>
"""

PRODUCT_HTML = """
<span id="productTitle">SAMSUNG Galaxy S24 Ultra Cell Phone, 256GB Unlocked Android Smartphone, Titanium Gray</span>
<a id="bylineInfo">Visit the SAMSUNG Store</a>
<div id="corePriceDisplay_desktop_feature_div">
  <span class="priceToPay"><span class="a-offscreen">$1,199.99</span></span>
  <span class="a-price a-text-price"><span class="a-offscreen">$1,419.99</span></span>
</div>
<div id="availability"><span>In Stock</span></div>
<span id="acrPopover" title="4.5 out of 5 stars"></span>
<span id="acrCustomerReviewText">12,483 ratings</span>
<img id="landingImage" data-old-hires="https://m.media-amazon.com/images/I/big.jpg"/>
<div id="feature-bullets"><ul>
  <li><span class="a-list-item">6.8-inch QHD+ Dynamic AMOLED 2X display with 120Hz refresh rate</span></li>
  <li><span class="a-list-item">200MP wide camera with 5x optical zoom</span></li>
  <li><span class="a-list-item">5000mAh battery with 45W fast charging</span></li>
</ul></div>
<table id="productDetails_techSpec_section_1">
  <tr><th>Brand</th><td>SAMSUNG</td></tr>
  <tr><th>Model Name</th><td>Galaxy S24 Ultra</td></tr>
  <tr><th>Operating System</th><td>Android 14</td></tr>
  <tr><th>RAM</th><td>12 GB</td></tr>
  <tr><th>Memory Storage Capacity</th><td>256 GB</td></tr>
  <tr><th>Screen Size</th><td>6.8 Inches</td></tr>
  <tr><th>CPU Model</th><td>Snapdragon 8 Gen 3</td></tr>
  <tr><th>Date First Available</th><td>January 17, 2024</td></tr>
</table>
"""

REVIEWS_HTML = """
<div data-hook="review" id="R1ABCDEFG">
  <span class="a-profile-name">Alex</span>
  <i data-hook="review-star-rating"><span class="a-icon-alt">5.0 out of 5 stars</span></i>
  <a data-hook="review-title" href="/gp/customer-reviews/R1ABCDEFG">
    <span>5.0 out of 5 stars</span><span>Battery is a beast</span>
  </a>
  <span data-hook="review-date">Reviewed in the United States on March 5, 2024</span>
  <span data-hook="avp-badge">Verified Purchase</span>
  <span data-hook="review-body"><span>The battery lasts two days easily. Display is amazing, but the camera performs poorly in low light.</span></span>
  <span data-hook="helpful-vote-statement">23 people found this helpful</span>
</div>
<div data-hook="review" id="R2HIJKLMN">
  <span class="a-profile-name">Priya</span>
  <i data-hook="review-star-rating"><span class="a-icon-alt">2.0 out of 5 stars</span></i>
  <a data-hook="review-title" href="/gp/customer-reviews/R2HIJKLMN">
    <span>2.0 out of 5 stars</span><span>Overheats badly</span>
  </a>
  <span data-hook="review-date">Reviewed in India on 12 April 2024</span>
  <span data-hook="review-body"><span>It overheats during gaming and lags a lot. Not worth the price.</span></span>
  <span data-hook="helpful-vote-statement">One person found this helpful</span>
</div>
"""

_failures: list[str] = []


def check(condition: bool, label: str, detail: str = "") -> None:
    if condition:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label} {detail}")
        _failures.append(label)


def test_number_and_price_parsing() -> None:
    print("\nNumber / price parsing")
    check(parse_number("$1,199.00") == 1199.0, "US grouping 1,199.00")
    check(parse_number("1.299,00 EUR") == 1299.0, "EU grouping 1.299,00")
    check(parse_number("\u20b91,29,900") == 129900.0, "Indian lakh grouping")
    check(parse_number("799") == 799.0, "plain integer")

    value, currency = parse_price("$1,199.99")
    check(value == 1199.99 and currency == "USD", "USD symbol", f"got {value} {currency}")
    value, currency = parse_price("\u20b974,999.00")
    check(value == 74999.0 and currency == "INR", "INR symbol", f"got {value} {currency}")
    value, currency = parse_price("\u00a3999.00")
    check(currency == "GBP", "GBP symbol", f"got {currency}")

    # Amazon localises to the delivery country, which produced an unlabelled
    # 191191.0 price from a real Sri Lankan listing.
    value, currency = parse_price("LKR 191,191.00")
    check(value == 191191.0 and currency == "LKR", "LKR spaced", f"got {value} {currency}")
    value, currency = parse_price("LKR191,191.00")
    check(value == 191191.0 and currency == "LKR", "LKR flush against amount", f"got {value} {currency}")

    check(parse_rating("4.3 out of 5 stars") == 4.3, "rating out of 5")
    check(parse_rating("4,5 von 5 Sternen") == 4.5, "rating localized")
    check(parse_rating("nonsense") is None, "rating rejects noise")
    check(parse_rating_count("12,483 ratings") == 12483, "rating count grouped")
    check(parse_rating_count("1.2K ratings") == 1200, "rating count compact")
    check(parse_helpful_votes("One person found this helpful") == 1, "helpful votes 'One'")
    check(parse_helpful_votes("23 people found this helpful") == 23, "helpful votes numeric")


def test_review_date_parsing() -> None:
    print("\nReview date parsing")
    date, country = parse_review_date("Reviewed in the United States on March 5, 2024")
    check(
        date is not None and (date.year, date.month, date.day) == (2024, 3, 5),
        "US date",
        f"got {date}",
    )
    check(country == "United States", "US country", f"got {country}")

    date, country = parse_review_date("Reviewed in India on 12 April 2024")
    check(
        date is not None and (date.year, date.month, date.day) == (2024, 4, 12),
        "IN date",
        f"got {date}",
    )
    check(country == "India", "IN country", f"got {country}")


def test_brand_model_and_specs() -> None:
    print("\nBrand / model / spec normalisation")
    title = "SAMSUNG Galaxy S24 Ultra Cell Phone, 256GB Unlocked Android Smartphone, Titanium Gray"
    brand = extract_brand(title, "Visit the SAMSUNG Store", "SAMSUNG")
    check(brand == "Samsung", "brand from byline", f"got {brand}")

    model = extract_model(title, brand, "Galaxy S24 Ultra")
    check(model == "Galaxy S24 Ultra", "model from spec table", f"got {model}")

    model_from_title = extract_model(title, brand, None)
    check(
        "Galaxy S24 Ultra" in model_from_title,
        "model from title strips noise",
        f"got {model_from_title}",
    )

    check(extract_brand("Apple iPhone 15 Pro 256GB") == "Apple", "brand Apple")
    check(extract_brand("iPhone 15 Pro Max, 512GB") == "Apple", "brand inferred from iPhone")

    specs = {
        "RAM": "12 GB",
        "Memory Storage Capacity": "256 GB",
        "Screen Size": "6.8 Inches",
        "CPU Model": "Snapdragon 8 Gen 3",
        "Operating System": "Android 14",
        "Date First Available": "January 17, 2024",
    }
    bullets = ["5000mAh battery with 45W fast charging", "200MP wide camera", "120Hz refresh rate"]
    normalized = normalize_specs(specs, title, bullets)

    check(normalized.get("ram_gb") == 12.0, "ram_gb", f"got {normalized.get('ram_gb')}")
    check(normalized.get("storage_gb") == 256.0, "storage_gb", f"got {normalized.get('storage_gb')}")
    check(normalized.get("display_inches") == 6.8, "display_inches", f"got {normalized.get('display_inches')}")
    check(normalized.get("battery_mah") == 5000, "battery_mah", f"got {normalized.get('battery_mah')}")
    check(normalized.get("rear_camera_mp") == 200.0, "rear_camera_mp", f"got {normalized.get('rear_camera_mp')}")
    check(normalized.get("refresh_rate_hz") == 120, "refresh_rate_hz", f"got {normalized.get('refresh_rate_hz')}")
    check("Snapdragon" in (normalized.get("chipset") or ""), "chipset", f"got {normalized.get('chipset')}")
    check(normalized.get("operating_system") == "Android 14", "operating_system", f"got {normalized.get('operating_system')}")
    check(normalized.get("release_year") == 2024, "release_year", f"got {normalized.get('release_year')}")

    # A 1TB listing must become 1024 GB, not 1.
    tb = normalize_specs({"Memory Storage Capacity": "1 TB"}, "Phone 1TB storage", [])
    check(tb.get("storage_gb") == 1024.0, "1TB -> 1024GB", f"got {tb.get('storage_gb')}")


def test_accessory_filter() -> None:
    print("\nAccessory filtering")
    check(looks_like_phone("SAMSUNG Galaxy S24 128GB Unlocked Smartphone"), "keeps a phone")
    check(not looks_like_phone("Spigen Tough Armor Case for Galaxy S24"), "drops a case")
    check(not looks_like_phone("Screen Protector for iPhone 15, 3 Pack Tempered Glass"), "drops a protector")
    check(not looks_like_phone("65W USB-C Fast Charger Cable for Samsung"), "drops a charger")
    check(looks_like_phone("Google Pixel 8 Pro - 256GB 5G Smartphone"), "keeps a 5G phone")


def test_search_parsing() -> None:
    print("\nSearch result parsing")
    results = parse_search_results(SEARCH_HTML, BASE)
    asins = [r["asin"] for r in results]

    check(len(results) == 2, "two phones kept of three cards", f"got {len(results)}")
    check("B0CMDRCZBJ" in asins, "galaxy asin captured")
    check("B0CACCESS01" not in asins, "case asin filtered out")

    first = next(r for r in results if r["asin"] == "B0CMDRCZBJ")
    check(first["price"] == 799.99, "search price", f"got {first['price']}")
    check(first["currency"] == "USD", "search currency")
    check(first["rating"] == 4.3, "search rating", f"got {first['rating']}")
    check(first["url"].endswith("/dp/B0CMDRCZBJ"), "product url built")
    check(first["image_url"] is not None, "image captured")

    sponsored = next(r for r in results if r["asin"] == "B0CMDRZZZZ")
    check(sponsored["sponsored"] is True, "sponsored flag detected")


def test_product_parsing() -> None:
    print("\nProduct page parsing")
    product = parse_product_page(PRODUCT_HTML, "B0CMDRCZBJ", BASE)

    check(product.brand == "Samsung", "brand", f"got {product.brand}")
    check(product.model == "Galaxy S24 Ultra", "model", f"got {product.model}")
    check(product.canonical_name == "Samsung Galaxy S24 Ultra", "canonical name", f"got {product.canonical_name}")
    check(product.site_rating == 4.5, "site rating", f"got {product.site_rating}")
    check(product.site_rating_count == 12483, "rating count", f"got {product.site_rating_count}")
    check(len(product.feature_bullets) == 3, "feature bullets", f"got {len(product.feature_bullets)}")
    check(len(product.specs_raw) >= 7, "spec table rows", f"got {len(product.specs_raw)}")
    check(product.image_url is not None and "big.jpg" in product.image_url, "hi-res image")

    assert product.price is not None
    check(product.price.price == 1199.99, "current price", f"got {product.price.price}")
    check(product.price.list_price == 1419.99, "list price", f"got {product.price.list_price}")
    check(product.price.currency == "USD", "currency")


def test_affordability_is_currency_aware() -> None:
    """A single foreign-currency price used to flatten every other phone to ~1.0."""
    print("\nAffordability index")

    index = affordability_index({1: 649.0, 2: 899.0, 3: 1099.0})
    check(index[1] == 1.0, "cheapest scores 1.0", f"got {index[1]}")
    check(index[3] == 0.0, "priciest scores 0.0", f"got {index[3]}")
    check(0.4 < index[2] < 0.6, "mid-price in between", f"got {index[2]}")

    # Same USD phones, plus one LKR phone two orders of magnitude larger.
    mixed = affordability_index(
        {1: 649.0, 2: 899.0, 3: 1099.0, 4: 191191.0},
        {1: "USD", 2: "USD", 3: "USD", 4: "LKR"},
    )
    check(mixed[1] == 1.0, "USD group unaffected by LKR outlier", f"got {mixed[1]}")
    check(mixed[3] == 0.0, "USD spread preserved", f"got {mixed[3]}")
    check(4 not in mixed, "lone-currency phone left unscored", f"got {mixed.get(4)}")

    # Without currencies the old flattening is what we would get, so prove the
    # guard is what fixes it rather than some incidental change.
    naive = affordability_index({1: 649.0, 2: 899.0, 3: 1099.0, 4: 191191.0})
    check(naive[1] - naive[3] < 0.01, "unlabelled prices still flatten (expected)", f"got {naive}")

    check(affordability_index({}) == {}, "empty input")
    check(affordability_index({1: 500.0}) == {}, "single phone unscored")


def test_product_title_fallback() -> None:
    """A real listing served no #productTitle and no <h1>, only <title>."""
    print("\nProduct title fallback")
    html = """
    <html><head><title>Amazon.com: Samsung Galaxy S24+ Plus 5G, US Version, 512GB,
    Onyx Black - Unlocked (Renewed) : Cell Phones &amp; Accessories</title></head>
    <body><div id="dp">no product title element here</div></body></html>
    """
    product = parse_product_page(html, "B0D2RXT37G", BASE)

    check(
        product.raw_title == "Samsung Galaxy S24+ Plus 5G, US Version, 512GB, Onyx Black - Unlocked (Renewed)",
        "title recovered from <title>",
        f"got {product.raw_title!r}",
    )
    check(product.brand == "Samsung", "brand from fallback title", f"got {product.brand}")

    # A page with genuinely no title must still return None rather than junk.
    empty = parse_product_page("<html><body>nothing</body></html>", "X", BASE)
    check(empty.raw_title in (None, ""), "no title stays empty", f"got {empty.raw_title!r}")
    check(product.price.availability == "In Stock", "availability", f"got {product.price.availability}")
    check(
        product.price.discount_pct is not None and 15 < product.price.discount_pct < 16,
        "discount percent",
        f"got {product.price.discount_pct}",
    )
    check(product.normalized_specs.get("battery_mah") == 5000, "normalised battery from bullets")


def test_review_parsing() -> None:
    print("\nReview parsing")
    reviews = parse_reviews(REVIEWS_HTML, BASE)
    check(len(reviews) == 2, "two reviews parsed", f"got {len(reviews)}")

    first, second = reviews
    check(first.source_review_id == "R1ABCDEFG", "review id", f"got {first.source_review_id}")
    check(first.rating == 5.0, "rating 5", f"got {first.rating}")
    check(first.title == "Battery is a beast", "title excludes star text", f"got {first.title!r}")
    check("lasts two days" in first.body, "body captured")
    check(first.verified_purchase is True, "verified badge")
    check(first.helpful_votes == 23, "helpful votes", f"got {first.helpful_votes}")
    check(first.country == "United States", "country", f"got {first.country}")
    check(first.review_date is not None and first.review_date.year == 2024, "date parsed")
    check(first.source_url is not None and first.source_url.startswith(BASE), "absolute review url")

    check(second.rating == 2.0, "rating 2", f"got {second.rating}")
    check(second.verified_purchase is False, "unverified review")
    check(second.helpful_votes == 1, "'One person' -> 1")


def test_preprocessing() -> None:
    print("\nStep 1 preprocessing")
    raw = (
        "<span>The battery lasts    two days!!!! </span> Read more "
        "Visit https://example.com for details. 5 people found this helpful"
    )
    result = preprocess_review(raw, title="Great phone")
    check("Read more" not in result.cleaned_text, "boilerplate removed")
    check("https://" not in result.cleaned_text, "url removed")
    check("<span>" not in result.cleaned_text, "html stripped")
    check("!!!!" not in result.cleaned_text, "punctuation collapsed")
    check(result.cleaned_text.startswith("Great phone"), "title prepended", result.cleaned_text[:40])
    check(result.usable, "review usable")

    check(preprocess_review("Good").excluded_reason is not None, "too-short rejected")
    check(preprocess_review("good product").is_spam, "generic-only flagged as spam")
    check(
        preprocess_review("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA").is_spam,
        "repeated chars flagged",
    )

    # De-duplication must ignore case, punctuation and whitespace differences.
    a = compute_content_hash("Battery lasts two days.")
    b = compute_content_hash("battery   lasts two DAYS!!")
    check(a == b, "hash normalises formatting")
    check(a != compute_content_hash("Camera is poor."), "different text, different hash")


def test_segmentation() -> None:
    print("\nStep 2 segmentation")
    sentences = segment_sentences(
        "Battery lasts two days. Display is amazing. Camera is poor."
    )
    check(len(sentences) == 3, "three sentences", f"got {len(sentences)}: {sentences}")

    decimals = segment_sentences("The 6.7 inch screen is great. I paid 1,199.99 for it.")
    check(len(decimals) == 2, "decimals not split", f"got {decimals}")

    merged = segment_sentences("Nice. The battery on this phone lasts a very long time indeed.")
    check(len(merged) == 1, "stray fragment merged", f"got {merged}")


def test_absa_engine() -> None:
    print("\nSteps 3-4 lexicon ABSA")
    engine = LexiconAbsaEngine()

    def labels(sentence: str) -> dict[str, str]:
        return {o.aspect: o.sentiment for o in engine.analyze([sentence])[0]}

    result = labels("Battery lasts two days and the display is amazing.")
    check(result.get("battery") == "positive", "battery positive", f"got {result}")
    check(result.get("display") == "positive", "display positive", f"got {result}")

    result = labels("The camera performs poorly in low light.")
    check(result.get("camera") == "negative", "camera negative", f"got {result}")

    # Contrast: opposing polarities in one sentence must not bleed together.
    result = labels("The camera is great but it overheats during gaming.")
    check(result.get("camera") == "positive", "camera positive before 'but'", f"got {result}")
    check(result.get("performance") == "negative", "performance negative after 'but'", f"got {result}")

    result = labels("Not worth the price at all.")
    check(result.get("price") == "negative", "negated price", f"got {result}")

    result = labels("The battery does not drain quickly.")
    check(result.get("battery") == "positive", "negated negative -> positive", f"got {result}")

    result = labels("Delivery was late and the box was damaged.")
    check(result == {}, "no aspect for off-topic sentence", f"got {result}")

    detected = detect_aspects("It dies by lunchtime", ("battery", "camera"))
    check("battery" in detected, "implicit battery keyword", f"got {detected}")


def main() -> int:
    print("=" * 70)
    print("Offline test suite: scraper parsers + NLP pipeline")
    print("=" * 70)

    test_number_and_price_parsing()
    test_review_date_parsing()
    test_brand_model_and_specs()
    test_accessory_filter()
    test_search_parsing()
    test_product_parsing()
    test_review_parsing()
    test_preprocessing()
    test_segmentation()
    test_absa_engine()

    print("\n" + "=" * 70)
    if _failures:
        print(f"{len(_failures)} check(s) FAILED:")
        for name in _failures:
            print(f"  - {name}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
