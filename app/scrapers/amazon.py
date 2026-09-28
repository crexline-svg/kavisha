"""Amazon smartphone scraper: search -> product details + price -> reviews.

Selectors are grouped in fallback tuples because Amazon runs several page
templates concurrently and rotates them by region and A/B bucket. Every
extraction walks its tuple until one selector yields text, so a single template
change degrades one field instead of breaking the run.

Note on review coverage: Amazon now gates the full paginated review list behind a
signed-in session in most locales. `python run.py login` opens a real browser once
so you can sign in manually; the session is reused afterwards. Without it the
scraper still collects the reviews embedded on the product page and warns you.
"""

from __future__ import annotations

import re
from urllib.parse import quote_plus, urljoin

from bs4 import BeautifulSoup

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.scrapers.base import (
    BrowserSession,
    ScrapedPrice,
    ScrapedProduct,
    ScrapedReview,
    ScraperError,
    fetch_html,
    human_delay,
    save_debug_artifact,
)
from app.scrapers.parsers import (
    build_canonical_name,
    extract_brand,
    extract_model,
    normalize_specs,
    parse_helpful_votes,
    parse_price,
    parse_rating,
    parse_rating_count,
    parse_review_date,
)

logger = get_logger(__name__)

SEARCH_RESULT_SELECTORS = (
    'div[data-component-type="s-search-result"]',
    "div.s-result-item[data-asin]",
)

TITLE_SELECTORS = ("#productTitle", "#title span", "h1#title")

# Some product page variants ship no #productTitle (and no <h1> at all), so the
# document title is the last resort. Format: "Amazon.com: <name> : <category>".
_TITLE_SITE_PREFIX = re.compile(r"^\s*Amazon(?:\.[a-z]{2,3})+\s*[:|-]\s*", re.IGNORECASE)
_TITLE_SITE_SUFFIX = re.compile(r"\s*[-|]\s*Amazon(?:\.[a-z]{2,3})+\s*$", re.IGNORECASE)

PRICE_SELECTORS = (
    "#corePriceDisplay_desktop_feature_div span.priceToPay span.a-offscreen",
    "#corePriceDisplay_desktop_feature_div span.a-price span.a-offscreen",
    "#corePrice_feature_div span.a-offscreen",
    "span.priceToPay span.a-offscreen",
    "#apex_desktop span.a-price span.a-offscreen",
    "#priceblock_ourprice",
    "#priceblock_dealprice",
    "#priceblock_saleprice",
    "#price_inside_buybox",
    "span.a-price span.a-offscreen",
)

LIST_PRICE_SELECTORS = (
    "#corePriceDisplay_desktop_feature_div span.a-price.a-text-price span.a-offscreen",
    "span[data-a-strike='true'] span.a-offscreen",
    ".basisPrice span.a-offscreen",
    "#listPrice",
    "#priceblock_listprice",
)

AVAILABILITY_SELECTORS = ("#availability span", "#availability", "#outOfStock .a-color-price")

IMAGE_SELECTORS = ("#landingImage", "#imgBlkFront", "#main-image", "#ebooksImgBlkFront")

RATING_SELECTORS = (
    "#acrPopover",
    "span[data-hook='rating-out-of-text']",
    "#averageCustomerReviews .a-icon-alt",
    "i.a-icon-star .a-icon-alt",
)

RATING_COUNT_SELECTORS = (
    "#acrCustomerReviewText",
    "span[data-hook='total-review-count']",
    "#averageCustomerReviews #acrCustomerReviewLink",
)

BYLINE_SELECTORS = ("#bylineInfo", "#brand", "a#bylineInfo")

REVIEW_CONTAINER_SELECTORS = (
    "div[data-hook='review']",
    "div.review[data-hook]",
    "li[data-hook='review']",
)

SIGNIN_MARKERS = ("/ap/signin", "ap_email", "auth-error-message-box", "signin-form")

# Listings that are accessories rather than handsets.
_ACCESSORY_PATTERN = re.compile(
    r"\b(?:case|cover|screen\s*protector|tempered\s*glass|charger|charging\s*cable|"
    r"cable|adapter|holder|mount|stand|popsocket|pop\s*socket|skin|decal|sticker|"
    r"lens\s*protector|earbud|earbuds|headphone|headset|power\s*bank|stylus|"
    r"sim\s*tray|replacement\s*(?:battery|screen|lcd|digitizer)|repair\s*kit|"
    r"tripod|selfie\s*stick|gimbal|car\s*charger|wireless\s*charger|"
    r"tempered|lanyard|strap|pouch|sleeve|wallet\s*case)\b",
    re.IGNORECASE,
)

_PHONE_SIGNAL_PATTERN = re.compile(
    r"\b(?:smartphone|smart\s*phone|cell\s*phone|cellphone|mobile\s*phone|unlocked|"
    r"dual\s*sim|sim\s*free|\d+\s*GB\b|\d+\s*TB\b|5G|4G\s*LTE|android\s*\d|"
    r"factory\s*unlocked|international\s*version)\b",
    re.IGNORECASE,
)

_ASIN_PATTERN = re.compile(r"^[A-Z0-9]{10}$")
_LRM = re.compile(r"[\u200e\u200f\u061c]")


# --------------------------------------------------------------------------- #
# Small soup helpers
# --------------------------------------------------------------------------- #
def _clean(text: str | None) -> str | None:
    if text is None:
        return None
    cleaned = _LRM.sub("", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or None


def _first_text(node, selectors: tuple[str, ...]) -> str | None:  # noqa: ANN001
    for selector in selectors:
        for element in node.select(selector):
            text = _clean(element.get_text(" ", strip=True))
            if text:
                return text
    return None


def _first_attr(node, selectors: tuple[str, ...], *attrs: str) -> str | None:  # noqa: ANN001
    for selector in selectors:
        for element in node.select(selector):
            for attr in attrs:
                value = element.get(attr)
                if value:
                    return _clean(value if isinstance(value, str) else " ".join(value))
    return None


def looks_like_phone(title: str | None) -> bool:
    """Filter accessories out of search results."""
    if not title:
        return False
    if _PHONE_SIGNAL_PATTERN.search(title):
        # Strong accessory words still win ("Case for Galaxy S24 5G, 128GB slot").
        head = title[:60]
        return not re.search(
            r"\b(?:case|cover|screen\s*protector|charger|cable|earbud|headphone)\b",
            head,
            re.IGNORECASE,
        )
    return not _ACCESSORY_PATTERN.search(title)


# --------------------------------------------------------------------------- #
# Page parsers (pure functions over rendered HTML)
# --------------------------------------------------------------------------- #
def parse_search_results(html: str, base_url: str) -> list[dict[str, object]]:
    soup = BeautifulSoup(html, "lxml")

    cards = []
    for selector in SEARCH_RESULT_SELECTORS:
        cards = soup.select(selector)
        if cards:
            break

    results: list[dict[str, object]] = []
    seen: set[str] = set()

    for card in cards:
        asin = _clean(card.get("data-asin") or "")
        if not asin or not _ASIN_PATTERN.match(asin) or asin in seen:
            continue

        title = _first_text(
            card,
            (
                "h2 a span",
                "h2 span",
                "h2",
                ".a-size-medium.a-color-base.a-text-normal",
                ".a-size-base-plus.a-color-base.a-text-normal",
            ),
        )
        if not looks_like_phone(title):
            logger.debug("Skipping non-phone result: %s", title)
            continue

        price_text = _first_text(card, ("span.a-price span.a-offscreen", "span.a-price"))
        price, currency = parse_price(price_text)

        seen.add(asin)
        results.append(
            {
                "asin": asin,
                "title": title,
                "url": urljoin(base_url, f"/dp/{asin}"),
                "image_url": _first_attr(card, ("img.s-image",), "src"),
                "rating": parse_rating(_first_attr(card, ("span[aria-label*='out of 5']",), "aria-label")
                                       or _first_text(card, ("i.a-icon-star-small .a-icon-alt",))),
                "rating_count": parse_rating_count(
                    _first_text(card, ("span[aria-label][data-csa-c-func-deps]", "span.a-size-base.s-underline-text"))
                ),
                "price": price,
                "currency": currency,
                "sponsored": bool(card.select_one("span.s-label-popover-default, .puis-sponsored-label-text")),
            }
        )

    return results


def _extract_spec_tables(soup: BeautifulSoup) -> dict[str, str]:
    """Collect key/value pairs from every spec layout Amazon uses."""
    specs: dict[str, str] = {}

    row_selectors = (
        "#productDetails_techSpec_section_1 tr",
        "#productDetails_techSpec_section_2 tr",
        "#productDetails_detailBullets_sections1 tr",
        "#productOverview_feature_div tr",
        "#technicalSpecifications_section_1 tr",
        "table.a-keyvalue tr",
        "#prodDetails tr",
    )
    for selector in row_selectors:
        for row in soup.select(selector):
            header = row.select_one("th, td.a-span3, td:first-child")
            value_cell = row.select_one("td.a-span9, td:last-child")
            if header is None or value_cell is None or header is value_cell:
                continue
            key = _clean(header.get_text(" ", strip=True))
            value = _clean(value_cell.get_text(" ", strip=True))
            if key and value and len(key) < 80:
                specs.setdefault(key.rstrip(":"), value)

    # Detail bullets use "Key ‏ : ‎ Value" inside a single list item.
    for item in soup.select("#detailBullets_feature_div li span.a-list-item, #detailBulletsWrapper_feature_div li span.a-list-item"):
        text = _clean(item.get_text(" ", strip=True))
        if not text or ":" not in text:
            continue
        key, _, value = text.partition(":")
        key, value = _clean(key), _clean(value)
        if key and value and len(key) < 80:
            specs.setdefault(key, value)

    return specs


def _title_from_document(soup: BeautifulSoup) -> str | None:
    """Recover the product name from <title> when the usual elements are absent."""
    if not soup.title:
        return None
    text = _clean(soup.title.get_text(" ", strip=True))
    if not text:
        return None

    text = _TITLE_SITE_PREFIX.sub("", text)
    text = _TITLE_SITE_SUFFIX.sub("", text)

    # Drop the trailing " : <category>" while keeping any colon inside the name.
    head, separator, tail = text.rpartition(" : ")
    if separator and len(head) > len(tail):
        text = head

    return _clean(text) or None


def parse_product_page(html: str, asin: str, base_url: str) -> ScrapedProduct:
    soup = BeautifulSoup(html, "lxml")

    title = _first_text(soup, TITLE_SELECTORS) or _title_from_document(soup)
    byline = _first_text(soup, BYLINE_SELECTORS)
    specs = _extract_spec_tables(soup)

    bullets = []
    for item in soup.select("#feature-bullets li span.a-list-item, #featurebullets_feature_div li span.a-list-item"):
        text = _clean(item.get_text(" ", strip=True))
        if text and len(text) > 3 and "see more" not in text.lower():
            bullets.append(text)
    bullets = bullets[:20]

    price_text = _first_text(soup, PRICE_SELECTORS)
    price, currency = parse_price(price_text)
    list_price_text = _first_text(soup, LIST_PRICE_SELECTORS)
    list_price, list_currency = parse_price(list_price_text)

    discount_pct = None
    if price and list_price and list_price > price > 0:
        discount_pct = round(100 * (list_price - price) / list_price, 2)

    spec_brand = None
    spec_model = None
    for key, value in specs.items():
        lowered = key.lower()
        if spec_brand is None and lowered in ("brand", "brand name", "manufacturer"):
            spec_brand = value
        if spec_model is None and lowered in ("model name", "model", "item model number", "model number"):
            spec_model = value

    brand = extract_brand(title, byline, spec_brand)
    model = extract_model(title, brand, spec_model)

    rating_text = _first_attr(soup, RATING_SELECTORS, "title") or _first_text(soup, RATING_SELECTORS)
    availability = _first_text(soup, AVAILABILITY_SELECTORS)

    return ScrapedProduct(
        source="amazon",
        source_product_id=asin,
        product_url=urljoin(base_url, f"/dp/{asin}"),
        raw_title=title,
        brand=brand,
        model=model,
        canonical_name=build_canonical_name(brand, model, title),
        image_url=_first_attr(soup, IMAGE_SELECTORS, "data-old-hires", "src"),
        site_rating=parse_rating(rating_text),
        site_rating_count=parse_rating_count(_first_text(soup, RATING_COUNT_SELECTORS)),
        specs_raw=specs,
        feature_bullets=bullets,
        normalized_specs=normalize_specs(specs, title, bullets),
        price=ScrapedPrice(
            price=price,
            list_price=list_price,
            currency=currency or list_currency,
            discount_pct=discount_pct,
            availability=availability,
        ),
    )


def parse_reviews(html: str, base_url: str) -> list[ScrapedReview]:
    soup = BeautifulSoup(html, "lxml")

    containers = []
    for selector in REVIEW_CONTAINER_SELECTORS:
        containers = soup.select(selector)
        if containers:
            break

    reviews: list[ScrapedReview] = []

    for node in containers:
        body = _first_text(
            node,
            (
                "span[data-hook='review-body'] span",
                "div[data-hook='review-collapsed'] span",
                "span[data-hook='review-body']",
                ".review-text-content span",
            ),
        )
        if not body:
            continue

        # The title anchor contains the star rating as hidden text; take the last
        # non-empty span, which is the human-written headline.
        title = None
        title_node = node.select_one("a[data-hook='review-title'], span[data-hook='review-title']")
        if title_node is not None:
            spans = [_clean(s.get_text(" ", strip=True)) for s in title_node.select("span")]
            candidates = [s for s in spans if s and "out of 5 stars" not in s.lower()]
            title = candidates[-1] if candidates else _clean(title_node.get_text(" ", strip=True))
            if title and "out of 5 stars" in title.lower():
                title = _clean(re.sub(r".*out of 5 stars", "", title, flags=re.IGNORECASE))

        rating_text = (
            _first_text(node, ("i[data-hook='review-star-rating'] span.a-icon-alt",
                               "i[data-hook='cmps-review-star-rating'] span.a-icon-alt",
                               "span[data-hook='review-star-rating'] span.a-icon-alt",
                               "i.a-icon-star span.a-icon-alt"))
            or _first_attr(node, ("i[data-hook='review-star-rating']", "div.a-row.a-spacing-none i"), "class")
        )
        rating = parse_rating(rating_text)
        if rating is None and rating_text:
            star_class = re.search(r"a-star-(\d)", rating_text)
            if star_class:
                rating = float(star_class.group(1))

        date_text = _first_text(node, ("span[data-hook='review-date']",))
        review_date, country = parse_review_date(date_text)

        review_id = _clean(node.get("id")) or _first_attr(node, ("a[data-hook='review-title']",), "href")
        review_url = None
        href = _first_attr(node, ("a[data-hook='review-title']",), "href")
        if href:
            review_url = urljoin(base_url, href)

        reviews.append(
            ScrapedReview(
                body=body,
                source_review_id=review_id,
                source_url=review_url,
                title=title,
                rating=rating,
                review_date=review_date,
                reviewer_name=_first_text(node, ("span.a-profile-name",)),
                verified_purchase=bool(node.select_one("span[data-hook='avp-badge']")),
                helpful_votes=parse_helpful_votes(
                    _first_text(node, ("span[data-hook='helpful-vote-statement']",))
                ),
                variant=_first_text(node, ("a[data-hook='format-strip']", "div[data-hook='format-strip']")),
                country=country,
            )
        )

    return reviews


def requires_signin(html: str, url: str) -> bool:
    if any(marker in url for marker in SIGNIN_MARKERS):
        return True
    head = html[:15_000].lower()
    return "ap_email" in head or ("sign in" in head and "id=\"authportal" in head)


# --------------------------------------------------------------------------- #
# Scraper
# --------------------------------------------------------------------------- #
class AmazonScraper:
    source = "amazon"

    def __init__(self, session: BrowserSession, settings: Settings | None = None) -> None:
        self.session = session
        self.settings = settings or get_settings()
        self.base_url = self.settings.marketplace.rstrip("/")
        self._signin_warned = False

    # -------------------- search -------------------- #
    async def search(self, query: str, max_results: int = 5) -> list[dict[str, object]]:
        """Return candidate phone listings for a search term."""
        collected: list[dict[str, object]] = []
        seen: set[str] = set()
        page = await self.session.new_page()

        try:
            for page_number in range(1, 4):  # up to 3 result pages
                if len(collected) >= max_results:
                    break

                url = (
                    f"{self.base_url}/s?k={quote_plus(query)}"
                    f"&i=electronics&page={page_number}&ref=sr_pg_{page_number}"
                )
                logger.info("Searching '%s' (page %s)", query, page_number)
                html = await fetch_html(
                    page,
                    url,
                    settings=self.settings,
                    wait_selector=SEARCH_RESULT_SELECTORS[0],
                )

                results = parse_search_results(html, self.base_url)
                if not results:
                    await save_debug_artifact(page, f"search-empty-{query}", self.settings)
                    logger.warning("No parsable results for '%s' on page %s.", query, page_number)
                    break

                for item in results:
                    asin = str(item["asin"])
                    if asin in seen:
                        continue
                    seen.add(asin)
                    collected.append(item)
                    if len(collected) >= max_results:
                        break

                await human_delay(self.settings)
        finally:
            await page.close()

        logger.info("Found %s phone listing(s) for '%s'.", len(collected), query)
        return collected[:max_results]

    # -------------------- product -------------------- #
    async def fetch_product(self, asin: str) -> ScrapedProduct:
        page = await self.session.new_page()
        try:
            url = f"{self.base_url}/dp/{asin}"
            logger.info("Fetching product %s", asin)
            html = await fetch_html(
                page, url, settings=self.settings, wait_selector="#productTitle"
            )
            product = parse_product_page(html, asin, self.base_url)
            if not product.raw_title:
                await save_debug_artifact(page, f"product-{asin}", self.settings)
                raise ScraperError(f"Could not parse a product title for {asin}.")

            # Reviews rendered on the product page are a guaranteed baseline.
            product.inline_reviews = parse_reviews(html, self.base_url)
            return product
        finally:
            await page.close()

    # -------------------- reviews -------------------- #
    async def fetch_reviews(
        self,
        asin: str,
        *,
        max_reviews: int = 100,
        max_pages: int = 10,
        sort: str = "recent",
    ) -> list[ScrapedReview]:
        """Paginate the dedicated review listing, newest or most helpful first."""
        if max_reviews <= 0 or max_pages <= 0:
            return []

        sort_by = "recent" if sort == "recent" else "helpful"
        collected: list[ScrapedReview] = []
        seen_ids: set[str] = set()
        page = await self.session.new_page()

        try:
            for page_number in range(1, max_pages + 1):
                if len(collected) >= max_reviews:
                    break

                url = (
                    f"{self.base_url}/product-reviews/{asin}/"
                    f"?ie=UTF8&reviewerType=all_reviews&formatType=all_formats"
                    f"&filterByStar=all_stars&sortBy={sort_by}&pageNumber={page_number}"
                )

                try:
                    html = await fetch_html(
                        page,
                        url,
                        settings=self.settings,
                        wait_selector=REVIEW_CONTAINER_SELECTORS[0],
                    )
                except ScraperError as exc:
                    logger.warning("Review page %s for %s failed: %s", page_number, asin, exc)
                    break

                if requires_signin(html, page.url):
                    if not self._signin_warned:
                        logger.warning(
                            "Amazon requires a signed-in session for the full review list. "
                            "Run 'python run.py login' once to save a session, then re-run. "
                            "Continuing with product-page reviews only."
                        )
                        self._signin_warned = True
                    break

                page_reviews = parse_reviews(html, self.base_url)
                if not page_reviews:
                    if page_number == 1:
                        await save_debug_artifact(page, f"reviews-empty-{asin}", self.settings)
                    logger.info("No more reviews for %s at page %s.", asin, page_number)
                    break

                new_count = 0
                for review in page_reviews:
                    key = review.source_review_id or review.body[:120]
                    if key in seen_ids:
                        continue
                    seen_ids.add(key)
                    collected.append(review)
                    new_count += 1
                    if len(collected) >= max_reviews:
                        break

                logger.info(
                    "%s: page %s yielded %s new review(s) (total %s).",
                    asin, page_number, new_count, len(collected),
                )

                # Amazon serves the last page repeatedly instead of 404ing.
                if new_count == 0:
                    break

                await human_delay(self.settings)
        finally:
            await page.close()

        return collected[:max_reviews]

    async def resolve_targets(
        self, queries: list[str], product_ids: list[str], max_per_query: int
    ) -> list[str]:
        """Turn search terms + explicit ASINs into a de-duplicated ASIN list."""
        asins: list[str] = []
        seen: set[str] = set()

        for asin in product_ids:
            candidate = asin.strip().upper()
            if _ASIN_PATTERN.match(candidate) and candidate not in seen:
                seen.add(candidate)
                asins.append(candidate)
            elif not _ASIN_PATTERN.match(candidate):
                logger.warning("Ignoring '%s': not a valid 10-character ASIN.", asin)

        for query in queries:
            for item in await self.search(query, max_per_query):
                candidate = str(item["asin"])
                if candidate not in seen:
                    seen.add(candidate)
                    asins.append(candidate)

        return asins
