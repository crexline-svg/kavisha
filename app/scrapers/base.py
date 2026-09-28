"""Playwright browser management shared by all scrapers.

Design choices worth noting for the methodology write-up:

* Playwright renders the page (marketplace listings are JavaScript-driven and
  plain HTTP clients get served stripped or blocked pages), but parsing is done
  with BeautifulSoup on the rendered HTML. That is far faster than issuing dozens
  of round-trip locator queries per page and keeps selectors in one place.
* Static assets are blocked, which typically cuts page weight by ~80%.
* Every navigation is retried with exponential backoff, and randomised delays are
  inserted between requests to stay polite and avoid rate limiting.
* Bot-check pages are detected explicitly and surfaced as a clear error instead of
  silently producing empty datasets.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from datetime import datetime
from types import TracebackType
from typing import Any

from app.core.config import Settings, get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

BLOCKED_RESOURCE_TYPES = {"image", "media", "font", "stylesheet"}

BLOCKED_URL_FRAGMENTS = (
    "google-analytics", "googletagmanager", "doubleclick", "facebook.net",
    "amazon-adsystem", "scorecardresearch", "criteo", "hotjar", "adsystem",
    "/ape/", "unagi.amazon", "fls-na.amazon", "/gp/overlays/",
)

BOT_CHECK_MARKERS = (
    "enter the characters you see below",
    "type the characters you see in this image",
    "sorry, we just need to make sure you're not a robot",
    "api-services-support@amazon.com",
    "robot check",
    "to discuss automated access to amazon data",
)

# A modern desktop fingerprint; the UA must match the headers Chromium sends.
STEALTH_INIT_SCRIPT = """
Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
Object.defineProperty(navigator, 'plugins', {
  get: () => [1, 2, 3, 4, 5].map(() => ({ name: 'Chrome PDF Plugin' })),
});
window.chrome = window.chrome || { runtime: {} };
const originalQuery = window.navigator.permissions && window.navigator.permissions.query;
if (originalQuery) {
  window.navigator.permissions.query = (parameters) =>
    parameters && parameters.name === 'notifications'
      ? Promise.resolve({ state: Notification.permission })
      : originalQuery(parameters);
}
Object.defineProperty(navigator, 'hardwareConcurrency', { get: () => 8 });
Object.defineProperty(navigator, 'deviceMemory', { get: () => 8 });
"""


class ScraperError(RuntimeError):
    pass


class BotCheckError(ScraperError):
    """Raised when the marketplace serves a CAPTCHA / robot-check page."""


# --------------------------------------------------------------------------- #
# Scraped payloads (plain data, decoupled from the ORM)
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class ScrapedPrice:
    price: float | None = None
    list_price: float | None = None
    currency: str | None = None
    discount_pct: float | None = None
    availability: str | None = None


@dataclass(slots=True)
class ScrapedProduct:
    source: str
    source_product_id: str
    product_url: str | None = None
    raw_title: str | None = None
    brand: str | None = None
    model: str | None = None
    canonical_name: str | None = None
    image_url: str | None = None
    site_rating: float | None = None
    site_rating_count: int | None = None
    specs_raw: dict[str, str] = field(default_factory=dict)
    feature_bullets: list[str] = field(default_factory=list)
    normalized_specs: dict[str, Any] = field(default_factory=dict)
    price: ScrapedPrice | None = None
    # Reviews rendered directly on the product page; a guaranteed baseline even
    # when the paginated review listing requires a signed-in session.
    inline_reviews: list["ScrapedReview"] = field(default_factory=list)


@dataclass(slots=True)
class ScrapedReview:
    body: str
    source_review_id: str | None = None
    source_url: str | None = None
    title: str | None = None
    rating: float | None = None
    review_date: datetime | None = None
    reviewer_name: str | None = None
    verified_purchase: bool | None = None
    helpful_votes: int | None = None
    variant: str | None = None
    country: str | None = None


# --------------------------------------------------------------------------- #
# Browser session
# --------------------------------------------------------------------------- #
class BrowserSession:
    """Async context manager owning the Playwright browser and page factory."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        headless: bool | None = None,
        use_storage_state: bool = True,
    ) -> None:
        self.settings = settings or get_settings()
        self.headless = self.settings.headless if headless is None else headless
        self.use_storage_state = use_storage_state
        self._playwright = None
        self._browser = None
        self._context = None

    async def __aenter__(self) -> "BrowserSession":
        from playwright.async_api import async_playwright

        self._playwright = await async_playwright().start()

        launch_kwargs: dict[str, Any] = {
            "headless": self.headless,
            "args": [
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--no-sandbox",
                "--disable-gpu",
                "--window-size=1440,900",
            ],
        }
        if self.settings.browser_channel:
            launch_kwargs["channel"] = self.settings.browser_channel
        if self.settings.proxy_server:
            launch_kwargs["proxy"] = {"server": self.settings.proxy_server}

        try:
            self._browser = await self._playwright.chromium.launch(**launch_kwargs)
        except Exception as exc:
            await self._shutdown()
            raise ScraperError(
                "Could not launch Chromium. Run 'python -m playwright install chromium' "
                f"first. Original error: {exc}"
            ) from exc

        context_kwargs: dict[str, Any] = {
            "user_agent": self.settings.user_agent,
            "locale": self.settings.locale,
            "timezone_id": self.settings.timezone_id,
            "viewport": {"width": 1440, "height": 900},
            "device_scale_factor": 1,
            "java_script_enabled": True,
            "extra_http_headers": {
                "Accept-Language": f"{self.settings.locale},en;q=0.9",
                "Upgrade-Insecure-Requests": "1",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
            },
        }

        state_path = self.settings.storage_state_path
        if self.use_storage_state and state_path.exists():
            context_kwargs["storage_state"] = str(state_path)
            logger.info("Reusing saved browser session from %s", state_path.name)

        self._context = await self._browser.new_context(**context_kwargs)
        self._context.set_default_timeout(self.settings.nav_timeout_ms)
        self._context.set_default_navigation_timeout(self.settings.nav_timeout_ms)
        await self._context.add_init_script(STEALTH_INIT_SCRIPT)

        if self.settings.block_resources:
            await self._context.route("**/*", _block_heavy_resources)

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self._shutdown()

    async def _shutdown(self) -> None:
        for closer in (self._context, self._browser):
            if closer is not None:
                try:
                    await closer.close()
                except Exception:
                    pass
        if self._playwright is not None:
            try:
                await self._playwright.stop()
            except Exception:
                pass
        self._context = self._browser = self._playwright = None

    async def new_page(self):
        if self._context is None:
            raise ScraperError("BrowserSession must be used as an async context manager.")
        return await self._context.new_page()

    async def save_storage_state(self) -> None:
        if self._context is None:
            return
        path = self.settings.storage_state_path
        path.parent.mkdir(parents=True, exist_ok=True)
        await self._context.storage_state(path=str(path))
        logger.info("Saved browser session to %s", path)


async def _block_heavy_resources(route, request) -> None:  # noqa: ANN001
    if request.resource_type in BLOCKED_RESOURCE_TYPES:
        await route.abort()
        return
    url = request.url.lower()
    if any(fragment in url for fragment in BLOCKED_URL_FRAGMENTS):
        await route.abort()
        return
    await route.continue_()


# --------------------------------------------------------------------------- #
# Navigation helpers
# --------------------------------------------------------------------------- #
async def human_delay(settings: Settings | None = None, factor: float = 1.0) -> None:
    settings = settings or get_settings()
    low = max(0.0, settings.min_delay_s) * factor
    high = max(low, settings.max_delay_s * factor)
    await asyncio.sleep(random.uniform(low, high))


def detect_bot_check(html: str) -> bool:
    lowered = html[:20_000].lower()
    return any(marker in lowered for marker in BOT_CHECK_MARKERS)


async def dismiss_overlays(page) -> None:  # noqa: ANN001
    """Accept cookie banners and close interstitials that hide content."""
    for selector in (
        "#sp-cc-accept",
        "input[data-cel-widget='sp-cc-accept']",
        "button[name='glowDoneButton']",
        "input[data-action-type='DISMISS']",
        ".a-button-close",
    ):
        try:
            element = page.locator(selector).first
            if await element.count() and await element.is_visible(timeout=800):
                await element.click(timeout=1500)
                await asyncio.sleep(0.4)
        except Exception:
            continue


async def fetch_html(
    page,  # noqa: ANN001
    url: str,
    *,
    settings: Settings | None = None,
    wait_selector: str | None = None,
    wait_state: str = "domcontentloaded",
    allow_bot_check_retry: bool = True,
) -> str:
    """Navigate with retries and return the rendered HTML.

    Raises BotCheckError when the marketplace serves a robot check on every attempt.
    """
    settings = settings or get_settings()
    attempts = max(1, settings.max_retries)
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            response = await page.goto(url, wait_until=wait_state)
            status = response.status if response is not None else 0

            if status in (404, 410):
                raise ScraperError(f"Page not found ({status}): {url}")
            if status in (403, 429, 503):
                raise ScraperError(f"Blocked or throttled (HTTP {status}): {url}")

            await dismiss_overlays(page)

            if wait_selector:
                try:
                    await page.wait_for_selector(wait_selector, timeout=8_000)
                except Exception:
                    # Selector may legitimately be absent (e.g. a phone with no reviews).
                    logger.debug("Selector %r not found on %s", wait_selector, url)

            html = await page.content()

            if detect_bot_check(html):
                if allow_bot_check_retry and attempt < attempts:
                    logger.warning("Bot check served; backing off before retry %s.", attempt + 1)
                    await asyncio.sleep(random.uniform(8, 16) * attempt)
                    continue
                raise BotCheckError(
                    "The marketplace served a CAPTCHA / robot-check page. Slow the scraper "
                    "down (raise MIN_DELAY_S / MAX_DELAY_S), set HEADLESS=false, or run "
                    "'python run.py login' to establish a real session first."
                )

            return html

        except BotCheckError:
            raise
        except Exception as exc:
            last_error = exc
            if attempt >= attempts:
                break
            backoff = min(2 ** attempt + random.uniform(0, 1.5), 30)
            logger.warning(
                "Navigation to %s failed (%s); retry %s/%s in %.1fs.",
                url, exc, attempt + 1, attempts, backoff,
            )
            await asyncio.sleep(backoff)

    raise ScraperError(f"Failed to load {url} after {attempts} attempts: {last_error}")


async def save_debug_artifact(page, name: str, settings: Settings | None = None) -> None:  # noqa: ANN001
    """Dump a screenshot + HTML so broken selectors can be diagnosed offline."""
    settings = settings or get_settings()
    directory = settings.artifacts_dir
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)[:60]
    try:
        await page.screenshot(path=str(directory / f"{stamp}_{safe}.png"), full_page=False)
        (directory / f"{stamp}_{safe}.html").write_text(await page.content(), encoding="utf-8")
        logger.info("Saved debug artifacts for %s to %s", name, directory)
    except Exception as exc:
        logger.debug("Could not save debug artifact: %s", exc)
