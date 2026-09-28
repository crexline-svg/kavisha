"""Pure parsing helpers: prices, ratings, dates, specs, brand/model extraction.

These are deliberately free of Playwright and database imports so they can be
unit-tested against saved HTML fixtures without a browser.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

CURRENCY_BY_SYMBOL: dict[str, str] = {
    "$": "USD", "US$": "USD", "C$": "CAD", "A$": "AUD", "R$": "BRL",
    "\u20b9": "INR", "Rs.": "INR", "Rs": "INR", "\u20a8": "INR",
    "\u00a3": "GBP", "\u20ac": "EUR", "\u00a5": "JPY", "\u20a9": "KRW",
    "\u20ba": "TRY", "\u20aa": "ILS", "\u0631.\u0633": "SAR", "\u062f.\u0625": "AED",
    "kr": "SEK", "z\u0142": "PLN", "CHF": "CHF", "MX$": "MXN", "S$": "SGD",
    "RM": "MYR", "\u20b1": "PHP", "\u20ab": "VND", "\u0e3f": "THB",
}

_CURRENCY_CODE_PATTERN = re.compile(
    r"\b(USD|EUR|GBP|INR|JPY|AUD|CAD|SGD|AED|SAR|BRL|MXN|SEK|PLN|CHF|CNY|KRW|TRY|ZAR|MYR|PHP|THB|VND"
    # Amazon localises prices for the delivery country, so these show up too.
    r"|LKR|PKR|BDT|NPR|NGN|KES|EGP|NZD|HKD|TWD|IDR|CZK|DKK|NOK|HUF|RON|ILS|QAR|KWD|BHD|OMR)"
    # Not \b: Amazon also renders the code flush against the amount ("LKR191,191.00").
    r"(?![A-Z])"
)

_NUMBER_PATTERN = re.compile(r"\d[\d.,\u00a0\s]*\d|\d")

KNOWN_BRANDS: tuple[str, ...] = (
    "Samsung", "Apple", "Xiaomi", "Redmi", "Poco", "OnePlus", "Google", "Motorola",
    "Moto", "Nokia", "Sony", "LG", "Huawei", "Honor", "Oppo", "Vivo", "Realme",
    "Asus", "Nothing", "Lenovo", "ZTE", "Tecno", "Infinix", "iQOO", "Micromax",
    "Lava", "TCL", "Alcatel", "BlackBerry", "HTC", "Meizu", "Ulefone", "Doogee",
    "Blackview", "Cubot", "Umidigi", "Fairphone", "CAT", "Cat", "Kyocera",
    "Panasonic", "Sharp", "Gionee", "Itel", "Symphony", "Walton", "Cellecor",
)

_APPLE_HINTS = ("iphone", "ipad")

# Marketing noise that should not end up in the canonical model name.
_MODEL_NOISE = re.compile(
    r"\b(?:"
    r"unlocked|factory\s+unlocked|dual\s+sim|single\s+sim|sim\s+free|international\s+version|"
    r"global\s+version|us\s+version|renewed|refurbished|brand\s+new|latest\s+model|"
    r"smartphone|smart\s+phone|mobile\s+phone|cell\s+phone|cellphone|android\s+phone|"
    r"with\s+charger|no\s+charger|warranty|official|genuine|sealed|new\s+launch|"
    r"gsm\s+only|cdma|not\s+compatible|verizon|at&t|t-mobile|sprint|tracfone|"
    r"free\s+shipping|best\s+seller|pack\s+of\s+\d+"
    r")\b",
    re.IGNORECASE,
)

_STORAGE_IN_MODEL = re.compile(r"\b\d+\s?(?:GB|TB|MB)\b", re.IGNORECASE)
_COLOR_WORDS = re.compile(
    r"\b(?:black|white|blue|green|red|gold|silver|grey|gray|purple|pink|yellow|"
    r"orange|graphite|midnight|starlight|titanium|lavender|mint|cream|obsidian|"
    r"phantom|cosmic|aurora|onyx|sierra|space|jet|rose|coral|teal|violet|bronze)\b",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------- #
# Numbers, prices, ratings
# --------------------------------------------------------------------------- #
def parse_number(text: str | None) -> float | None:
    """Parse a localised number, handling both 1,234.56 and 1.234,56 groupings."""
    if not text:
        return None
    match = _NUMBER_PATTERN.search(text.replace("\u00a0", " "))
    if not match:
        return None

    raw = re.sub(r"[\s\u00a0]", "", match.group(0))
    has_comma, has_dot = "," in raw, "." in raw

    if has_comma and has_dot:
        # Whichever separator comes last is the decimal separator.
        decimal_sep = "," if raw.rfind(",") > raw.rfind(".") else "."
        thousands_sep = "." if decimal_sep == "," else ","
        raw = raw.replace(thousands_sep, "").replace(decimal_sep, ".")
    elif has_comma:
        tail = raw.split(",")[-1]
        # "1,50" is a decimal; "1,50,000" and "1,500" are grouped thousands.
        raw = raw.replace(",", ".") if (raw.count(",") == 1 and len(tail) == 2) else raw.replace(",", "")
    elif has_dot:
        tail = raw.split(".")[-1]
        if not (raw.count(".") == 1 and len(tail) in (1, 2)):
            raw = raw.replace(".", "")

    try:
        return float(raw)
    except ValueError:
        return None


def parse_currency(text: str | None, fallback: str | None = None) -> str | None:
    if not text:
        return fallback
    code_match = _CURRENCY_CODE_PATTERN.search(text.upper())
    if code_match:
        return code_match.group(1)
    for symbol in sorted(CURRENCY_BY_SYMBOL, key=len, reverse=True):
        if symbol in text:
            return CURRENCY_BY_SYMBOL[symbol]
    return fallback


def parse_price(text: str | None, fallback_currency: str | None = None) -> tuple[float | None, str | None]:
    if not text:
        return None, fallback_currency
    return parse_number(text), parse_currency(text, fallback_currency)


def parse_rating(text: str | None) -> float | None:
    """Parse '4.3 out of 5 stars' / '4,3 von 5 Sternen' / '4.3'."""
    if not text:
        return None

    # In a rating a comma is always a decimal mark ("4,3"), never a thousands
    # separator, so it is normalised before the generic number parser sees it.
    match = re.search(
        r"(\d+(?:[.,]\d+)?)\s*(?:out of|/|von|sur|su|de)\s*5", text, re.IGNORECASE
    )
    if match is None:
        match = re.fullmatch(r"\s*(\d+(?:[.,]\d+)?)\s*", text)

    if match is not None:
        try:
            value = float(match.group(1).replace(",", "."))
        except ValueError:
            return None
    else:
        value = parse_number(text)

    if value is None:
        return None
    return round(value, 2) if 0 <= value <= 5 else None


def parse_int(text: str | None) -> int | None:
    if not text:
        return None
    digits = re.sub(r"[^\d]", "", text)
    if not digits:
        return None
    try:
        return int(digits)
    except ValueError:
        return None


def parse_rating_count(text: str | None) -> int | None:
    """Parse '12,345 ratings' and '1.2K ratings'."""
    if not text:
        return None
    compact = re.search(r"(\d+(?:[.,]\d+)?)\s*([KkMm])\b", text)
    if compact:
        value = parse_number(compact.group(1)) or 0
        multiplier = 1_000 if compact.group(2).lower() == "k" else 1_000_000
        return int(value * multiplier)
    return parse_int(text)


# --------------------------------------------------------------------------- #
# Review metadata
# --------------------------------------------------------------------------- #
_DATE_FORMATS = (
    "%B %d, %Y", "%d %B %Y", "%b %d, %Y", "%d %b %Y",
    "%d.%m.%Y", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d",
)


def parse_review_date(text: str | None) -> tuple[datetime | None, str | None]:
    """Parse 'Reviewed in the United States on March 5, 2024' -> (date, country)."""
    if not text:
        return None, None

    cleaned = re.sub(r"\s+", " ", text).strip()

    country = None
    country_match = re.search(
        r"(?:reviewed|rezension|commentaire|recensito)\s+in\s+(?:the\s+)?(.+?)\s+on\s+",
        cleaned,
        re.IGNORECASE,
    )
    if country_match:
        country = country_match.group(1).strip().title()[:64]

    date_part = cleaned
    on_match = re.search(r"\bon\s+(.+)$", cleaned, re.IGNORECASE)
    if on_match:
        date_part = on_match.group(1).strip()
    date_part = date_part.rstrip(".").strip()

    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(date_part, fmt).replace(tzinfo=timezone.utc), country
        except ValueError:
            continue

    iso = re.search(r"(\d{4})-(\d{2})-(\d{2})", date_part)
    if iso:
        try:
            return datetime(
                int(iso.group(1)), int(iso.group(2)), int(iso.group(3)), tzinfo=timezone.utc
            ), country
        except ValueError:
            pass

    return None, country


def parse_helpful_votes(text: str | None) -> int | None:
    """'One person found this helpful' -> 1; '23 people found this helpful' -> 23."""
    if not text:
        return None
    lowered = text.lower()
    if lowered.startswith("one ") or " one person" in lowered:
        return 1
    return parse_int(text)


# --------------------------------------------------------------------------- #
# Brand / model
# --------------------------------------------------------------------------- #
def extract_brand(title: str | None, byline: str | None = None, spec_brand: str | None = None) -> str | None:
    """Return a manufacturer brand when possible.

    Carrier or seller names in the brand field (e.g. ``AT&T Prepaid``, ``Tmobile``)
    must not win over a real brand present in the product title (e.g. Apple iPhone).
    Unknown ``spec_brand`` values are only used as a last resort after title/byline.
    """
    deferred_unknown: str | None = None
    for candidate in (spec_brand, byline, title):
        if not candidate:
            continue
        text = re.sub(r"^(?:visit the|brand:)\s*", "", candidate.strip(), flags=re.IGNORECASE)
        text = re.sub(r"\s*store$", "", text, flags=re.IGNORECASE).strip()
        lowered = text.lower()
        for brand in KNOWN_BRANDS:
            if re.search(rf"(?<!\w){re.escape(brand.lower())}(?!\w)", lowered):
                if brand.lower() in {"apple", "moto"}:
                    return "Apple" if brand.lower() == "apple" else "Motorola"
                return brand
        if any(hint in lowered for hint in _APPLE_HINTS):
            return "Apple"
        # Do not return unknown seller/carrier brands before the title is checked.
        if candidate is spec_brand and text:
            deferred_unknown = text[:64]
    return deferred_unknown


def extract_model(title: str | None, brand: str | None = None, spec_model: str | None = None) -> str | None:
    """Pull a clean model name out of a marketplace title."""
    if spec_model and 2 < len(spec_model) < 60:
        candidate = spec_model
    elif title:
        # Titles are comma/paren separated: keep the informative head.
        candidate = re.split(r"[,(\[|\u2013\u2014]|\s-\s", title)[0]
    else:
        return None

    candidate = _MODEL_NOISE.sub(" ", candidate)
    candidate = _STORAGE_IN_MODEL.sub(" ", candidate)
    candidate = _COLOR_WORDS.sub(" ", candidate)
    candidate = re.sub(r"\b(?:\d+\s?GB\s?RAM|\d+\s?mAh|\d+\s?MP|\d+\s?Hz)\b", " ", candidate, flags=re.IGNORECASE)

    if brand:
        candidate = re.sub(rf"(?<!\w){re.escape(brand)}(?!\w)", " ", candidate, flags=re.IGNORECASE)

    candidate = re.sub(r"[^\w\s+.\-]", " ", candidate)
    candidate = re.sub(r"\s+", " ", candidate).strip(" -.+")

    return candidate[:128] or None


def build_canonical_name(brand: str | None, model: str | None, title: str | None) -> str | None:
    parts = [p for p in (brand, model) if p]
    if parts:
        return re.sub(r"\s+", " ", " ".join(parts)).strip()[:200]
    return (title or "").strip()[:200] or None


# --------------------------------------------------------------------------- #
# Spec normalisation
# --------------------------------------------------------------------------- #
def _search_all(patterns: tuple[str, ...], haystack: str) -> re.Match[str] | None:
    for pattern in patterns:
        match = re.search(pattern, haystack, re.IGNORECASE)
        if match:
            return match
    return None


def _spec_lookup(specs: dict[str, str], *needles: str) -> str | None:
    for key, value in specs.items():
        lowered_key = key.lower()
        if any(needle in lowered_key for needle in needles):
            if value and value.strip():
                return value.strip()
    return None


def _match_spec_first(
    patterns: tuple[str, ...], primary: str | None, haystack: str
) -> re.Match[str] | None:
    """Search the authoritative spec field before the general text.

    Without this, a pattern can straddle two adjacent spec entries and read one
    field's value as another's (e.g. "RAM: 12 GB | Memory Storage: 256 GB").
    """
    if primary:
        match = _search_all(patterns, primary)
        if match is not None:
            return match
    return _search_all(patterns, haystack)


def normalize_specs(
    specs: dict[str, str] | None,
    title: str | None = None,
    bullets: list[str] | None = None,
) -> dict[str, Any]:
    """Best-effort extraction of comparable numeric specs.

    Marketplace spec tables are inconsistent, so every field falls back to
    scanning the title and feature bullets. Missing values stay None rather than
    being guessed.
    """
    specs = specs or {}
    # Every field is separated by " | " so that no regex can match across two
    # unrelated values; a plain space would let "RAM: 12 GB Memory Storage: 256 GB"
    # be read as a 12 GB storage capacity.
    haystack = " | ".join(
        filter(
            None,
            [
                title or "",
                " | ".join(bullets or []),
                " | ".join(f"{k}: {v}" for k, v in specs.items()),
            ],
        )
    )

    out: dict[str, Any] = {}

    # RAM
    ram_text = _spec_lookup(specs, "ram", "memory size", "installed ram")
    ram_match = _match_spec_first(
        (r"(\d+(?:\.\d+)?)\s*GB\s*(?:of\s*)?RAM", r"RAM[:\s]*(\d+(?:\.\d+)?)\s*GB"),
        ram_text,
        haystack,
    )
    if ram_match:
        out["ram_gb"] = float(ram_match.group(1))
    elif ram_text:
        value = parse_number(ram_text)
        if value and value <= 64:
            out["ram_gb"] = value

    # Storage
    storage_text = _spec_lookup(
        specs, "storage capacity", "memory storage", "rom", "hard drive", "flash memory"
    )
    storage_patterns = (
        r"(\d+(?:\.\d+)?)\s*(TB|GB)\s*(?:ROM|internal|storage)",
        r"(?:ROM|internal|storage)(?:\s+capacity)?[:\s]*(\d+(?:\.\d+)?)\s*(TB|GB)",
    )
    storage_match = _match_spec_first(storage_patterns, storage_text, haystack)
    if storage_match:
        value = float(storage_match.group(1))
        out["storage_gb"] = value * 1024 if storage_match.group(2).upper() == "TB" else value
    elif storage_text:
        value = parse_number(storage_text)
        if value:
            out["storage_gb"] = value * 1024 if "tb" in storage_text.lower() else value

    # Titles almost always state the storage tier, so use them to fill a gap or to
    # correct an implausible reading (storage below RAM).
    storage_value = out.get("storage_gb")
    ram_value = out.get("ram_gb")
    if storage_value is None or (ram_value is not None and storage_value <= ram_value):
        candidates = [
            float(value) for value in re.findall(r"(\d+(?:\.\d+)?)\s*GB", title or "", re.IGNORECASE)
        ] + [
            float(value) * 1024
            for value in re.findall(r"(\d+(?:\.\d+)?)\s*TB", title or "", re.IGNORECASE)
        ]
        plausible = [value for value in candidates if value >= (ram_value or 0)]
        if plausible:
            out["storage_gb"] = max(plausible)

    # Display size
    display_text = _spec_lookup(specs, "screen size", "display size", "standing screen")
    display_match = _match_spec_first(
        (r"(\d\.\d{1,2})\s*(?:-)?\s*(?:inch|inches|\"|''|in\b)",), display_text, haystack
    )
    if display_match is None and display_text:
        # A dedicated screen-size field may hold just the number ("6.8").
        display_match = re.search(r"(\d\.\d{1,2})", display_text)
    if display_match:
        value = float(display_match.group(1))
        if 3.0 <= value <= 9.0:
            out["display_inches"] = value

    # Battery
    battery_match = re.search(r"(\d{3,5})\s*mAh", haystack, re.IGNORECASE)
    if battery_match:
        value = int(battery_match.group(1))
        if 500 <= value <= 15_000:
            out["battery_mah"] = value

    # Rear camera megapixels: take the largest plausible figure.
    mp_values = [
        float(m) for m in re.findall(r"(\d{1,3}(?:\.\d)?)\s*MP", haystack, re.IGNORECASE)
    ]
    plausible = [v for v in mp_values if 0.3 <= v <= 250]
    if plausible:
        out["rear_camera_mp"] = max(plausible)

    # Refresh rate
    hz_values = [int(m) for m in re.findall(r"(\d{2,3})\s*Hz", haystack, re.IGNORECASE)]
    plausible_hz = [v for v in hz_values if 30 <= v <= 240]
    if plausible_hz:
        out["refresh_rate_hz"] = max(plausible_hz)

    # Chipset
    chipset_text = _spec_lookup(specs, "chipset", "processor", "cpu model", "soc")
    chipset_match = _match_spec_first(
        (
            r"(Snapdragon\s+[\w\s+]{2,25})",
            r"(Exynos\s+[\w\s]{2,20})",
            r"(Dimensity\s+[\w\s]{2,20})",
            r"(A\d{2}\s+(?:Pro\s+)?Bionic)",
            r"(Tensor\s+G?\d?)",
            r"(Helio\s+[\w\d]{2,10})",
            r"(Unisoc\s+[\w\d]{2,12})",
            r"(Kirin\s+[\w\d]{2,10})",
        ),
        chipset_text,
        haystack,
    )
    if chipset_match:
        out["chipset"] = re.sub(r"\s+", " ", chipset_match.group(1)).strip()[:128]
    elif chipset_text:
        out["chipset"] = chipset_text[:128]

    # OS
    os_text = _spec_lookup(specs, "operating system", "os")
    os_match = _match_spec_first(
        (
            r"(Android\s*\d{1,2}(?:\.\d)?)",
            r"(iOS\s*\d{1,2}(?:\.\d)?)",
            r"\b(Android|iOS|HarmonyOS)\b",
        ),
        os_text,
        haystack,
    )
    if os_match:
        out["operating_system"] = os_match.group(1).strip()[:64]

    # Release year, from the "date first available" field when present.
    date_text = _spec_lookup(specs, "date first available", "release date", "first available")
    if date_text:
        year_match = re.search(r"(19|20)\d{2}", date_text)
        if year_match:
            year = int(year_match.group(0))
            if 2005 <= year <= datetime.now().year + 1:
                out["release_year"] = year

    return out
