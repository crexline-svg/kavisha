"""Step 1 - Data preprocessing.

Cleaning, normalisation, spam/quality filtering, language filtering and
deterministic de-duplication hashing. Reviews are never silently dropped: each
rejected review is flagged with a reason so the dataset section of the
dissertation can report exact exclusion counts.
"""

from __future__ import annotations

import hashlib
import html
import re
import unicodedata
from dataclasses import dataclass, field

from app.core.logging import get_logger

logger = get_logger(__name__)

# Marketplace UI text that leaks into scraped review bodies.
BOILERPLATE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bread more\b",
        r"\bread less\b",
        r"\bshow more\b",
        r"the media could not be loaded\.?",
        r"\d+ (?:people|person) found this helpful",
        r"helpful\s*\|\s*report abuse",
        r"\breport abuse\b",
        r"\bverified purchase\b",
        r"reviewed in .{0,40} on \w+ \d{1,2}, \d{4}",
        r"colou?r:\s*[^|\n]{1,40}\|",
        r"\bsize:\s*[^|\n]{1,40}\|",
        r"see more reviews",
        r"translated from \w+",
        r"originally posted in \w+",
    )
)

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
EMAIL_PATTERN = re.compile(r"\S+@\S+\.\S+")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
REPEATED_CHAR_PATTERN = re.compile(r"(.)\1{3,}")
REPEATED_PUNCT_PATTERN = re.compile(r"([!?.,])\1{1,}")
WHITESPACE_PATTERN = re.compile(r"\s+")
NON_PRINTABLE_PATTERN = re.compile(r"[\u200b-\u200f\u2028\u2029\ufeff\u00ad]")

# Zero-information reviews that add noise to aspect statistics.
GENERIC_ONLY = frozenset(
    {
        "good", "nice", "ok", "okay", "bad", "great", "awesome", "super",
        "excellent", "worst", "best", "fine", "average", "yes", "no", "n/a",
        "na", "none", "nothing", "cool", "wow", "thanks", "thank you", "good product",
        "nice product", "very good", "very nice", "good one", "best product",
        "value for money", "as expected", "not bad", "love it", "worth it",
    }
)


@dataclass(slots=True)
class CleanedReview:
    """Outcome of preprocessing a single review."""

    cleaned_text: str
    content_hash: str
    language: str | None = None
    is_spam: bool = False
    excluded_reason: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return not self.is_spam and self.excluded_reason is None and bool(self.cleaned_text)


def strip_html(text: str) -> str:
    text = HTML_TAG_PATTERN.sub(" ", text)
    return html.unescape(text)


def normalise_text(text: str) -> str:
    """Unicode-normalise, remove boilerplate/URLs and collapse whitespace."""
    if not text:
        return ""

    text = strip_html(text)
    # NFKC folds full-width characters and odd typographic variants.
    text = unicodedata.normalize("NFKC", text)
    text = NON_PRINTABLE_PATTERN.sub("", text)

    text = URL_PATTERN.sub(" ", text)
    text = EMAIL_PATTERN.sub(" ", text)

    for pattern in BOILERPLATE_PATTERNS:
        text = pattern.sub(" ", text)

    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-").replace("\u2026", "...")

    # "sooooo goooood" -> "soo good" (keeps emphasis, kills token explosion)
    text = REPEATED_CHAR_PATTERN.sub(r"\1\1", text)
    text = REPEATED_PUNCT_PATTERN.sub(r"\1", text)

    text = WHITESPACE_PATTERN.sub(" ", text).strip()
    return text


def compute_content_hash(text: str) -> str:
    """Hash of an aggressively normalised form, so near-identical text collides."""
    canonical = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    canonical = WHITESPACE_PATTERN.sub(" ", canonical)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def detect_language(text: str) -> str | None:
    """Best-effort language code. Falls back to an English-stopword heuristic."""
    sample = text.strip()
    if len(sample) < 20:
        return _heuristic_language(sample)
    try:
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 0
        return detect(sample)
    except Exception:
        return _heuristic_language(sample)


def _heuristic_language(text: str) -> str | None:
    if not text:
        return None
    lowered = text.lower()
    ascii_letters = sum(1 for ch in lowered if "a" <= ch <= "z")
    if not ascii_letters:
        return None
    if ascii_letters / max(len(lowered), 1) < 0.4:
        return None
    common = {
        "the", "and", "is", "it", "this", "for", "with", "but", "very", "good",
        "phone", "battery", "camera", "not", "was", "have", "great", "you",
    }
    tokens = set(re.findall(r"[a-z']+", lowered))
    return "en" if len(tokens & common) >= 2 else None


def _looks_like_spam(text: str, raw: str | None = None) -> tuple[bool, str | None]:
    """Quality gate. `raw` is the pre-normalisation text.

    Some signals only survive in the raw form: normalisation collapses
    "AAAAAAAA..." to "AA" and lowercases nothing, so keyboard mashing and
    shouting must be judged before that happens.
    """
    lowered = text.lower().strip(" .!?")

    if lowered in GENERIC_ONLY:
        return True, "generic_only"

    if raw:
        condensed = re.sub(r"\s+", "", raw)
        # Keyboard mashing: long text built from one or two distinct characters.
        if len(condensed) >= 12 and len(set(condensed.lower())) <= 2:
            return True, "repeated_chars"

        raw_letters = sum(1 for ch in raw if ch.isalpha())
        raw_upper = sum(1 for ch in raw if ch.isupper())
        if raw_letters > 25 and raw_upper / max(raw_letters, 1) > 0.8:
            return True, "all_caps"

    tokens = re.findall(r"[a-z']+", lowered)
    if not tokens:
        return True, "no_alpha_content"

    # A single token repeated over and over.
    if len(tokens) >= 6 and len(set(tokens)) / len(tokens) < 0.3:
        return True, "low_lexical_diversity"

    letters = sum(1 for ch in text if ch.isalpha())
    if letters / max(len(text), 1) < 0.45:
        return True, "low_alpha_ratio"

    if len(URL_PATTERN.findall(text)) >= 2:
        return True, "link_spam"

    return False, None


def preprocess_review(
    raw_body: str,
    *,
    title: str | None = None,
    min_chars: int = 15,
    language_filter: str | None = "en",
) -> CleanedReview:
    """Clean one review and decide whether it enters the analysis corpus."""
    notes: list[str] = []

    body = normalise_text(raw_body or "")
    clean_title = normalise_text(title or "")

    # The title often carries the sharpest opinion; prepend it when it is not
    # already the opening of the body.
    if clean_title and not body.lower().startswith(clean_title.lower()[:30]):
        combined = f"{clean_title.rstrip('.!?')}. {body}".strip()
        notes.append("title_prepended")
    else:
        combined = body

    content_hash = compute_content_hash(combined)

    if not combined:
        return CleanedReview("", content_hash, excluded_reason="empty", notes=notes)

    # Quality is judged before length: "good product" is more usefully reported as
    # a content-free review than as a short one.
    raw_combined = f"{title or ''} {raw_body or ''}".strip()
    is_spam, spam_reason = _looks_like_spam(combined, raw_combined)
    if is_spam:
        return CleanedReview(
            combined, content_hash, is_spam=True, excluded_reason=spam_reason, notes=notes
        )

    if len(combined) < min_chars:
        return CleanedReview(
            combined, content_hash, excluded_reason="too_short", notes=notes
        )

    language = detect_language(combined)
    if language_filter and language != language_filter:
        return CleanedReview(
            combined,
            content_hash,
            language=language,
            excluded_reason="language_filtered",
            notes=notes,
        )

    return CleanedReview(combined, content_hash, language=language, notes=notes)
