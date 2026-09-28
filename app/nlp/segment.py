"""Step 2 - Sentence segmentation.

Reviews are split into sentences because a single review usually mixes opinions
about several aspects ("Battery lasts two days. Camera is poor."). Sentence-level
input is what makes the aspect/sentiment pairs in Steps 3-4 unambiguous.

`pysbd` is used when available (rule-based, no model download, handles
abbreviations and decimals such as "6.7 inch"); a regex splitter is the fallback.
"""

from __future__ import annotations

import re

from app.core.logging import get_logger

logger = get_logger(__name__)

_SEGMENTER = None
_SEGMENTER_READY = False

# Fallback splitter: break on . ! ? … or newlines, but not inside decimals
# ("6.7"), abbreviations ("e.g.") or version numbers ("Android 14.0").
_FALLBACK_SPLIT = re.compile(
    r"""
    (?<=[.!?\u2026])      # after a terminator
    [\)\"'\u201d]*        # optional closing quote/bracket
    \s+                   # whitespace
    (?=[A-Z0-9\u201c"'(])  # next sentence starts here
    |
    \n+
    """,
    re.VERBOSE,
)

_ABBREVIATIONS = (
    "e.g.", "i.e.", "etc.", "vs.", "mr.", "mrs.", "ms.", "dr.", "no.",
    "approx.", "min.", "max.", "hrs.", "hr.", "sec.", "fig.",
)

# Clause markers used to split run-on sentences that carry opposing opinions.
_CLAUSE_SPLIT = re.compile(
    r"\s*(?:,\s*)?\b(?:but|however|although|though|whereas|yet)\b\s*", re.IGNORECASE
)


def _get_segmenter():
    global _SEGMENTER, _SEGMENTER_READY
    if _SEGMENTER_READY:
        return _SEGMENTER
    _SEGMENTER_READY = True
    try:
        import pysbd

        _SEGMENTER = pysbd.Segmenter(language="en", clean=False)
        logger.debug("Using pysbd for sentence segmentation.")
    except Exception as exc:
        logger.warning("pysbd unavailable (%s); falling back to regex segmentation.", exc)
        _SEGMENTER = None
    return _SEGMENTER


def _protect_abbreviations(text: str) -> str:
    for abbr in _ABBREVIATIONS:
        text = re.sub(
            rf"(?<!\w){re.escape(abbr)}", abbr.replace(".", "\u2024"), text, flags=re.IGNORECASE
        )
    # Decimal points: 6.7 -> 6․7
    text = re.sub(r"(?<=\d)\.(?=\d)", "\u2024", text)
    return text


def _restore_abbreviations(text: str) -> str:
    return text.replace("\u2024", ".")


def _regex_segment(text: str) -> list[str]:
    protected = _protect_abbreviations(text)
    parts = _FALLBACK_SPLIT.split(protected)
    return [_restore_abbreviations(p) for p in parts if p and p.strip()]


def _split_long_clauses(sentence: str, max_chars: int) -> list[str]:
    """Split an over-long sentence on contrast markers so opposing opinions separate."""
    if len(sentence) <= max_chars:
        return [sentence]

    pieces = [p.strip() for p in _CLAUSE_SPLIT.split(sentence) if p and p.strip()]
    if len(pieces) <= 1:
        # No contrast marker: fall back to comma chunks.
        pieces = [p.strip() for p in sentence.split(",") if p.strip()]

    merged: list[str] = []
    buffer = ""
    for piece in pieces:
        candidate = f"{buffer}, {piece}" if buffer else piece
        if len(candidate) <= max_chars:
            buffer = candidate
        else:
            if buffer:
                merged.append(buffer)
            buffer = piece
    if buffer:
        merged.append(buffer)
    return merged or [sentence[:max_chars]]


def segment_sentences(
    text: str,
    *,
    min_chars: int = 3,
    max_chars: int = 320,
    merge_short: bool = True,
) -> list[str]:
    """Split cleaned review text into analysis-ready sentences."""
    if not text or not text.strip():
        return []

    segmenter = _get_segmenter()
    if segmenter is not None:
        try:
            raw = [str(s) for s in segmenter.segment(text)]
        except Exception as exc:
            logger.debug("pysbd failed on a review (%s); using regex fallback.", exc)
            raw = _regex_segment(text)
    else:
        raw = _regex_segment(text)

    candidates = [s.strip() for s in raw if s and s.strip()]

    expanded: list[str] = []
    for sentence in candidates:
        expanded.extend(_split_long_clauses(sentence, max_chars))

    if not merge_short:
        return [s for s in expanded if len(s) >= min_chars]

    # Attach stray fragments ("Nice.", "5 stars") to a neighbour so they do not
    # become aspect-less sentences. A fragment merges backwards when possible and
    # forwards otherwise, which covers the common "Nice. <real sentence>" opening.
    result: list[str] = []
    carry = ""

    def _is_fragment(text: str) -> bool:
        return len(text) < 15 or len(text.split()) < 3

    for sentence in expanded:
        if len(sentence) < min_chars:
            continue

        if carry:
            sentence = f"{carry} {sentence}".strip()
            carry = ""

        if _is_fragment(sentence):
            if result and len(result[-1]) + len(sentence) + 1 <= max_chars:
                result[-1] = f"{result[-1]} {sentence}".strip()
            else:
                carry = sentence
        else:
            result.append(sentence)

    if carry:
        if result and len(result[-1]) + len(carry) + 1 <= max_chars:
            result[-1] = f"{result[-1]} {carry}".strip()
        else:
            result.append(carry)

    return [s for s in result if len(s) >= min_chars]


def count_words(text: str) -> int:
    return len(re.findall(r"\b[\w']+\b", text))
