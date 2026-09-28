"""Steps 3 & 4 - Aspect extraction and sentiment classification.

Two interchangeable engines implement the same interface:

* ``LLMAbsaEngine``     - the methodology's LLM annotator (one batched call per
                          group of sentences, strict JSON output).
* ``LexiconAbsaEngine`` - a rule-based baseline that needs no API key. It runs
                          offline and doubles as the comparison baseline the
                          LLM results can be evaluated against.

Both return, for each input sentence, a list of ``AspectOpinion`` records.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Protocol

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.nlp.aspects import (
    CONTRAST_MARKERS,
    DIMINISHERS,
    INTENSIFIERS,
    NEGATION_TOKENS,
    NEGATIVE_LEXICON,
    POSITIVE_LEXICON,
    aspects_for,
    detect_aspects,
    normalise_aspect,
    normalise_sentiment,
)
from app.nlp.prompts import build_absa_messages

logger = get_logger(__name__)


@dataclass(slots=True)
class AspectOpinion:
    aspect: str
    sentiment: str
    confidence: float
    opinion_term: str | None = None
    method: str = "lexicon"
    model_name: str | None = None


class AbsaEngine(Protocol):
    name: str
    model_name: str | None

    def analyze(self, sentences: list[str]) -> list[list[AspectOpinion]]:
        """Annotate sentences; result[i] corresponds to sentences[i]."""


# --------------------------------------------------------------------------- #
# Rule-based baseline
# --------------------------------------------------------------------------- #
_TOKEN_PATTERN = re.compile(r"[a-z0-9']+")

_MULTIWORD_POS = tuple(sorted((p for p in POSITIVE_LEXICON if " " in p), key=len, reverse=True))
_MULTIWORD_NEG = tuple(sorted((p for p in NEGATIVE_LEXICON if " " in p), key=len, reverse=True))

_MULTIWORD_PATTERN = re.compile(
    "(?<!\\w)(?:" + "|".join(re.escape(p) for p in (*_MULTIWORD_NEG, *_MULTIWORD_POS)) + ")(?!\\w)",
    re.IGNORECASE,
)

_CONTRAST_SPLIT = re.compile(
    "|".join(re.escape(marker) for marker in CONTRAST_MARKERS), re.IGNORECASE
)

POSITIVE_THRESHOLD = 0.18
NEGATIVE_THRESHOLD = -0.18


class LexiconAbsaEngine:
    """Aspect keyword matching + negation-aware sentiment lexicon scoring."""

    name = "lexicon"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.allowed = aspects_for(self.settings.aspect_set)
        self.model_name = "domain-lexicon-v1"

    def analyze(self, sentences: list[str]) -> list[list[AspectOpinion]]:
        return [self._analyze_one(sentence) for sentence in sentences]

    def _analyze_one(self, sentence: str) -> list[AspectOpinion]:
        if not sentence or not sentence.strip():
            return []

        clauses = [c.strip() for c in _CONTRAST_SPLIT.split(f" {sentence} ") if c and c.strip()]
        if not clauses:
            clauses = [sentence]

        # Aspects mentioned anywhere in the sentence, used to resolve pronoun
        # clauses ("Battery is great. It never dies." -> both about battery).
        sentence_aspects = detect_aspects(sentence, self.allowed)

        found: dict[str, AspectOpinion] = {}

        for clause in clauses:
            clause_aspects = detect_aspects(clause, self.allowed)
            targets = clause_aspects or (
                sentence_aspects if len(sentence_aspects) == 1 else {}
            )
            if not targets:
                continue

            clause_polarity, clause_term = self._score_span(clause)

            for aspect, surface_forms in targets.items():
                polarity, term = clause_polarity, clause_term
                if len(targets) > 1:
                    # Several aspects share a clause: score a window around each
                    # keyword so "camera is sharp, screen is dim" splits correctly.
                    window_polarity, window_term = self._score_window(
                        clause, surface_forms[0]
                    )
                    if window_term is not None:
                        polarity, term = window_polarity, window_term

                sentiment = self._to_label(polarity)
                confidence = self._to_confidence(polarity, sentiment)

                previous = found.get(aspect)
                if previous is None or confidence > previous.confidence:
                    found[aspect] = AspectOpinion(
                        aspect=aspect,
                        sentiment=sentiment,
                        confidence=confidence,
                        opinion_term=term,
                        method=self.name,
                        model_name=self.model_name,
                    )

        return list(found.values())

    def _score_window(self, clause: str, surface: str, radius: int = 7) -> tuple[float, str | None]:
        tokens = _TOKEN_PATTERN.findall(clause.lower())
        surface_head = _TOKEN_PATTERN.findall(surface.lower())
        if not tokens or not surface_head:
            return 0.0, None
        try:
            centre = tokens.index(surface_head[0])
        except ValueError:
            return self._score_span(clause)
        start = max(0, centre - radius)
        window = " ".join(tokens[start : centre + radius + 1])
        return self._score_span(window)

    def _score_span(self, span: str) -> tuple[float, str | None]:
        """Signed polarity in roughly [-1, 1] plus the strongest opinion term."""
        lowered = span.lower()
        hits: list[tuple[float, str]] = []

        # Multi-word entries first; mask them so their parts are not double counted.
        masked = lowered
        for match in _MULTIWORD_PATTERN.finditer(lowered):
            phrase = match.group(0)
            weight = POSITIVE_LEXICON.get(phrase)
            if weight is not None:
                polarity = weight
            else:
                polarity = -NEGATIVE_LEXICON.get(phrase, 0.0)
            if polarity:
                prefix = lowered[max(0, match.start() - 40) : match.start()]
                polarity *= self._context_multiplier(prefix)
                hits.append((polarity, phrase))
            masked = masked.replace(phrase, " " * len(phrase), 1)

        tokens = _TOKEN_PATTERN.findall(masked)
        for index, token in enumerate(tokens):
            weight = POSITIVE_LEXICON.get(token)
            polarity = weight if weight is not None else -NEGATIVE_LEXICON.get(token, 0.0)
            if not polarity:
                continue
            context = " ".join(tokens[max(0, index - 3) : index])
            polarity *= self._context_multiplier(context)
            hits.append((polarity, token))

        if not hits:
            return 0.0, None

        total = sum(polarity for polarity, _ in hits)
        # Average softened by count: many weak cues should not outrank one strong cue.
        score = total / (len(hits) ** 0.5)
        strongest = max(hits, key=lambda item: abs(item[0]))[1]
        return max(-1.0, min(1.0, score)), strongest

    @staticmethod
    def _context_multiplier(context: str) -> float:
        """Negation flip plus intensifier/diminisher scaling from preceding words."""
        multiplier = 1.0
        context_tokens = _TOKEN_PATTERN.findall(context)

        if any(token in NEGATION_TOKENS for token in context_tokens[-3:]):
            # A negated positive ("not great") is milder than an outright negative.
            multiplier *= -0.8

        for token in context_tokens[-2:]:
            if token in INTENSIFIERS:
                multiplier *= INTENSIFIERS[token]
            elif token in DIMINISHERS:
                multiplier *= DIMINISHERS[token]

        for phrase, factor in DIMINISHERS.items():
            if " " in phrase and phrase in context:
                multiplier *= factor

        return multiplier

    @staticmethod
    def _to_label(polarity: float) -> str:
        if polarity >= POSITIVE_THRESHOLD:
            return "positive"
        if polarity <= NEGATIVE_THRESHOLD:
            return "negative"
        return "neutral"

    @staticmethod
    def _to_confidence(polarity: float, sentiment: str) -> float:
        if sentiment == "neutral":
            return round(0.35 + 0.2 * (1 - min(abs(polarity) / POSITIVE_THRESHOLD, 1.0)), 3)
        return round(min(0.95, 0.5 + abs(polarity) * 0.45), 3)


# --------------------------------------------------------------------------- #
# LLM engine
# --------------------------------------------------------------------------- #
class LLMError(RuntimeError):
    pass


class LLMAbsaEngine:
    """Batched ABSA against any OpenAI-compatible chat completions endpoint."""

    name = "llm"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        if not self.settings.llm_available():
            raise LLMError(
                "LLM_API_KEY is not set. Set it in .env, or use ABSA_ENGINE=lexicon."
            )
        self.allowed = aspects_for(self.settings.aspect_set)
        self.model_name = self.settings.llm_model
        self.batch_size = max(1, self.settings.llm_batch_size)
        self._fallback = LexiconAbsaEngine(self.settings)
        self._supports_json_mode = True

        import httpx

        self._client = httpx.Client(
            base_url=self.settings.llm_base_url.rstrip("/"),
            timeout=self.settings.llm_timeout_s,
            headers={
                "Authorization": f"Bearer {self.settings.llm_api_key}",
                "Content-Type": "application/json",
            },
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "LLMAbsaEngine":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def analyze(self, sentences: list[str]) -> list[list[AspectOpinion]]:
        results: list[list[AspectOpinion]] = []
        for start in range(0, len(sentences), self.batch_size):
            batch = sentences[start : start + self.batch_size]
            results.extend(self._analyze_batch(batch))
        return results

    def _analyze_batch(self, batch: list[str]) -> list[list[AspectOpinion]]:
        messages = build_absa_messages(batch, self.allowed)
        last_error: Exception | None = None

        for attempt in range(1, self.settings.llm_max_retries + 1):
            try:
                content = self._chat(messages)
                parsed = self._parse_response(content, len(batch))
                if parsed is not None:
                    return parsed
                last_error = LLMError("Response JSON did not match the expected schema.")
            except Exception as exc:
                last_error = exc

            backoff = min(2 ** attempt, 20)
            logger.warning(
                "ABSA batch attempt %s/%s failed (%s); retrying in %ss.",
                attempt,
                self.settings.llm_max_retries,
                last_error,
                backoff,
            )
            time.sleep(backoff)

        logger.error(
            "LLM annotation failed for a batch of %s sentences (%s). "
            "Falling back to the lexicon engine for this batch.",
            len(batch),
            last_error,
        )
        fallback = self._fallback.analyze(batch)
        for opinions in fallback:
            for opinion in opinions:
                opinion.method = "lexicon_fallback"
        return fallback

    def _chat(self, messages: list[dict[str, str]]) -> str:
        payload: dict[str, object] = {
            "model": self.settings.llm_model,
            "messages": messages,
            "temperature": self.settings.llm_temperature,
        }
        if self._supports_json_mode:
            payload["response_format"] = {"type": "json_object"}

        response = self._client.post("/chat/completions", json=payload)

        if response.status_code == 400 and self._supports_json_mode:
            # Provider/model without JSON mode: disable and retry once.
            logger.info("Endpoint rejected response_format; disabling JSON mode.")
            self._supports_json_mode = False
            payload.pop("response_format", None)
            response = self._client.post("/chat/completions", json=payload)

        if response.status_code == 429:
            raise LLMError("Rate limited by the LLM provider (HTTP 429).")
        if response.status_code >= 400:
            raise LLMError(f"HTTP {response.status_code}: {response.text[:400]}")

        data = response.json()
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected response shape: {str(data)[:400]}") from exc

    def _parse_response(self, content: str, expected: int) -> list[list[AspectOpinion]] | None:
        data = _extract_json(content)
        if not isinstance(data, dict):
            return None

        raw_results = data.get("results")
        if not isinstance(raw_results, list):
            return None

        by_id: dict[int, list[AspectOpinion]] = {}
        for entry in raw_results:
            if not isinstance(entry, dict):
                continue
            try:
                sentence_id = int(entry.get("id"))
            except (TypeError, ValueError):
                continue

            opinions: list[AspectOpinion] = []
            seen: set[str] = set()
            for item in entry.get("aspects") or []:
                if not isinstance(item, dict):
                    continue
                aspect = normalise_aspect(str(item.get("aspect", "")), self.allowed)
                sentiment = normalise_sentiment(str(item.get("sentiment", "")))
                if not aspect or not sentiment or aspect in seen:
                    continue
                seen.add(aspect)

                try:
                    confidence = float(item.get("confidence", 0.8))
                except (TypeError, ValueError):
                    confidence = 0.8
                confidence = max(0.0, min(1.0, confidence))

                opinion_term = str(item.get("opinion") or "").strip()[:200] or None

                opinions.append(
                    AspectOpinion(
                        aspect=aspect,
                        sentiment=sentiment,
                        confidence=confidence,
                        opinion_term=opinion_term,
                        method=self.name,
                        model_name=self.model_name,
                    )
                )
            by_id[sentence_id] = opinions

        # Ids are 1-based and must cover the whole batch.
        if not by_id:
            return None
        return [by_id.get(index + 1, []) for index in range(expected)]


_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _extract_json(content: str) -> object | None:
    """Parse JSON from a model reply, tolerating fences and surrounding prose."""
    if not content:
        return None

    candidates: list[str] = []
    fenced = _JSON_FENCE.search(content)
    if fenced:
        candidates.append(fenced.group(1))
    candidates.append(content)

    start, end = content.find("{"), content.rfind("}")
    if 0 <= start < end:
        candidates.append(content[start : end + 1])

    for candidate in candidates:
        text = candidate.strip()
        if not text:
            continue
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Trailing commas are the most common malformation.
            repaired = re.sub(r",\s*([}\]])", r"\1", text)
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                continue
    return None


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def get_absa_engine(
    engine: str | None = None, settings: Settings | None = None
) -> AbsaEngine:
    """Build the requested engine, degrading to the lexicon baseline if needed."""
    settings = settings or get_settings()
    requested = engine or settings.absa_engine

    if requested == "auto":
        requested = settings.resolved_absa_engine()

    if requested == "llm":
        try:
            return LLMAbsaEngine(settings)
        except Exception as exc:
            logger.warning("Cannot use the LLM engine (%s); using lexicon baseline.", exc)
            return LexiconAbsaEngine(settings)

    return LexiconAbsaEngine(settings)
