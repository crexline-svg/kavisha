"""Prompt construction for LLM-based aspect extraction + sentiment classification.

Steps 3 and 4 of the methodology are issued as a *single* call per sentence
batch. Doing both in one pass is faster, cheaper and removes the inconsistency
that appears when a second call re-reads an aspect it did not extract itself.
"""

from __future__ import annotations

import json

from app.nlp.aspects import ASPECT_DEFINITIONS

SYSTEM_PROMPT = (
    "You are a meticulous annotator performing aspect-based sentiment analysis "
    "(ABSA) on smartphone product reviews. You follow the label taxonomy exactly "
    "and you only ever reply with valid JSON that matches the requested schema. "
    "You never add commentary, explanation or markdown fences."
)

_RULES = """\
TASK
For every input sentence, extract the smartphone aspects it expresses an opinion about,
and classify the sentiment towards each one.

ALLOWED ASPECTS (use these exact lowercase labels, nothing else):
{aspect_block}

ALLOWED SENTIMENTS: "positive", "negative", "neutral"

RULES
1. Only output an aspect when the sentence actually says something evaluative or
   factual about it. Do not guess aspects that are merely plausible for a phone.
2. One sentence may yield several aspects. Output one object per (aspect, sentiment) pair.
3. If a sentence expresses no opinion about any allowed aspect (greetings, delivery
   complaints, seller issues, packaging, unrelated chatter), return an empty "aspects" list.
4. Implicit mentions count. "It dies by lunchtime" -> battery/negative.
   "Photos come out grainy at night" -> camera/negative.
5. Sentiment is the reviewer's evaluation of the aspect, not the overall tone of the
   sentence. In "camera is great but it overheats", camera=positive AND performance=negative.
6. Use "neutral" for purely factual or genuinely mixed statements with no clear valence
   ("it has a 5000 mAh battery", "battery is okay, nothing special").
7. Handle negation carefully: "not worth the price" -> price/negative.
   "battery does not drain" -> battery/positive.
8. "price" means value for money / whether the cost is justified, NOT the numeric price.
9. "opinion" must be a short verbatim span copied from the sentence (max 8 words) that
   carries the sentiment. Use "" when the sentiment is implicit with no clear span.
10. "confidence" is your certainty in the (aspect, sentiment) pair, a number in [0, 1].

OUTPUT SCHEMA (return exactly this shape, one entry per input sentence, same ids):
{{
  "results": [
    {{
      "id": 1,
      "aspects": [
        {{"aspect": "battery", "sentiment": "positive", "opinion": "lasts two days", "confidence": 0.96}}
      ]
    }}
  ]
}}

EXAMPLE
Input:
[
  {{"id": 1, "text": "Battery lasts two days and the display is amazing, but the camera performs poorly in low light."}},
  {{"id": 2, "text": "Delivery was late and the box was damaged."}},
  {{"id": 3, "text": "It has a 5000 mAh cell."}}
]
Output:
{{"results": [
  {{"id": 1, "aspects": [
    {{"aspect": "battery", "sentiment": "positive", "opinion": "lasts two days", "confidence": 0.97}},
    {{"aspect": "display", "sentiment": "positive", "opinion": "is amazing", "confidence": 0.95}},
    {{"aspect": "camera", "sentiment": "negative", "opinion": "performs poorly in low light", "confidence": 0.94}}
  ]}},
  {{"id": 2, "aspects": []}},
  {{"id": 3, "aspects": [
    {{"aspect": "battery", "sentiment": "neutral", "opinion": "5000 mAh cell", "confidence": 0.8}}
  ]}}
]}}\
"""


def build_absa_messages(
    sentences: list[str], allowed_aspects: tuple[str, ...]
) -> list[dict[str, str]]:
    """Build the chat messages for one batch of sentences."""
    aspect_block = "\n".join(
        f'  - "{aspect}": {ASPECT_DEFINITIONS.get(aspect, aspect)}' for aspect in allowed_aspects
    )
    payload = [{"id": index + 1, "text": text} for index, text in enumerate(sentences)]

    user_prompt = (
        _RULES.format(aspect_block=aspect_block)
        + "\n\nNow annotate these sentences.\nInput:\n"
        + json.dumps(payload, ensure_ascii=False, indent=1)
        + "\nOutput:"
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
