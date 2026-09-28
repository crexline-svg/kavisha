"""Aspect taxonomy and the domain sentiment lexicon.

The taxonomy is closed on purpose: a fixed label set keeps the LLM output
comparable across runs and makes the aggregated score table well defined.

`CORE_ASPECTS` matches the six features in the methodology
(battery, camera, display, performance, design, price). `EXTENDED_ASPECTS`
adds four more for studies that need finer granularity (ASPECT_SET=extended).
"""

from __future__ import annotations

import re
from functools import lru_cache

CORE_ASPECTS: tuple[str, ...] = (
    "battery",
    "camera",
    "display",
    "performance",
    "design",
    "price",
)

EXTRA_ASPECTS: tuple[str, ...] = ("software", "connectivity", "audio", "durability")

EXTENDED_ASPECTS: tuple[str, ...] = CORE_ASPECTS + EXTRA_ASPECTS

SENTIMENTS: tuple[str, ...] = ("positive", "negative", "neutral")

ASPECT_LABELS: dict[str, str] = {
    "battery": "Battery",
    "camera": "Camera",
    "display": "Display",
    "performance": "Performance",
    "price": "Price / Value for money",
    "design": "Design & build",
    "software": "Software & updates",
    "connectivity": "Connectivity & network",
    "audio": "Audio & speakers",
    "durability": "Durability & reliability",
}

ASPECT_DEFINITIONS: dict[str, str] = {
    "battery": "battery life, endurance, screen-on time, charging speed, power drain",
    "camera": "photo and video quality, lenses, zoom, low-light shots, selfies, stabilisation",
    "display": "screen quality, brightness, colours, resolution, refresh rate, touch response",
    "performance": "speed, responsiveness, processor, RAM, multitasking, gaming, lag, heating",
    "price": "price, cost, value for money, whether the phone is worth what it costs",
    "design": "look and feel, materials, size, weight, ergonomics, colours of the body",
    "software": "operating system, UI, updates, bloatware, bugs, features",
    "connectivity": "network reception, signal, 5G, Wi-Fi, Bluetooth, GPS, call quality",
    "audio": "loudspeaker, earpiece, sound quality, headphone output",
    "durability": "build robustness, scratches, drops, water resistance, long-term reliability",
}

# Surface forms used by the lexicon baseline engine to detect which aspect a
# sentence talks about. Longer phrases are matched first.
ASPECT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "battery": (
        "battery life", "battery back up", "battery backup", "battery health", "battery",
        "charge", "charges", "charging", "charger", "fast charging", "recharge",
        "power bank", "screen on time", "screen-on time", "mah", "standby", "endurance",
        "drain", "drains", "draining", "juice", "last all day", "lasts all day",
        # Implicit battery-death phrasing ("it dies by lunchtime").
        "dies", "die", "died", "dying", "runs out", "ran out", "power off",
    ),
    "camera": (
        "camera", "cameras", "photo", "photos", "photography", "picture", "pictures",
        "pic", "pics", "image quality", "video", "videos", "recording", "selfie",
        "selfies", "lens", "lenses", "zoom", "megapixel", "mp camera", "portrait mode",
        "night mode", "low light", "low-light", "shutter", "stabilisation",
        "stabilization", "ois", "front camera", "rear camera", "ultrawide", "telephoto",
    ),
    "display": (
        "display", "screen", "amoled", "oled", "lcd", "resolution", "brightness",
        "bright", "colours", "colors", "color accuracy", "refresh rate", "hz",
        "touch response", "touchscreen", "viewing angle", "bezel", "bezels", "notch",
        "sunlight legibility", "panel", "hdr", "contrast",
    ),
    "performance": (
        "performance", "speed", "fast", "slow", "lag", "lags", "lagging", "laggy",
        "stutter", "stutters", "smooth", "snappy", "processor", "cpu", "gpu",
        "chipset", "snapdragon", "exynos", "dimensity", "bionic", "tensor", "ram",
        "multitask", "multitasking", "gaming", "games", "fps", "frame rate", "heat",
        "heats", "heating", "overheat", "overheats", "overheating", "throttle",
        "throttling", "benchmark", "responsive", "hang", "hangs", "freeze", "freezes",
        "crash", "crashes",
    ),
    "price": (
        "price", "priced", "pricing", "cost", "costs", "value for money",
        "value-for-money", "worth the money", "worth it", "worth every penny",
        "money", "budget", "expensive", "pricey", "cheap", "affordable", "overpriced",
        "bang for the buck", "bang for buck", "deal", "discount", "offer",
        "value", "paisa vasool", "vfm",
    ),
    "design": (
        "design", "build", "build quality", "look", "looks", "premium feel",
        "materials", "glass back", "aluminium", "aluminum", "plastic", "weight",
        "heavy", "lightweight", "slim", "thin", "bulky", "ergonomic", "grip",
        "in hand feel", "in-hand feel", "aesthetic", "colour options",
    ),
    "software": (
        "software", "os", "android", "ios", "one ui", "miui", "oxygenos", "coloros",
        "hyperos", "update", "updates", "firmware", "bloatware", "ads in ui",
        "user interface", "ui", "ux", "bug", "bugs", "buggy", "customisation",
        "customization", "security patch", "features",
    ),
    "connectivity": (
        "signal", "reception", "network", "5g", "4g", "lte", "volte", "wifi", "wi-fi",
        "bluetooth", "gps", "hotspot", "call quality", "calls drop", "call drops",
        "sim", "dual sim", "nfc", "connectivity",
    ),
    "audio": (
        "speaker", "speakers", "sound", "audio", "loudness", "volume", "bass",
        "stereo", "earpiece", "headphone", "headphones", "earphone", "dolby",
        "mic", "microphone", "music",
    ),
    "durability": (
        "durable", "durability", "sturdy", "fragile", "scratch", "scratches",
        "scratched", "crack", "cracked", "shatter", "shattered", "waterproof",
        "water resistant", "water-resistant", "ip68", "ip67", "gorilla glass",
        "drop", "dropped", "reliable", "reliability", "stopped working",
        "dead after", "defective", "faulty",
    ),
}

# Domain sentiment lexicon. Weights are magnitudes in (0, 1].
POSITIVE_LEXICON: dict[str, float] = {
    "amazing": 0.95, "excellent": 0.95, "outstanding": 0.95, "superb": 0.95,
    "fantastic": 0.9, "brilliant": 0.9, "perfect": 0.95, "flawless": 0.95,
    "great": 0.8, "good": 0.65, "nice": 0.6, "fine": 0.45, "decent": 0.5,
    "solid": 0.6, "impressive": 0.85, "stunning": 0.9, "gorgeous": 0.85,
    "beautiful": 0.8, "premium": 0.7, "love": 0.85, "loved": 0.85, "loves": 0.8,
    "like": 0.5, "liked": 0.5, "happy": 0.75, "satisfied": 0.7, "pleased": 0.7,
    "recommend": 0.75, "recommended": 0.75, "worth": 0.7, "worthy": 0.7,
    "best": 0.9, "better": 0.55, "awesome": 0.9, "incredible": 0.9,
    "exceptional": 0.9, "top notch": 0.85, "top-notch": 0.85,
    # battery
    "long lasting": 0.85, "long-lasting": 0.85, "lasts": 0.6, "lasting": 0.6,
    "all day": 0.7, "endures": 0.6, "efficient": 0.7,
    # camera / display
    "crisp": 0.8, "sharp": 0.75, "vivid": 0.8, "vibrant": 0.8, "clear": 0.7,
    "detailed": 0.7, "punchy": 0.7, "bright": 0.65, "accurate": 0.7,
    # performance
    "fast": 0.8, "faster": 0.7, "quick": 0.7, "snappy": 0.85, "smooth": 0.85,
    "seamless": 0.85, "responsive": 0.8, "powerful": 0.8, "buttery": 0.9,
    "lag free": 0.85, "lag-free": 0.85, "stable": 0.65, "reliable": 0.75,
    # price
    "affordable": 0.75, "cheap": 0.4, "reasonable": 0.65, "bargain": 0.85,
    "value for money": 0.85, "worth every penny": 0.95, "paisa vasool": 0.9,
    "well priced": 0.8, "well-priced": 0.8, "budget friendly": 0.75,
    "budget-friendly": 0.75, "competitive": 0.6,
    # durability
    "sturdy": 0.75, "durable": 0.8, "robust": 0.75,
}

NEGATIVE_LEXICON: dict[str, float] = {
    "terrible": 0.95, "horrible": 0.95, "awful": 0.95, "worst": 0.95,
    "useless": 0.9, "garbage": 0.95, "rubbish": 0.9, "pathetic": 0.9,
    "bad": 0.75, "poor": 0.8, "poorly": 0.8, "disappointing": 0.85,
    "disappointed": 0.85, "disappointment": 0.85, "mediocre": 0.6,
    "average": 0.35, "meh": 0.5, "subpar": 0.7, "sub-par": 0.7,
    "hate": 0.9, "hated": 0.9, "regret": 0.85, "annoying": 0.7,
    "frustrating": 0.8, "unacceptable": 0.9, "waste": 0.9, "waste of money": 0.95,
    "problem": 0.6, "problems": 0.6, "issue": 0.55, "issues": 0.55,
    "complaint": 0.6, "faulty": 0.85, "defective": 0.9, "broken": 0.85,
    # battery
    "drains": 0.8, "drain": 0.75, "draining": 0.8, "dies": 0.8, "died": 0.8,
    "dying": 0.8, "short battery": 0.85, "barely lasts": 0.85,
    # camera / display
    "blurry": 0.85, "blurred": 0.8, "grainy": 0.8, "noisy": 0.7,
    "washed out": 0.75, "washed-out": 0.75, "dull": 0.7, "dim": 0.7,
    "oversaturated": 0.5, "soft": 0.4, "unusable": 0.9,
    # performance
    "slow": 0.8, "sluggish": 0.85, "laggy": 0.85, "lags": 0.8, "lagging": 0.8,
    "stutters": 0.8, "stuttering": 0.8, "freezes": 0.85, "freezing": 0.85,
    "hangs": 0.8, "crashes": 0.85, "crashing": 0.85, "overheats": 0.9,
    "overheating": 0.9, "heats up": 0.8, "throttles": 0.75, "throttling": 0.75,
    "unresponsive": 0.85,
    # price
    "expensive": 0.7, "overpriced": 0.9, "pricey": 0.65, "costly": 0.7,
    "not worth": 0.9, "too much money": 0.8, "rip off": 0.95, "rip-off": 0.95,
    # durability / software
    "fragile": 0.8, "scratches easily": 0.8, "cracked": 0.8, "shattered": 0.85,
    "bloatware": 0.7, "buggy": 0.8, "bugs": 0.7, "ads": 0.6,
}

NEGATION_TOKENS: frozenset[str] = frozenset(
    {
        "not", "no", "never", "none", "nothing", "neither", "nor", "cannot",
        "cant", "can't", "wont", "won't", "dont", "don't", "doesnt", "doesn't",
        "didnt", "didn't", "isnt", "isn't", "arent", "aren't", "wasnt", "wasn't",
        "werent", "weren't", "hardly", "barely", "scarcely", "without", "lacks",
        "lack", "lacking", "fails", "failed", "less",
    }
)

INTENSIFIERS: dict[str, float] = {
    "very": 1.35, "extremely": 1.6, "really": 1.3, "super": 1.4, "so": 1.2,
    "incredibly": 1.55, "absolutely": 1.5, "highly": 1.35, "totally": 1.4,
    "insanely": 1.6, "way": 1.3, "too": 1.25, "quite": 1.1,
}

DIMINISHERS: dict[str, float] = {
    "slightly": 0.6, "somewhat": 0.7, "a bit": 0.65, "a little": 0.65,
    "kind of": 0.7, "kinda": 0.7, "sort of": 0.7, "fairly": 0.85,
    "mostly": 0.9, "reasonably": 0.85, "relatively": 0.85, "okayish": 0.6,
}

# Clause boundaries: sentiment before "but" should not leak past it.
CONTRAST_MARKERS: tuple[str, ...] = (
    " but ", " however ", " although ", " though ", " whereas ", " while ",
    " yet ", " except ", " apart from ", ", but ", "; ",
)


def aspects_for(aspect_set: str) -> tuple[str, ...]:
    return EXTENDED_ASPECTS if aspect_set == "extended" else CORE_ASPECTS


def normalise_aspect(raw: str, allowed: tuple[str, ...]) -> str | None:
    """Map a free-form LLM aspect label onto the closed taxonomy."""
    if not raw:
        return None
    key = re.sub(r"[^a-z ]", "", raw.strip().lower()).strip()
    if key in allowed:
        return key
    aliases = {
        "battery life": "battery", "charging": "battery", "power": "battery",
        "photo": "camera", "photos": "camera", "photography": "camera",
        "video": "camera", "picture quality": "camera", "image quality": "camera",
        "screen": "display", "screen quality": "display",
        "speed": "performance", "processor": "performance", "cpu": "performance",
        "gaming": "performance", "heating": "performance", "thermals": "performance",
        "cost": "price", "value": "price", "value for money": "price",
        "price value": "price", "affordability": "price",
        "build": "design", "build quality": "design", "look": "design",
        "aesthetics": "design", "ergonomics": "design",
        "os": "software", "operating system": "software", "ui": "software",
        "updates": "software", "user interface": "software",
        "network": "connectivity", "signal": "connectivity", "call quality": "connectivity",
        "sound": "audio", "speaker": "audio", "speakers": "audio",
        "reliability": "durability", "build durability": "durability",
    }
    mapped = aliases.get(key)
    if mapped and mapped in allowed:
        return mapped
    for candidate in allowed:
        if candidate in key:
            return candidate
    return None


def normalise_sentiment(raw: str) -> str | None:
    if not raw:
        return None
    key = raw.strip().lower()
    if key in SENTIMENTS:
        return key
    aliases = {
        "pos": "positive", "positive sentiment": "positive", "good": "positive",
        "neg": "negative", "negative sentiment": "negative", "bad": "negative",
        "neu": "neutral", "mixed": "neutral", "objective": "neutral",
        "none": "neutral", "unknown": "neutral",
    }
    return aliases.get(key)


@lru_cache(maxsize=1)
def _compiled_aspect_patterns() -> dict[str, re.Pattern[str]]:
    compiled: dict[str, re.Pattern[str]] = {}
    for aspect, phrases in ASPECT_KEYWORDS.items():
        ordered = sorted(phrases, key=len, reverse=True)
        alternation = "|".join(re.escape(p) for p in ordered)
        compiled[aspect] = re.compile(rf"(?<!\w)(?:{alternation})(?!\w)", re.IGNORECASE)
    return compiled


def detect_aspects(text: str, allowed: tuple[str, ...]) -> dict[str, list[str]]:
    """Return {aspect: [matched surface forms]} for a piece of text."""
    patterns = _compiled_aspect_patterns()
    found: dict[str, list[str]] = {}
    for aspect in allowed:
        pattern = patterns.get(aspect)
        if pattern is None:
            continue
        matches = pattern.findall(text)
        if matches:
            found[aspect] = matches
    return found
