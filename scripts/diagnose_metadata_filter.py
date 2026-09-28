#!/usr/bin/env python3
"""Diagnose why Amazon-Reviews-2023 metadata rows become smartphone candidates.

Streams product metadata (no full download) and prints counters at each filter
gate — both a simple keyword/brand funnel (like a Colab notebook) and the
production filters used by ``build_hf_corpus.py`` / ``is_real_smartphone``.

Usage::

    python scripts/diagnose_metadata_filter.py
    python scripts/diagnose_metadata_filter.py --limit 50000
    python scripts/diagnose_metadata_filter.py --limit 100000 --brands samsung apple google
    python scripts/diagnose_metadata_filter.py --limit 20000 --show-samples 5
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.scrapers.amazon import looks_like_phone
from app.services.hf_ingest import (
    META_PARQUET,
    _ACCESSORY_EXTRA,
    _PHONE_PRODUCT,
    _has_phone_category,
    _stream_parquet,
)
from scripts.build_hf_corpus import is_real_smartphone

# Simple funnel lists (easy to tweak in Colab-style diagnostics).
DEFAULT_BRANDS = (
    "samsung",
    "apple",
    "google",
    "xiaomi",
    "oneplus",
    "motorola",
    "nokia",
    "oppo",
    "vivo",
    "realme",
    "honor",
    "sony",
    "lg",
    "huawei",
    "zte",
    "tcl",
    "nothing",
)

EXCLUDE_TERMS = (
    "case for",
    "cover for",
    "screen protector",
    "tempered glass",
    "charger",
    "cable",
    "earbud",
    "headphone",
    "watch band",
    "watch strap",
    "power bank",
    "memory card",
    "sim card",
    "holder for",
    "mount for",
    "compatible with",
    "replacement",
)

PHONE_KEYWORDS = (
    "smartphone",
    "smart phone",
    "cell phone",
    "unlocked",
    "dual sim",
    "5g",
    "4g lte",
    "iphone",
    "galaxy s",
    "galaxy a",
    "galaxy z",
    "pixel ",
    "redmi",
    "poco",
    "moto g",
    " android ",
)


def diagnose_simple_funnel(
    row: dict[str, Any],
    *,
    brands: tuple[str, ...],
    stats: Counter[str],
) -> None:
    """Keyword/brand funnel matching the user's Colab-style Counter."""
    stats["total_seen"] += 1

    title = str(row.get("title") or "").lower()
    if not title:
        stats["no_title"] += 1
        return

    if brands:
        if any(brand in title for brand in brands):
            stats["brand_match"] += 1
        else:
            stats["brand_fail"] += 1
            return

    if any(term in title for term in EXCLUDE_TERMS):
        stats["accessory_excluded"] += 1
        return

    if any(word in title for word in PHONE_KEYWORDS):
        stats["phone_keyword_match"] += 1
    else:
        stats["phone_keyword_fail"] += 1
        return

    stats["simple_accepted"] += 1


def diagnose_production_filter(
    row: dict[str, Any],
    *,
    stats: Counter[str],
    samples: dict[str, list[str]],
    max_samples: int,
) -> None:
    """Production ``is_real_smartphone`` gates used by the corpus builder."""
    stats["prod_total_seen"] += 1

    title = (row.get("title") or "").strip()
    if not title:
        stats["prod_no_title"] += 1
        return

    if _ACCESSORY_EXTRA.search(title):
        stats["prod_accessory_extra"] += 1
        if len(samples["prod_accessory_extra"]) < max_samples:
            samples["prod_accessory_extra"].append(title[:120])
        return

    if not looks_like_phone(title):
        stats["prod_looks_like_phone_fail"] += 1
        if len(samples["prod_looks_like_phone_fail"]) < max_samples:
            samples["prod_looks_like_phone_fail"].append(title[:120])
        return

    if not _PHONE_PRODUCT.search(title):
        stats["prod_phone_product_fail"] += 1
        if len(samples["prod_phone_product_fail"]) < max_samples:
            samples["prod_phone_product_fail"].append(title[:120])
        return

    if not _has_phone_category(row.get("categories")):
        stats["prod_category_fail"] += 1
        if len(samples["prod_category_fail"]) < max_samples:
            samples["prod_category_fail"].append(title[:120])
        return

    stats["prod_accepted"] += 1
    if len(samples["prod_accepted"]) < max_samples:
        samples["prod_accepted"].append(title[:120])


def run_diagnosis(
    *,
    limit: int,
    brands: tuple[str, ...],
    show_samples: int,
) -> tuple[Counter[str], dict[str, list[str]]]:
    simple = Counter()
    production = Counter()
    samples: dict[str, list[str]] = {
        "prod_accessory_extra": [],
        "prod_looks_like_phone_fail": [],
        "prod_phone_product_fail": [],
        "prod_category_fail": [],
        "prod_accepted": [],
    }

    print(f"Streaming up to {limit:,} metadata rows from:\n  {META_PARQUET}\n")

    for index, row in enumerate(_stream_parquet(META_PARQUET), start=1):
        record = dict(row)
        diagnose_simple_funnel(record, brands=brands, stats=simple)
        diagnose_production_filter(
            record,
            stats=production,
            samples=samples,
            max_samples=show_samples,
        )
        if is_real_smartphone(record):
            production["prod_is_real_smartphone_check"] += 1

        if index % 10_000 == 0:
            print(
                f"  … {index:,} rows | simple accepted {simple['simple_accepted']:,} | "
                f"prod accepted {production['prod_accepted']:,}",
                flush=True,
            )
        if index >= limit:
            break

    combined = Counter()
    combined.update(simple)
    combined.update(production)
    return combined, samples


def _print_section(title: str, keys: list[str], stats: Counter[str]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for key in keys:
        print(f"  {key:32} {stats.get(key, 0):>8,}")
    total = stats.get(keys[0], 0)
    accepted_key = keys[-1]
    accepted = stats.get(accepted_key, 0)
    if total:
        print(f"  {'accept rate':32} {100 * accepted / total:>7.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=20_000,
        help="How many metadata rows to scan (default: 20000).",
    )
    parser.add_argument(
        "--brands",
        nargs="*",
        default=[],
        help="If set, simple funnel only keeps titles containing these brands.",
    )
    parser.add_argument(
        "--show-samples",
        type=int,
        default=3,
        help="Example titles per production rejection reason (default: 3).",
    )
    args = parser.parse_args()

    brand_tuple = tuple(b.lower() for b in args.brands) if args.brands else DEFAULT_BRANDS
    if args.brands:
        print("Simple funnel brand filter:", ", ".join(brand_tuple))
    else:
        print("Simple funnel brand filter: default list (use --brands to override)")

    stats, samples = run_diagnosis(
        limit=args.limit,
        brands=brand_tuple,
        show_samples=args.show_samples,
    )

    print("\n" + "=" * 60)
    print("SIMPLE FUNNEL (brand -> exclude accessory terms -> phone keywords)")
    _print_section(
        "Counters",
        [
            "total_seen",
            "no_title",
            "brand_fail",
            "brand_match",
            "accessory_excluded",
            "phone_keyword_fail",
            "phone_keyword_match",
            "simple_accepted",
        ],
        stats,
    )

    print("\n" + "=" * 60)
    print("PRODUCTION FILTER (build_hf_corpus / is_real_smartphone)")
    _print_section(
        "Counters",
        [
            "prod_total_seen",
            "prod_no_title",
            "prod_accessory_extra",
            "prod_looks_like_phone_fail",
            "prod_phone_product_fail",
            "prod_category_fail",
            "prod_accepted",
            "prod_is_real_smartphone_check",
        ],
        stats,
    )

    if args.show_samples:
        print("\nSample titles (production gates)")
        for reason, titles in samples.items():
            if not titles:
                continue
            print(f"\n  [{reason}]")
            for title in titles:
                print(f"    - {title}")

    print(
        f"\nNote: scanned {stats['total_seen']:,} rows. "
        "Raise --limit to approximate full-category counts."
    )


if __name__ == "__main__":
    main()
