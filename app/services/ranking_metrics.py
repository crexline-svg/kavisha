"""Ranking-quality metrics from per-phone user ratings (graded relevance).

Per-phone satisfaction (1–5) is treated as the user's graded judgment of how well
each recommended phone fits their requirements. System order is the recommender's
rank. Primary metric: NDCG@3. Secondary: Spearman rank correlation when enough
phones were rated in a session.
"""

from __future__ import annotations

import math
from typing import Sequence


def _dcg(relevances: Sequence[float], k: int) -> float:
    total = 0.0
    for i, rel in enumerate(relevances[:k], start=1):
        gain = (2.0 ** float(rel)) - 1.0
        total += gain / math.log2(i + 1)
    return total


def ndcg_at_k(relevances_in_system_order: Sequence[float], k: int = 3) -> float | None:
    """NDCG@k for one ranking session.

    ``relevances_in_system_order`` must be sorted by system rank ascending
    (position 0 = system #1). Values are graded relevance (here: satisfaction 1–5).
    """
    if not relevances_in_system_order or k < 1:
        return None
    dcg = _dcg(relevances_in_system_order, k)
    ideal = sorted((float(r) for r in relevances_in_system_order), reverse=True)
    idcg = _dcg(ideal, k)
    if idcg <= 0:
        return 0.0
    return round(dcg / idcg, 4)


def _average_ranks(values: Sequence[float]) -> list[float]:
    """Competition ranks with average ties (1-based)."""
    indexed = sorted(enumerate(values), key=lambda pair: pair[1])
    ranks = [0.0] * len(values)
    i = 0
    n = len(indexed)
    while i < n:
        j = i
        while j + 1 < n and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        for t in range(i, j + 1):
            ranks[indexed[t][0]] = avg
        i = j + 1
    return ranks


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    n = len(xs)
    if n < 2 or n != len(ys):
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return None
    return round(num / (den_x * den_y), 4)


def spearman_correlation(
    system_ranks: Sequence[int],
    relevances: Sequence[float],
    *,
    min_items: int = 3,
) -> float | None:
    """Spearman ρ between system rank and user preference order.

    Higher relevance = better for the user. System rank 1 is best for the system.
    Correlates system_ranks with preference ranks derived from relevances
    (best relevance → preference rank 1).
    """
    if len(system_ranks) != len(relevances) or len(system_ranks) < min_items:
        return None
    pref_ranks = _average_ranks([-float(r) for r in relevances])
    sys_ranks = [float(r) for r in system_ranks]
    return _pearson(sys_ranks, pref_ranks)


def session_ranking_metrics(
    phone_ratings: Sequence[dict],
    *,
    k: int = 3,
    min_spearman_items: int = 3,
) -> dict[str, float | None]:
    """Compute NDCG@k and Spearman for one feedback session's phone_ratings."""
    rows: list[tuple[int, float]] = []
    for item in phone_ratings:
        if not isinstance(item, dict):
            continue
        rank = item.get("rank")
        sat = item.get("satisfaction")
        if rank is None or sat is None:
            continue
        rows.append((int(rank), float(sat)))
    if not rows:
        return {"ndcg_at_3": None, "spearman": None}

    rows.sort(key=lambda pair: pair[0])
    ranks = [r for r, _ in rows]
    relevances = [s for _, s in rows]
    return {
        "ndcg_at_3": ndcg_at_k(relevances, k=k),
        "spearman": spearman_correlation(
            ranks, relevances, min_items=min_spearman_items
        ),
    }
