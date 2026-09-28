"""Gold-label ABSA evaluation (proposal Stage 7 technical metrics).

Workflow:
1. ``python run.py sample-gold`` → CSV template with system predictions.
2. Fill ``gold_sentiment`` (and optionally ``gold_aspect``) by hand.
3. ``python run.py evaluate-absa --gold path.csv`` → metrics JSON + print.
4. Evaluation tab reads ``data/artifacts/absa_eval_latest.json``.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.models.entities import AspectSentiment, Sentence

SENTIMENTS = ("positive", "negative", "neutral")


def sample_gold_template(
    db: Session,
    *,
    n: int = 200,
    out_path: Path | None = None,
    settings: Settings | None = None,
) -> Path:
    """Export a stratified sample of aspect predictions for manual gold labels."""
    settings = settings or get_settings()
    out = out_path or (settings.artifacts_dir / "gold_absa_template.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    # Stratify by predicted sentiment so rare classes appear.
    rows: list[dict[str, Any]] = []
    per_class = max(1, n // 3)
    for sentiment in SENTIMENTS:
        stmt = (
            select(
                AspectSentiment.id,
                AspectSentiment.sentence_id,
                AspectSentiment.aspect,
                AspectSentiment.sentiment,
                AspectSentiment.method,
                Sentence.text,
            )
            .join(Sentence, Sentence.id == AspectSentiment.sentence_id)
            .where(AspectSentiment.sentiment == sentiment)
            .order_by(func.random())
            .limit(per_class)
        )
        for as_id, sent_id, aspect, pred, method, text in db.execute(stmt).all():
            rows.append(
                {
                    "aspect_sentiment_id": as_id,
                    "sentence_id": sent_id,
                    "sentence_text": text,
                    "aspect": aspect,
                    "predicted_sentiment": pred,
                    "method": method,
                    "gold_sentiment": "",  # fill: positive | negative | neutral
                    "gold_aspect": aspect,  # change only if aspect label is wrong
                    "notes": "",
                }
            )

    frame = pd.DataFrame(rows)
    if len(frame) > n:
        frame = frame.sample(n=n, random_state=42)
    frame.to_csv(out, index=False)
    return out


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    )
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "support": tp + fn,
    }


def evaluate_gold_csv(gold_path: Path, *, settings: Settings | None = None) -> dict[str, Any]:
    """Score predicted_sentiment vs gold_sentiment; write absa_eval_latest.json."""
    settings = settings or get_settings()
    frame = pd.read_csv(gold_path)
    if "gold_sentiment" not in frame.columns or "predicted_sentiment" not in frame.columns:
        raise ValueError("CSV must include predicted_sentiment and gold_sentiment columns.")

    labeled = frame[
        frame["gold_sentiment"].notna()
        & (frame["gold_sentiment"].astype(str).str.strip() != "")
    ].copy()
    if labeled.empty:
        raise ValueError("No gold_sentiment labels filled yet.")

    labeled["gold_sentiment"] = labeled["gold_sentiment"].astype(str).str.strip().str.lower()
    labeled["predicted_sentiment"] = (
        labeled["predicted_sentiment"].astype(str).str.strip().str.lower()
    )

    # Confusion matrix gold × predicted
    confusion: dict[str, dict[str, int]] = {
        g: {p: 0 for p in SENTIMENTS} for g in SENTIMENTS
    }
    for _, row in labeled.iterrows():
        g = row["gold_sentiment"]
        p = row["predicted_sentiment"]
        if g not in SENTIMENTS or p not in SENTIMENTS:
            continue
        confusion[g][p] += 1

    per_class: dict[str, dict[str, float]] = {}
    f1s: list[float] = []
    for label in SENTIMENTS:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in SENTIMENTS if other != label)
        fn = sum(confusion[label][other] for other in SENTIMENTS if other != label)
        stats = _prf(tp, fp, fn)
        per_class[label] = stats
        f1s.append(stats["f1"])

    correct = sum(confusion[s][s] for s in SENTIMENTS)
    total = int(sum(sum(row.values()) for row in confusion.values()))
    accuracy = correct / total if total else 0.0
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0

    # Pair-level: aspect+sentiment both match gold (when gold_aspect provided)
    pair_total = 0
    pair_correct = 0
    if "gold_aspect" in labeled.columns and "aspect" in labeled.columns:
        for _, row in labeled.iterrows():
            gold_aspect = str(row.get("gold_aspect") or "").strip().lower()
            pred_aspect = str(row.get("aspect") or "").strip().lower()
            if not gold_aspect:
                continue
            pair_total += 1
            if (
                gold_aspect == pred_aspect
                and row["gold_sentiment"] == row["predicted_sentiment"]
            ):
                pair_correct += 1
    pair_f1_proxy = pair_correct / pair_total if pair_total else None  # accuracy of pairs

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gold_path": str(gold_path),
        "labeled_rows": total,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion_matrix": confusion,
        "pair_exact_match_rate": round(pair_f1_proxy, 4) if pair_f1_proxy is not None else None,
        "pair_labeled_rows": pair_total,
        "note": (
            "Macro-F1 is the main sentiment metric (handles class imbalance). "
            "Pair exact-match rate is aspect+sentiment both correct when gold_aspect is filled."
        ),
    }

    out = settings.artifacts_dir / "absa_eval_latest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["saved_to"] = str(out)
    return report


def load_latest_absa_eval(settings: Settings | None = None) -> dict[str, Any] | None:
    settings = settings or get_settings()
    path = settings.artifacts_dir / "absa_eval_latest.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
