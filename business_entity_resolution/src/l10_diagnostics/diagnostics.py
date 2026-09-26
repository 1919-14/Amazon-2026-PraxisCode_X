"""L10.5: diagnostic error analysis over the decision engine's output.

Where L10 produces the decision, L10.5 explains the *loss*. Macro F0.5 is
averaged per Source-1 entity, so the first cut is the per-entity error class:

  * ``true_singleton``  – correctly predicted empty (scores 1.0);
  * ``false_positive``  – predicted a match for a true singleton (scores 0.0,
    the most expensive error under a precision-weighted metric);
  * ``miss``            – true matches exist but none were predicted (0.0);
  * ``exact``           – predicted set equals the ground truth (1.0);
  * ``partial``         – overlap exists but precision/recall are imperfect.

Those classes are then broken down by the dimensions the brief asks for:
country, matched-source composition (S2 / S3 / both) and reference name/address
length, plus script and missing-address flags from the L2 normalized records.
Finally, open-set entities (France, test-only) get a spot-check: with no ground
truth we can only inspect volume and confidence, so the report surfaces the
top-scoring predictions for manual review.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

try:  # pragma: no cover - path fallback when run outside the package
    from src.l1_validation.metrics import f_beta
except ImportError:  # pragma: no cover
    from l1_validation.metrics import f_beta

try:  # pragma: no cover
    from src.l3_l5_blocking.buckets import iter_source_shards
except ImportError:  # pragma: no cover
    from l3_l5_blocking.buckets import iter_source_shards

ERROR_CLASSES = ("true_singleton", "false_positive", "miss", "exact", "partial")

REF_METADATA_COLUMNS = [
    "entity_id",
    "country_norm",
    "name_core",
    "script_type",
    "is_missing_name",
    "is_missing_addr",
    "addr_norm",
]


def length_bucket(n: int) -> str:
    """Bucket a character length into fixed ranges for reporting."""
    if n <= 0:
        return "0"
    if n <= 10:
        return "1-10"
    if n <= 20:
        return "11-20"
    if n <= 40:
        return "21-40"
    return "41+"


def source_composition(ids: Iterable[str]) -> str:
    """Classify an id set by which source files it draws from."""
    ids = list(ids)
    n2 = sum(1 for i in ids if str(i).startswith("S2-"))
    n3 = sum(1 for i in ids if str(i).startswith("S3-"))
    if n2 and n3:
        return "s2+s3"
    if n2:
        return "s2"
    if n3:
        return "s3"
    return "none"


def classify_entity(pred: Iterable[str], gt: Iterable[str]) -> dict:
    """Score one entity and label its error class, mirroring the official metric."""
    pred_set = set(pred)
    gt_set = set(gt)

    if not gt_set and not pred_set:
        return {"error_class": "true_singleton", "f05": 1.0, "precision": 1.0, "recall": 1.0}
    if not gt_set and pred_set:
        return {"error_class": "false_positive", "f05": 0.0, "precision": 0.0, "recall": 0.0}
    if gt_set and not pred_set:
        return {"error_class": "miss", "f05": 0.0, "precision": 0.0, "recall": 0.0}

    tp = len(pred_set & gt_set)
    precision = tp / len(pred_set)
    recall = tp / len(gt_set)
    if pred_set == gt_set:
        error_class = "exact"
    else:
        error_class = "partial"
    return {
        "error_class": error_class,
        "f05": f_beta(precision, recall, 0.5),
        "precision": precision,
        "recall": recall,
    }


def aggregate(records: list[dict]) -> dict:
    """Aggregate per-entity records into a macro/mean/error-count summary."""
    n = len(records)
    if n == 0:
        return {"n": 0, "macro_f05": 0.0, "mean_precision": 0.0, "mean_recall": 0.0,
                "error_classes": {cls: 0 for cls in ERROR_CLASSES}}

    classes = Counter(r["error_class"] for r in records)
    singleton_scores = [r["f05"] for r in records if r["error_class"] == "true_singleton"]
    match_scores = [r["f05"] for r in records if r["error_class"] not in ("true_singleton",)]

    return {
        "n": n,
        "macro_f05": sum(r["f05"] for r in records) / n,
        "mean_precision": sum(r["precision"] for r in records) / n,
        "mean_recall": sum(r["recall"] for r in records) / n,
        "n_singletons": len(singleton_scores),
        "singleton_f05": (sum(singleton_scores) / len(singleton_scores)) if singleton_scores else 0.0,
        "match_f05": (sum(match_scores) / len(match_scores)) if match_scores else 0.0,
        "error_classes": {cls: int(classes.get(cls, 0)) for cls in ERROR_CLASSES},
    }


def group_metrics(records: list[dict], key: str, top_n: int | None = None) -> dict:
    """Aggregate records by a categorical key, largest groups first."""
    buckets: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        buckets[str(record.get(key, "other"))].append(record)
    ordered = sorted(buckets.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    if top_n is not None:
        ordered = ordered[:top_n]
    return {name: aggregate(rows) for name, rows in ordered}


def error_focus(records: list[dict]) -> dict:
    """Summarise where the loss concentrates: count and F0.5 mass per class."""
    total = len(records)
    classes = Counter(r["error_class"] for r in records)
    f05_by_class = {cls: 0.0 for cls in ERROR_CLASSES}
    for record in records:
        f05_by_class[record["error_class"]] += record["f05"]
    return {
        "counts": {cls: int(classes.get(cls, 0)) for cls in ERROR_CLASSES},
        "fraction": {cls: (classes.get(cls, 0) / total if total else 0.0) for cls in ERROR_CLASSES},
        "f05_mass": {cls: round(value, 4) for cls, value in f05_by_class.items()},
    }


def load_predictions_tsv(path: str | Path) -> dict[str, list[str]]:
    """Load a ``source1_entity_id <tab> comma-separated ids`` TSV into a dict."""
    predictions: dict[str, list[str]] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader, None)
        if header is not None and header[0].strip().lower() != "source1_entity_id":
            raise ValueError(f"unexpected header in {path}: {header}")
        for row in reader:
            if not row:
                continue
            s1_id = row[0]
            raw = row[1] if len(row) > 1 else ""
            predictions[s1_id] = [m for m in raw.split(",") if m]
    return predictions


def load_reference_metadata(
    split: str,
    ref_ids: set[str],
    source_indices: tuple[int, ...] = (1,),
) -> dict[str, dict]:
    """Load normalized metadata for the given references from the L2 shards.

    Only the columns needed for the breakdown are read, and rows are filtered by
    the requested ids, so memory stays proportional to the reference set.
    """
    wanted = set(ref_ids)
    metadata: dict[str, dict] = {}
    for source_idx in source_indices:
        for shard in iter_source_shards(split, source_idx):
            df = pd.read_parquet(shard, columns=REF_METADATA_COLUMNS)
            df = df[df["entity_id"].isin(wanted)]
            if df.empty:
                continue
            for record in df.to_dict(orient="records"):
                metadata[record["entity_id"]] = record
    return metadata


def france_spot_check(
    records: list[dict],
    top_n: int = 10,
) -> dict:
    """Summarise open-set predictions where no ground truth exists.

    Args:
        records: one dict per open-set reference with keys ``s1_id``, ``pred``,
            ``top_score`` (optional) and ``meta`` (optional normalized record).
        top_n: how many highest-confidence examples to include for manual review.

    Returns:
        Volume/confidence summary plus a sample of the top predictions.
    """
    n = len(records)
    if n == 0:
        return {"n": 0, "n_non_empty": 0, "n_empty": 0, "mean_pred_len": 0.0,
                "top_score_quantiles": {}, "samples": []}

    non_empty = [r for r in records if r.get("pred")]
    pred_len = [len(r["pred"]) for r in records]
    scores = sorted(float(r["top_score"]) for r in records if r.get("top_score") is not None)

    quantiles: dict[str, float] = {}
    if scores:
        for q in (0.0, 0.25, 0.5, 0.75, 1.0):
            idx = min(len(scores) - 1, int(round(q * (len(scores) - 1))))
            quantiles[f"p{int(q * 100)}"] = round(scores[idx], 4)

    def sort_key(record: dict) -> float:
        return -float(record.get("top_score", 0.0))

    samples = []
    for record in sorted(non_empty, key=sort_key)[:top_n]:
        meta = record.get("meta") or {}
        samples.append(
            {
                "s1_id": record["s1_id"],
                "top_score": round(float(record.get("top_score", 0.0)), 4),
                "n_matched": len(record["pred"]),
                "matched": list(record["pred"])[:5],
                "name_core": (meta.get("name_core") or "")[:60],
                "is_missing_addr": bool(meta.get("is_missing_addr", False)),
            }
        )

    return {
        "n": n,
        "n_non_empty": len(non_empty),
        "n_empty": n - len(non_empty),
        "mean_pred_len": sum(pred_len) / n,
        "top_score_quantiles": quantiles,
        "samples": samples,
    }
