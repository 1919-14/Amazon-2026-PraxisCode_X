"""Decision v2: entity-level expected-F0.5 selection vs a global threshold.

Scored against the REAL train ground truth (so candidate recall losses count as
false negatives -- this is the actual competition metric), on the cached India OOF.

Run:
  venv/Scripts/python.exe business_entity_resolution/experiments/SCALE_UP/decision_v2.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd

csv.field_size_limit(10_000_000)
ROOT = Path(__file__).resolve().parents[3]
DS = ROOT / "DATA SET" / "student_resource" / "dataset"
OOF = ROOT / "business_entity_resolution" / "artifacts" / "models" / "oof_calibrated_a.parquet"


def load_gt():
    gt = {}
    with open(DS / "train" / "train_ground_truth.tsv", encoding="utf-8", newline="") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for row in r:
            if len(row) >= 2:
                gt[row[0]] = frozenset(x for x in row[1].split(",") if x)
    return gt


def f_beta(pred: set, true: frozenset, beta: float = 0.5) -> float:
    if not true:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p = tp / len(pred)
    r = tp / len(true)
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r)


def main():
    df = pd.read_parquet(OOF)
    prob_col = "calibrated_prob" if "calibrated_prob" in df.columns else "oof_prob"
    df = df[["s1_id", "cand_id", prob_col]].rename(columns={prob_col: "prob"})
    df = df.sort_values("s1_id", kind="stable").reset_index(drop=True)
    gt = load_gt()

    ids = df["s1_id"].to_numpy()
    cands = df["cand_id"].to_numpy()
    prob = df["prob"].to_numpy()
    uniq, starts = np.unique(ids, return_index=True)
    ends = np.append(starts[1:], len(ids))
    print(f"OOF rows {len(df):,} | entities {len(uniq):,}")

    # Precompute per entity: candidate ids, probs (sorted desc), gt set
    entities = []
    for s, e in zip(starts, ends):
        c = cands[s:e]
        p = prob[s:e]
        order = np.argsort(-p, kind="stable")
        entities.append((c[order], p[order], gt.get(ids[s], frozenset())))

    def score_all(pred_fn):
        total = 0.0
        for c, p, true in entities:
            total += f_beta(set(pred_fn(c, p)), true)
        return total / len(entities)

    # ---- oracle (perfect filter within candidates) ----
    def oracle_fn(c, p):
        _, _, true = None, None, None  # placeholder
    total = 0.0
    for c, p, true in entities:
        total += f_beta(set(c) & set(true), true)
    oracle_real = total / len(entities)
    print(f"\nORACLE (true matches inside candidates only): {oracle_real:.4f}")

    # ---- baseline: global threshold rule ----
    best = (None, -1.0)
    for tau in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        for tau_s in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            def fn(c, p, tau=tau, tau_s=tau_s):
                top = p[0] if len(p) else 0.0
                if top < tau_s:
                    return []
                return [x for x, pp in zip(c, p) if pp >= tau]
            s = score_all(fn)
            if s > best[1]:
                best = ((tau, tau_s), s)
    print(f"BASELINE global: tau_match={best[0][0]} tau_s={best[0][1]} -> {best[1]:.4f}")

    # ---- entity-level expected F0.5 ----
    print("\nENTITY-LEVEL expected-F0.5 selection:")
    best_ent = (None, -1.0)
    for recall_hat in [0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.0]:
        def fn(c, p, rh=recall_hat):
            total_q = float(np.sum(p))
            T = max(total_q / rh, 1e-9)
            csum = np.cumsum(p)
            k = np.arange(1, len(p) + 1)
            e_p = csum / k
            e_r = np.minimum(csum / T, 1.0)
            e_f = (1.25 * e_p * e_r) / (0.25 * e_p + e_r + 1e-12)
            e_empty = float(np.prod(1.0 - np.clip(p, 0, 1)))
            bk = int(np.argmax(e_f)) + 1
            if e_empty > e_f[bk - 1]:
                return []
            return list(c[:bk])
        s = score_all(fn)
        print(f"  recall_hat={recall_hat:.2f} -> {s:.4f}")
        if s > best_ent[1]:
            best_ent = (recall_hat, s)
    print(f"\nBEST entity-level: recall_hat={best_ent[0]} -> {best_ent[1]:.4f}")
    print(f"DELTA vs baseline: {best_ent[1] - best[1]:+.4f}")
    print(f"Gap to oracle: {oracle_real - best_ent[1]:+.4f}")


if __name__ == "__main__":
    main()
