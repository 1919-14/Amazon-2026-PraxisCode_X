"""Inspect cached artifacts (schemas, row counts, label balance)."""
import os
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
A = os.path.join(ROOT, "artifacts")


def peek(path, name, n=3):
    if not os.path.exists(path):
        print(f"[missing] {name}: {path}")
        return
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        print(f"[error] {name}: {e}")
        return
    print(f"\n=== {name} === shape={df.shape}")
    print("cols:", list(df.columns))
    if "label" in df.columns:
        print("label counts:", df["label"].value_counts().to_dict())
    if "country" in df.columns:
        print("country counts:", df["country"].value_counts().to_dict())
    if "neg_type" in df.columns:
        print("neg_type counts:", df["neg_type"].value_counts().to_dict())
    print(df.head(n).to_string()[:1200])


peek(os.path.join(A, "features", "features_variant_a_india.parquet"), "features_variant_a_india")
peek(os.path.join(A, "features", "features_variant_a.parquet"), "features_variant_a")
peek(os.path.join(A, "train_pairs", "variant_a.parquet"), "variant_a pairs")
peek(os.path.join(A, "train_pairs", "variant_b.parquet"), "variant_b pairs")
peek(os.path.join(A, "models", "oof_a.parquet"), "oof_a")
peek(os.path.join(A, "models", "oof_calibrated_a.parquet"), "oof_calibrated_a")
peek(os.path.join(A, "signals", "candidates_train_train.parquet"), "signals train_train")
peek(os.path.join(A, "signals", "candidates_test_all.parquet"), "signals test_all")

for name in ("features_variant_a_india.meta.json", "features_variant_a.meta.json"):
    p = os.path.join(A, "features", name)
    if os.path.exists(p):
        print(f"\n=== {name} ===")
        print(open(p, encoding="utf-8").read())

p = os.path.join(ROOT, "output", "l6_pairs_report.json")
if os.path.exists(p):
    print("\n=== l6_pairs_report.json ===")
    print(open(p, encoding="utf-8").read()[:2000])
