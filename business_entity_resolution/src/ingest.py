"""TSV ingest utilities with strict schema assertions and ground truth parsing."""

from typing import Any, Optional
import pandas as pd

from config import (
    PATH_TEST_S1,
    PATH_TEST_S2,
    PATH_TEST_S3,
    PATH_TRAIN_GT,
    PATH_TRAIN_S1,
    PATH_TRAIN_S2,
    PATH_TRAIN_S3,
)

EXPECTED_RECORD_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
EXPECTED_GT_COLUMNS = ["source1_entity_id", "matched_entity_ids"]


def load_split(split: str) -> dict[str, Any]:
    """Load all TSV files for a given split (train or test) with string dtype and strict validation."""
    if split == "train":
        s1_path, s2_path, s3_path = PATH_TRAIN_S1, PATH_TRAIN_S2, PATH_TRAIN_S3
        gt_path: Optional[Any] = PATH_TRAIN_GT
    elif split == "test":
        s1_path, s2_path, s3_path = PATH_TEST_S1, PATH_TEST_S2, PATH_TEST_S3
        gt_path = None
    else:
        raise ValueError(f"Unknown split '{split}'; expected 'train' or 'test'")

    s1_df = pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False)
    s2_df = pd.read_csv(s2_path, sep="\t", dtype=str, keep_default_na=False)
    s3_df = pd.read_csv(s3_path, sep="\t", dtype=str, keep_default_na=False)

    for name, df, prefix in [("s1", s1_df, "S1-"), ("s2", s2_df, "S2-"), ("s3", s3_df, "S3-")]:
        if list(df.columns) != EXPECTED_RECORD_COLUMNS:
            raise AssertionError(f"{split}_{name} columns mismatch: {list(df.columns)} != {EXPECTED_RECORD_COLUMNS}")
        if not df["entity_id"].str.startswith(prefix).all():
            invalid_count = (~df["entity_id"].str.startswith(prefix)).sum()
            raise AssertionError(f"{split}_{name} has {invalid_count} records not starting with '{prefix}'")

    gt_df = None
    if gt_path is not None and gt_path.exists():
        gt_df = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)
        if list(gt_df.columns) != EXPECTED_GT_COLUMNS:
            raise AssertionError(f"GT columns mismatch: {list(gt_df.columns)} != {EXPECTED_GT_COLUMNS}")

    return {"s1": s1_df, "s2": s2_df, "s3": s3_df, "gt": gt_df}


def parse_ground_truth(gt_df: pd.DataFrame, s1_df: Optional[pd.DataFrame] = None) -> dict[str, set[str]]:
    """Parse ground truth DataFrame into mapping of reference entity ID to set of matching candidate IDs."""
    gt_map: dict[str, set[str]] = {}
    for _, row in gt_df.iterrows():
        s1_id = row["source1_entity_id"]
        raw_matches = row["matched_entity_ids"]
        if raw_matches:
            matches = {m.strip() for m in raw_matches.split(",") if m.strip()}
        else:
            matches = set()
        gt_map[s1_id] = matches

    if s1_df is not None:
        valid_s1_ids = set(s1_df["entity_id"])
        gt_s1_ids = set(gt_map.keys())
        missing_in_s1 = gt_s1_ids - valid_s1_ids
        if missing_in_s1:
            raise AssertionError(f"Ground truth contains {len(missing_in_s1)} entity IDs not present in Source 1")

    return gt_map
