"""Parquet serialization and artifact directory caching utilities."""

from pathlib import Path
from typing import Union
import pandas as pd

from config import PATH_ARTIFACTS_DIR, PATH_OUTPUT_DIR


def save_parquet(df: pd.DataFrame, path: Union[Path, str]) -> None:
    """Save a pandas DataFrame to parquet with snappy compression."""
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(target_path, engine="pyarrow", compression="snappy", index=False)


def load_parquet(path: Union[Path, str]) -> pd.DataFrame:
    """Load a pandas DataFrame from a parquet file."""
    return pd.read_parquet(Path(path), engine="pyarrow")


def ensure_artifact_dirs() -> None:
    """Ensure all required artifact and output cache directories exist."""
    subdirs = [
        "normalized",
        "blocking",
        "features",
        "models",
        "embeddings",
        "splits",
        "train_pairs",
        "error_analysis",
    ]
    PATH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PATH_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    for subdir in subdirs:
        (PATH_ARTIFACTS_DIR / subdir).mkdir(parents=True, exist_ok=True)
