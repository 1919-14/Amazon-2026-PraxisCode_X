"""Global configuration, directory paths, and hyperparameters for Layer 0-13."""

from pathlib import Path

# Resolve base project directory dynamically
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

# Resolve dataset directory across workspace layouts
if (PROJECT_ROOT / "dataset").exists():
    DATASET_DIR = PROJECT_ROOT / "dataset"
elif (WORKSPACE_ROOT / "DATA SET" / "student_resource" / "dataset").exists():
    DATASET_DIR = WORKSPACE_ROOT / "DATA SET" / "student_resource" / "dataset"
elif (PROJECT_ROOT / "DATA SET" / "student_resource" / "dataset").exists():
    DATASET_DIR = PROJECT_ROOT / "DATA SET" / "student_resource" / "dataset"
else:
    DATASET_DIR = Path("DATA SET/student_resource/dataset")

# Dataset File Paths
PATH_TRAIN_S1 = DATASET_DIR / "train" / "train_source1.tsv"
PATH_TRAIN_S2 = DATASET_DIR / "train" / "train_source2.tsv"
PATH_TRAIN_S3 = DATASET_DIR / "train" / "train_source3.tsv"
PATH_TRAIN_GT = DATASET_DIR / "train" / "train_ground_truth.tsv"

PATH_TEST_S1 = DATASET_DIR / "test" / "test_source1.tsv"
PATH_TEST_S2 = DATASET_DIR / "test" / "test_source2.tsv"
PATH_TEST_S3 = DATASET_DIR / "test" / "test_source3.tsv"

# Output and Artifact Directories
PATH_OUTPUT_DIR = PROJECT_ROOT / "output"
PATH_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Global Random Seed
SEED = 42

# Layer 3-5 Blocking and Candidate Selection Placeholders
BLOCKING_TOPK: dict[str, int] = {}
RRF_K_GRID: list[int] = [10, 20, 40, 60]
COARSE_THRESHOLDS: dict[str, float] = {}
BUDGET_TIERS: dict[str, int] = {}

# Layer 10 Decision Engine Placeholders
TAU_MATCH_GRID: list[float] = []
TAU_S_GRID: list[float] = []
