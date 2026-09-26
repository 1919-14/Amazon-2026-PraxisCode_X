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

# ---------------------------------------------------------------------------
# Layer 3: Country-Stratified Multi-Channel Blocking
# ---------------------------------------------------------------------------
# Country buckets are treated as an open set; unknown labels fall through as-is.
L3_COUNTRIES: tuple[str, ...] = ("us", "india", "france")

# Maximum candidates returned per channel per reference entity (L4 fuses these).
L3_CHANNEL_TOPK: int = 50

# Channel A (exact normalized name + postal/house key): drop keys that are too common.
L3_A_MAX_POSTING: int = 2000

# Channel C (high-IDF token inverted index).
L3_C_MIN_DF: int = 2
L3_C_MAX_DF_FRAC: float = 0.05

# Channel D (character 2-4 gram TF-IDF nearest neighbours).
L3_D_NGRAM_RANGE: tuple[int, int] = (2, 4)
L3_D_MIN_DF: int = 2
L3_D_MAX_DF_FRAC: float = 0.05
L3_ENABLE_CHAR_CHANNEL: bool = True

# Sparse query batch size (kept for API compatibility; retrieval is per-query).
L3_QUERY_BATCH: int = 64

# Rare-term inverted retrieval: per query, only the rarest terms are scanned.
# This bounds cost by posting-list length instead of vocabulary size.
L3_CHANNEL_MAX_QUERY_TERMS: int = 12
L3_CHANNEL_MAX_POSTING_SCAN: int = 10000

# Development / smoke-test caps. None means "use everything".
L3_MAX_CANDIDATES: int | None = None
L3_MAX_REFS: int | None = None

# ---------------------------------------------------------------------------
# Layer 4: Reciprocal Rank Fusion
# ---------------------------------------------------------------------------
RRF_K_GRID: list[int] = [10, 20, 40, 60]

# ---------------------------------------------------------------------------
# Layer 5: Adaptive Candidate Truncation
# ---------------------------------------------------------------------------
# Target compactness band from the challenge brief (candidates per S1 entity).
L5_K_TARGET: float = 6.25
L5_K_MIN_TARGET: float = 5.5
L5_K_MAX_TARGET: float = 7.0

# Hard per-reference bounds applied after the adaptive score threshold.
L5_K_MIN: int = 1
L5_K_MAX: int = 12

# Default fraction of the top coarse score a candidate must retain to survive.
L5_COARSE_RATIO: float = 0.5
L5_RATIO_GRID: list[float] = [0.3, 0.5, 0.7, 0.9]

# Coarse score = w_rrf * normalized RRF + w_agree * channel agreement + w_exact * exact-key hit.
L5_WEIGHT_RRF: float = 0.6
L5_WEIGHT_AGREE: float = 0.25
L5_WEIGHT_EXACT: float = 0.15

# ---------------------------------------------------------------------------
# Layer 6: Training Pair Construction & Hard-Negative A/B Test
# ---------------------------------------------------------------------------
# Sampling ratio per reference: 1 positive : N hard negatives : M easy negatives.
L6_POS_HARD_RATIO: int = 3
L6_POS_EASY_RATIO: int = 2

# Singletons (no true match) still need negatives so the model learns to reject.
L6_SINGLETON_HARD: int = 3
L6_SINGLETON_EASY: int = 2

# Reservoir of random Source 2/3 ids per country used to draw easy negatives.
L6_EASY_POOL_SIZE: int = 200_000
L6_SEED: int = 42

# Development cap on the number of reference entities sampled (None = all).
L6_MAX_REFERENCES: int | None = None

# Layer 7-9 feature/model placeholders
L7_CHAR_NGRAM_SIZE: int = 3
L8_N_FOLDS: int = 5
L9_ENABLE_CALIBRATION: bool = True

# ---------------------------------------------------------------------------
# Layer 10: Decision Engine & Singleton Protection
# ---------------------------------------------------------------------------
# Joint grid searched jointly with the L1 scorer: a candidate is kept when its
# score >= max(tau_match, top - margin); an entity is a singleton when its top
# score < tau_s.
TAU_MATCH_GRID: list[float] = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
TAU_S_GRID: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
L10_MARGIN: float = 0.05

# Countries the matcher was trained on. Anything else is "open set": France is
# test-only, so its decision rule must be more conservative.
L10_SEEN_COUNTRIES: tuple[str, ...] = ("us", "india")
L10_OPEN_SET_COUNTRIES: tuple[str, ...] = ("france",)

# Open-set fallback veto: raise the singleton threshold by the boost and veto to
# empty whenever the top candidate score is below the minimum confidence.
L10_OPEN_SET_TAU_BOOST: float = 0.10
L10_OPEN_SET_VETO_MIN_CONFIDENCE: float = 0.50
