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

# Include normalized address text in the sparse retrieval channels (C/D).
# Name-only retrieval is a hard recall ceiling: two records can describe the same
# business while the names differ (abbreviations, transliteration, word order),
# and the address is then the only shared evidence that can retrieve the pair.
# When enabled the channels index/query ``name_core + " " + addr_norm``. Kept as
# an explicit switch so the two retrieval texts can be measured against each
# other (L3 union recall) before the change is adopted.
L3_RETRIEVAL_INCLUDE_ADDRESS: bool = False
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
# Layer 3 memory budget: block-wise, disk-backed candidate index
# ---------------------------------------------------------------------------
# The naive design held one TF-IDF CSC matrix for an entire country bucket plus
# every candidate list for every reference. For the train buckets (6.2M / 4.1M
# candidates) that is multiple GB of matrix plus a Python list of ~10^8 id
# strings - far past the 2 GB budget. The block-wise engine instead keeps one
# candidate block index resident at a time and streams references past it.
L3_BLOCK_SIZE: int = 500_000
L3_REF_BLOCK_SIZE: int = 250_000
# Documents sampled to fit the shared vocabulary + IDF (one sample per channel,
# so every block scores in the same space and scores stay comparable).
L3_VOCAB_SAMPLE: int = 300_000
# Keep the scratch index after a run (debugging); otherwise it is removed.
L3_KEEP_SCRATCH: bool = False
# Warn when peak RSS exceeds this budget (MB).
L3_MEMORY_BUDGET_MB: int = 1800

# ---------------------------------------------------------------------------
# Layer 4: Reciprocal Rank Fusion
# ---------------------------------------------------------------------------
RRF_K_GRID: list[int] = [10, 20, 40, 60]

# References processed per streaming batch in L4/L5. A full country bucket cannot
# be held in Python objects (US test ~663k refs x ~370 candidates x 3 channels is
# >10 GB of id strings), so both layers stream the upstream artifact in batches of
# this many references and keep only the batch in flight resident.
L4_REF_BATCH: int = 10_000

# ---------------------------------------------------------------------------
# Layer 5: Adaptive Candidate Truncation
# ---------------------------------------------------------------------------
# Target compactness band from the challenge brief (candidates per S1 entity).
L5_K_TARGET: float = 6.25
L5_K_MIN_TARGET: float = 5.5
L5_K_MAX_TARGET: float = 7.0

# Hard per-reference bounds applied after the adaptive score threshold.
#
# k_min is a rank floor, not just a safety net: the relative score threshold alone
# keeps almost nothing when a reference has one very strong top hit (an exact
# name+postal key gives score 1.0, so a 0.7 cut drops every other candidate even
# though the entity has several true matches). Measured on the India pool that
# cost 30 recall points (46.6% at k_min=1 vs 76.4% at k_min=6, same 0.9 ratio),
# because this is a multi-match problem, not a top-1 problem.
L5_K_MIN: int = 6
L5_K_MAX: int = 12

# Default fraction of the top coarse score a candidate must retain to survive.
# Frozen at the value L5 tuning selects on the training pool; used verbatim on the
# test split, where tuning is impossible (no ground truth).
L5_COARSE_RATIO: float = 0.9
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
