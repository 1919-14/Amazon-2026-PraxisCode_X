"""L6: training pair construction (positive / hard / easy negatives).

For every Source 1 reference entity we build supervised pairs from three
sources:

* **Positive**  - a candidate that is in the ground truth (label 1).
* **Hard negative** - a *retrieved* candidate that is NOT in the ground truth.
  These are the confusable decoys blocking ranked highly, so they teach the
  matcher where the decision boundary actually lies.
* **Easy negative** - a random same-country Source 2/3 record that was neither
  retrieved nor a true match; cheap background signal.

Two datasets are produced for the A/B test (executed in L8, which has features
and a model):
  * Variant A - positives + hard negatives + easy negatives
  * Variant B - positives + easy negatives (hard negatives removed)

Sampling ratio per reference: 1 positive : ``pos_hard`` hard : ``pos_easy`` easy.
Reference entities with no true matches (singletons) still receive a small fixed
number of negatives so the model can learn to predict the empty list.
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

import pandas as pd

# Ensure src/ is importable when this module is used directly.
_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import (
    L6_EASY_POOL_SIZE,
    L6_POS_EASY_RATIO,
    L6_POS_HARD_RATIO,
    L6_SINGLETON_EASY,
    L6_SINGLETON_HARD,
)
from l3_l5_blocking.buckets import aliases_for, iter_source_shards

POSITIVE = "pos"
HARD = "hard"
EASY = "easy"


@dataclass(slots=True)
class PairSample:
    """One supervised training pair."""

    s1_id: str
    cand_id: str
    label: int
    neg_type: str
    country: str
    rank: int = -1  # position in the candidate list (-1 for easy negatives)


@dataclass(slots=True)
class SamplingConfig:
    """Sampling ratios (defaults come from ``config.py``)."""

    pos_hard: int = L6_POS_HARD_RATIO
    pos_easy: int = L6_POS_EASY_RATIO
    singleton_hard: int = L6_SINGLETON_HARD
    singleton_easy: int = L6_SINGLETON_EASY


def iter_candidate_pairs(path: str | Path) -> Iterator[tuple[str, list[str]]]:
    """Yield ``(s1_id, candidate_ids)`` rows from an L5 candidate-pairs TSV.

    Handles the header row and empty candidate fields (singletons).
    """
    with open(path, "r", encoding="utf-8") as handle:
        first = handle.readline().rstrip("\n")
        parts = first.split("\t")
        if parts and parts[0] != "source1_entity_id":
            # No header: treat the line as data.
            yield parts[0], _split_ids(parts[1] if len(parts) > 1 else "")
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if not parts or not parts[0]:
                continue
            yield parts[0], _split_ids(parts[1] if len(parts) > 1 else "")


def _split_ids(raw: str) -> list[str]:
    return [token for token in raw.split(",") if token]


def build_easy_pool(
    split: str,
    country: str,
    size: int = L6_EASY_POOL_SIZE,
    seed: int = 42,
) -> list[str]:
    """Reservoir-sample ``size`` same-country Source 2/3 ids for easy negatives."""
    rng = random.Random(seed)
    aliases = aliases_for(country)
    pool: list[str] = []
    seen = 0

    for source_idx in (2, 3):
        for shard in iter_source_shards(split, source_idx):
            df = pd.read_parquet(shard, columns=["entity_id", "country_norm"])
            df = df[df["country_norm"].isin(aliases)]
            for entity_id in df["entity_id"].tolist():
                seen += 1
                if len(pool) < size:
                    pool.append(entity_id)
                else:
                    j = rng.randint(0, seen - 1)
                    if j < size:
                        pool[j] = entity_id

    return pool


def _draw_easy(
    pool: Sequence[str],
    exclude: set[str],
    n: int,
    rng: random.Random,
) -> list[str]:
    """Draw up to ``n`` pool ids not in ``exclude`` (random, no duplicates)."""
    if n <= 0 or not pool:
        return []
    out: list[str] = []
    chosen: set[str] = set()
    attempts = 0
    max_attempts = n * 20 + 50
    while len(out) < n and attempts < max_attempts:
        candidate = pool[rng.randrange(len(pool))]
        attempts += 1
        if candidate in exclude or candidate in chosen:
            continue
        chosen.add(candidate)
        out.append(candidate)
    return out


def sample_reference_pairs(
    s1_id: str,
    candidate_ids: Sequence[str],
    ground_truth: set[str],
    country: str,
    easy_pool: Sequence[str],
    rng: random.Random,
    config: Optional[SamplingConfig] = None,
) -> list[PairSample]:
    """Build the positive / hard / easy pairs for one reference entity.

    The candidate list is assumed ranked best-first, so the highest-ranked
    non-matches are the hardest negatives.
    """
    config = config or SamplingConfig()

    # Deduplicate while preserving the ranking order.
    candidates = list(dict.fromkeys(candidate_ids))
    rank = {candidate: i for i, candidate in enumerate(candidates)}

    positives = [c for c in candidates if c in ground_truth]
    hard = [c for c in candidates if c not in ground_truth]

    n_pos = len(positives)
    if n_pos > 0:
        n_hard = min(len(hard), config.pos_hard * n_pos)
        n_easy = config.pos_easy * n_pos
    else:
        n_hard = min(len(hard), config.singleton_hard)
        n_easy = config.singleton_easy

    hard_selected = hard[:n_hard]

    # Easy negatives must not be candidates or true matches for this reference.
    exclude = set(candidates) | ground_truth
    easy_selected = _draw_easy(easy_pool, exclude, n_easy, rng)

    samples: list[PairSample] = []
    for candidate in positives:
        samples.append(PairSample(s1_id, candidate, 1, POSITIVE, country, rank[candidate]))
    for candidate in hard_selected:
        samples.append(PairSample(s1_id, candidate, 0, HARD, country, rank[candidate]))
    for candidate in easy_selected:
        samples.append(PairSample(s1_id, candidate, 0, EASY, country, -1))
    return samples


def variant_b_rows(samples: Iterable[PairSample]) -> list[PairSample]:
    """Variant B: drop hard negatives, keep positives and easy negatives."""
    return [sample for sample in samples if sample.neg_type != HARD]


def summarize(samples: Sequence[PairSample]) -> dict:
    """Count pairs by label and negative type."""
    positives = sum(1 for s in samples if s.neg_type == POSITIVE)
    hard = sum(1 for s in samples if s.neg_type == HARD)
    easy = sum(1 for s in samples if s.neg_type == EASY)
    references = len({s.s1_id for s in samples})
    return {
        "pairs": len(samples),
        "positives": positives,
        "hard_negatives": hard,
        "easy_negatives": easy,
        "references": references,
        "pos_hard_ratio": (hard / positives) if positives else 0.0,
        "pos_easy_ratio": (easy / positives) if positives else 0.0,
    }
