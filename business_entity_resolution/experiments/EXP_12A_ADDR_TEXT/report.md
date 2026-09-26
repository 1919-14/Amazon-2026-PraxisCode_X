# EXP-12A — Address Text in Sparse Retrieval (L3) + Rank-Floor Truncation (L5)

Status: **KEEP** · Gate 2 (candidate recall ≥ 90% at the union, and the best
achievable final candidate recall) · India training pool, 100k reference sample,
`topk=200`, `seed=42`.

## 1. What was wrong

The L3 sparse channels (C = word TF-IDF, D = char 2-4 gram TF-IDF) were built and
queried on **`name_core` only**. Normalized address text was never used for
candidate generation — only channel A's exact `name|postal` / `name|house` hashes
touched the address. Two records describing the same business with different
names (abbreviations, transliteration, word order) could therefore never be
retrieved, which is a hard ceiling on recall that no downstream model can undo.

A second, independent flaw sat in L5: `adaptive_truncate` kept candidates by a
threshold relative to the reference's top coarse score. A reference whose best
candidate is an exact `name|postal` hit scores 1.0, so a 0.7 cut keeps **only that
one candidate** — discarding the entity's other true matches. This is a
multi-match problem, so that policy is wrong.

> An earlier `output/l4_*`/`l5_*` reading of "3–4% recall" was a **stale artifact
> mismatch** (an old 60k-reference L3 artifact scored against a differently
> sampled ground-truth map). The live RRF ranking is healthy (mean first-hit rank
> 2.4); that number should not be used.

## 2. What changed

| Change | File | Effect |
|---|---|---|
| `--addr-text` / `L3_RETRIEVAL_INCLUDE_ADDRESS` | `config.py`, `main_l3.py`, `l3_l5_blocking/blocked.py`, `buckets.py` | Channels C/D index and query `name_core + " " + addr_norm`; channel A stays name-only. |
| `L5_K_MIN = 6` (rank floor) | `config.py` | Every reference keeps at least its top-6 coarse-ranked candidates. |
| `L5_COARSE_RATIO = 0.9` | `config.py` | Frozen ratio for the test split (tuning is impossible without ground truth). |

`addr_norm` was added to `l3_l5_blocking.buckets.CANDIDATE_COLUMNS`.

## 3. Measured results (India train, 100k refs, topk=200)

| Channel | name-only | + addr text |
|---|---|---|
| A (exact key) | 31.70% | **31.70%** (unchanged by design) |
| C (word TF-IDF) | 47.72% | **87.25%** |
| D (char TF-IDF) | 50.23% | **84.56%** |
| **UNION** | **63.74%** | **92.23%** |
| any-hit | 85.62% | 99.06% |
| avg candidates | 286.4 | 367.0 |

Channel A is bit-identical across the two runs, which is the invariant that the
new unit test pins.

RRF ranking (L4, k=60): recall@5 = 72.40%, @10 = 80.75%, @20 = 84.21%, @50 = 87.82%.

L5 truncation (final candidate set fed to the matcher):

| Policy | avg K | candidate recall |
|---|---|---|
| ratio 0.7, k_min 1 (old) | 4.37 | 46.64% |
| **ratio 0.9, k_min 6** | **6.29** | **76.36%** |

avg K 6.29 is inside the challenge compactness band [5.5, 7.0].

## 4. Cost

- L3 India 100k refs: 963.8 s → 1819.6 s (~1.9×; longer text means more tokens/grams).
- Peak RSS 2894 MB (still under the 3 GB budget, but above the configured 1800 MB warn line).
- L4 fusion: 130.9 s. L5 truncation: 312.0 s.

## 5. Honest limitations

- **The final candidate recall is 76.4%, not ≥90%.** Raising the union recall to
  92.2% does not translate into 90% *after* truncation to K≈6: recall@6 of the
  fused ranking is 76.0%. Closing that gap needs better **ranking**, not more
  candidates.
- Measured on the India pool only. The US pool (where the earlier baseline ran
  with the char channel **disabled**) still needs the same A/B.
- L4/L5 load whole artifacts into Python lists and are not memory-bounded like
  L3; with the richer candidate sets this is the next scaling risk.
- No F0.5 is reported here: this is the candidate-generation gate, upstream of
  the matcher.

## 6. Reproduce

```bash
cd business_entity_resolution/src
python main_l3.py --split train --refs train --countries india \
  --ref-sample 100000 --ref-seed 42 --topk 200 \
  --block-size 2000000 --ref-block-size 125000 --addr-text
python main_l4.py --split train --refs train --countries india --k-grid 60
python main_l5.py --split train --refs train --countries india
python l3_l5_blocking/unit_tests.py   # 23/23
```
