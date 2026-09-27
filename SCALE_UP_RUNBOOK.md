# Scale-Up Runbook — execute once, in this order

Companion to [`SCALE_UP_PLAN.md`](SCALE_UP_PLAN.md). All code for phases P0–P5 is
in place; this is the single execution pass. Run from the repo root with the
project interpreter (`venv/Scripts/python.exe` on Windows).

---

## 0. Prerequisites (install before the run)

| Need | Why | Install |
|---|---|---|
| `catboost`, `xgboost` | stack base learners (P2) | `python -m pip install catboost xgboost` |
| `indic-transliteration` | phonetic channel (P1b, Antigravity) | `python -m pip install indic-transliteration` |
| `sentence-transformers` + E5 weights | dense channel (P5, Codex) | fix the torchao import, then let the model download once |

The stack degrades gracefully: a missing `catboost`/`xgboost` is skipped and the
stack runs on the remaining base learners.

### Sanity (fast, ~2 min)
```bash
python business_entity_resolution/src/l1_validation/unit_tests.py
python business_entity_resolution/src/l3_l5_blocking/unit_tests.py
python business_entity_resolution/src/l6_l8_matching/unit_tests.py
python business_entity_resolution/src/l10_decision/unit_tests.py
python business_entity_resolution/src/l11_inference/unit_tests.py
```
Expected: L1 10 · L3–L5 23 · L6–L8 19 · L10 17 · L10.5 8 · L11 8 · utils 10.

---

## 1. Train the matcher (India pool)

```bash
cd business_entity_resolution/src

# 1a. Blocking: address text on, 200 candidates per channel (recall lever, P1a)
python main_l3.py --split train --refs train --ref-sample 400000 \
    --countries india --topk 200 --addr-text

# 1b. RRF fusion of A/C/D
python main_l4.py --split train --refs train --countries india

# 1c. Fuse the phonetic (G) + dense (H) channels into the union (P1b/P5)
python main_l4_extra.py --split train --refs train --countries india \
    --extra-template "artifacts/blocking/phon_{split}_{refs}_country={country}.parquet" \
    --extra-template "artifacts/blocking/dense_{split}_{refs}_country={country}.parquet"

# 1d. Truncate to K=200 (candidate_pairs.tsv is NOT scored) + write both sidecars
#     (lexical signals + extra phon/dense signals). --l4-template reads the fused
#     l4x artifact produced in 1c.
python main_l5.py --split train --refs train --countries india \
    --no-tune --ratio 0.0 --k-min 200 --k-max 200 \
    --l4-template "artifacts/blocking/l4x_{split}_{refs}_country={country}.parquet" \
    --extra-template "artifacts/blocking/phon_{split}_{refs}_country={country}.parquet" \
    --extra-template "artifacts/blocking/dense_{split}_{refs}_country={country}.parquet"

# 1e. Training pairs (hard negatives)
python main_l6.py --candidates output/candidate_pairs_train.tsv

# 1f. Features v2 (45) + signal sidecars
python main_l7.py --pairs artifacts/train_pairs/variant_a.parquet --signals-refs train

# 1g. GBDT stack + meta-learner (P2)
python main_l8.py --variant a --model-kind stack --bases lgbm catboost xgboost

# 1h. Calibration (kept only if it helps)
python main_l9.py --variant a

# 1i. Decision v2: tune per-country + entity-level expected-F0.5 (P3)
python main_l10.py --split train --variant a --decision-v2 --use-calibrated
```

Outputs: `artifacts/models/stack_a/` (fold base models + `stack_meta.json`),
`output/l10_decision_v2_policy.json` (recall_hat per country), OOF tables.

---

## 2. Test inference (submission)

```bash
cd business_entity_resolution/src

python main_l3.py --split test --refs all --topk 200 --addr-text
python main_l4.py --split test --refs all
python main_l4_extra.py --split test --refs all \
    --extra-template "artifacts/blocking/phon_{split}_{refs}_country={country}.parquet" \
    --extra-template "artifacts/blocking/dense_{split}_{refs}_country={country}.parquet"
python main_l5.py --split test --refs all --no-tune --ratio 0.0 --k-min 200 --k-max 200 \
    --l4-template "artifacts/blocking/l4x_{split}_{refs}_country={country}.parquet" \
    --extra-template "artifacts/blocking/phon_{split}_{refs}_country={country}.parquet" \
    --extra-template "artifacts/blocking/dense_{split}_{refs}_country={country}.parquet"

# Score with the stack + both sidecars, then decide with the tuned v2 policy
python main_l11.py --variant a --split test --model-kind stack \
    --signals-refs all --coverage-refs all
```

`main_l11.py` runs the official validator and writes `output/matching_results.tsv`.

---

## 3. Flags that matter

| Flag | Effect |
|---|---|
| `--topk 200` (L3) | per-channel candidate depth; raises union recall |
| `--addr-text` (L3) | sparse channels index `name_core + addr_norm` (cross-lingual bridge) |
| `--k-min 200 --k-max 200 --ratio 0.0 --no-tune` (L5) | keep 200 candidates/entity (not scored) |
| `--model-kind stack` (L8/L11) | LightGBM + CatBoost + XGBoost + logistic meta-learner |
| `--decision-v2` (L10) | entity-level expected-F0.5 with per-country `recall_hat` |
| `--extra` / `--extra-template` (L4x, L5) | fuse + record phonetic/dense channel scores |

---

## 3b. Memory-safe components (this pass)

| File | What it does |
|---|---|
| `src/run_scaled_pipeline.py` | **One-command driver** for L3→L11 with per-stage RAM guards. `--dry-run` prints commands; `--start/--stop` run a slice. |
| `src/l3_l5_blocking/dense_retrieval.py` | **Rewritten memory-bounded**: streams L2 shards, caches float16 embedding parts on disk, builds FAISS `IndexIVFPQ` for large pools (`IndexFlatIP` for small), streams query + output. |
| `src/main_phon.py` | **Memory-lean phonetic generator** (streams shards, no `pd.concat`, honours `--max-candidates`), row-aligned to the L3 reference order. |
| `src/l6_l8_matching/stack.py` | GBDT stack (LightGBM + CatBoost + XGBoost + logistic meta), grouped OOF, save/load. |
| `src/l6_l8_matching/extra_signals.py` | Phonetic/dense signal sidecar with train/serve lockstep parity. |
| `src/l10_decision/decision_v2.py` | Per-country + entity-level expected-F0.5 selection. |

Driver examples:
```bash
cd business_entity_resolution
# full train pass (tune matcher) for India
..\venv\Scripts\python.exe src\run_scaled_pipeline.py --mode train --countries india
# test pass -> submission
..\venv\Scripts\python.exe src\run_scaled_pipeline.py --mode test --countries us india france
```

Bounded dense proof run (fast, proves the channel lifts union recall):
```bash
..\venv\Scripts\python.exe src\l3_l5_blocking\dense_retrieval.py \
    --split train --country india --ref-sample 30000 --candidate-sample 200000 --evaluate
```

## 4. Compaction note

`candidate_pairs.tsv` is used for measurement only and is not scored, so K=200 is
free in *score*. If a smaller blocking set is needed for the compactness
tie-break, re-run only L5 with `--k-min 6 --k-max 12 --ratio 0.9` to emit a
compact `candidate_pairs.tsv`; the matcher input can stay the 200-candidate set.
