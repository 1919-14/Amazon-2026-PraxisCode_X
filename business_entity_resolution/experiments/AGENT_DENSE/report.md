# Dense retrieval channel

## Implementation

`src/l3_l5_blocking/dense_retrieval.py` provides standalone Layer 2 loading,
E5 query/passage text construction, normalized embedding caches, exact FAISS
inner-product retrieval, the frozen three-column parquet writer, and the
requested recall/oracle/union metric calculations. Candidate embeddings are
added to FAISS source by source to avoid a second full candidate matrix copy.
The retrieval command emits every in-scope S1 entity when `--ref-sample` is
omitted. S2 and S3 vectors and their ID sidecars are cached separately under
`artifacts/embeddings/`.

## India train evaluation status

| Measure | Result |
| --- | ---: |
| Model | `intfloat/multilingual-e5-small` (configured; weights not available locally) |
| Embedding dimension | 384 (model specification; no local encoding completed) |
| Refs / candidates | Not run |
| Encode throughput | Not measured |
| Dense recall @ 5 / 10 / 20 / 50 / 100 / 200 | Not measured |
| Sparse + dense union recall @ 200 | Not measured |
| Dense oracle macro F0.5 @ 200 | Not measured |
| Runtime / peak RAM / peak VRAM | Not measured |

The requested experiment could not be completed in this environment. The local
Hugging Face cache contains `sentence-transformers/all-MiniLM-L6-v2`, but not
the requested multilingual E5 weights. The installed project interpreter
inherits a global `sentence-transformers 5.4.1` / `transformers 5.5.0` / Torch
2.5.1+cu121 environment whose optional `torchao` integration fails on import
(`torchao` expects `torch.int1`). CUDA itself is visible as an RTX 4050 with
`torch.cuda.is_available() == True`; FAISS CPU is installed. A Transformers
compatibility import probe did not complete, so there are no recall results to
report and no dense blocking parquet was produced. Existing sparse union recall
of 0.9223 remains the only measured value supplied for this workstream.

## Reproduce

From the repository root, after installing a mutually compatible
SentenceTransformers / Transformers / Torch stack and making the E5 model
weights available:

```powershell
venv/Scripts/python.exe business_entity_resolution/src/l3_l5_blocking/dense_retrieval.py `
  --split train --country india --ref-sample 100000 --seed 42 --k 200 `
  --batch-size 256 --device cuda --evaluate
```

The command intersects country-filtered Layer 2 S1 records with
`artifacts/splits/train_ids.json`, samples with the requested seed, encodes all
India S2/S3 records once into reusable caches, writes
`artifacts/blocking/dense_train_train_country=india.parquet`, reports dense
recall at the requested cutoffs, and computes union recall against the existing
`l4_train_train_country=india.parquet` channel. To build the full country
blocking output, omit `--ref-sample`; use `--split test --country france` for
France test retrieval (`--evaluate` applies to train only).

The parquet writer was checked against the required types:

- `source1_entity_id: string`
- `candidate_entity_ids: list<string>`
- `dense_scores: list<float32>`
