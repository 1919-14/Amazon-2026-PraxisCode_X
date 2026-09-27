"""Dense multilingual candidate retrieval (channel H) - memory-bounded rewrite.

The previous version loaded every candidate of a country into pandas and then
held *all* float32 embeddings plus a float32 FAISS index in RAM. For India that
is ~9M x 384 x 4 bytes = ~13.8 GB for the embeddings alone, on a 16 GB box - it
cannot run. This rewrite streams instead:

1. L2 parquet shards are scanned in blocks (``pyarrow.dataset`` + filter), never
   materialising a whole country.
2. Each block is encoded with ``intfloat/multilingual-e5-small`` (fp16 on CUDA)
   and appended as a **float16** ``.npy`` part on disk, with a matching ids part.
3. A manifest pins part order so cache reuse is verifiable.
4. The index is a FAISS ``IndexIVFPQ`` (compressed) whenever the pool is large,
   falling back to exact ``IndexFlatIP`` for small pools. Source blocks are cast
   just-in-time and freed, so the index, not the corpus, bounds memory.
5. Query (Source-1) vectors are searched in blocks and results are written
   streaming, so a full test run never holds all S1 embeddings either.

Model: ``intfloat/multilingual-e5-small`` (MIT, 118M) with the e5 prefixes
(``query:`` for S1, ``passage:`` for S2/S3) and text = ``name_core | addr_norm``.
No external lookups; only model weights are fetched.

Frozen output schema (one row per Source-1 entity)::

    source1_entity_id   : string
    candidate_entity_ids: list<string>   (ranked best-first)
    dense_scores        : list<float32>

Run (bounded proof run, ~minutes)::

    venv/Scripts/python.exe business_entity_resolution/src/l3_l5_blocking/dense_retrieval.py \
      --split train --country india --ref-sample 30000 --candidate-sample 200000 --evaluate

Run (full country, memory-bounded but slow)::

    venv/Scripts/python.exe business_entity_resolution/src/l3_l5_blocking/dense_retrieval.py \
      --split test --country france --k 200
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT / "business_entity_resolution" / "artifacts"
NORMALIZED = ARTIFACTS / "normalized"
EMBEDDINGS = ARTIFACTS / "embeddings"
BLOCKING = ARTIFACTS / "blocking"
MODEL_NAME = "intfloat/multilingual-e5-small"
TOP_K = 200

# Rows per encoded block. Bounds encode memory: batch*384*4 bytes (float32) and
# the matching float16 part on disk. 25k keeps peak well under a few hundred MB.
PART_ROWS = 25_000
# Pools at or below this use the exact float32 IndexFlatIP; larger pools use
# IndexIVFPQ because a flat index would need #pools * dim * 4 bytes of RAM.
FLAT_LIMIT = 1_500_000


# ---------------------------------------------------------------------------
# Source streaming
# ---------------------------------------------------------------------------

def _dataset(split: str, source: str) -> ds.Dataset:
    path = NORMALIZED / f"{split}_{source}"
    if not path.is_dir():
        raise FileNotFoundError(path)
    return ds.dataset(str(path), format="parquet")


def iter_source_blocks(
    split: str,
    source: str,
    country: str,
    block_size: int = PART_ROWS,
    max_rows: int | None = None,
    columns: tuple[str, ...] = ("entity_id", "name_core", "addr_norm"),
) -> Iterator["object"]:
    """Yield pandas blocks for one split/source/country without loading it all."""
    dataset = _dataset(split, source)
    filt = ds.field("country_norm") == country.lower()
    scanner = dataset.scanner(columns=list(columns), filter=filt, batch_size=block_size)
    seen = 0
    buffer: list = []
    for batch in scanner.to_batches():
        if max_rows is not None and seen >= max_rows:
            break
        buffer.append(batch)
        seen += batch.num_rows
        if sum(b.num_rows for b in buffer) >= block_size:
            table = pa.Table.from_batches(buffer).combine_chunks()
            frame = table.to_pandas().drop_duplicates("entity_id", keep="first")
            yield frame
            buffer = []
    if buffer:
        table = pa.Table.from_batches(buffer).combine_chunks()
        yield table.to_pandas().drop_duplicates("entity_id", keep="first")


def make_text(frame, source: str) -> list[str]:
    """e5 text: prefixed ``name_core | addr_norm`` (address is the cross-lingual bridge)."""
    prefix = "query" if source == "s1" else "passage"
    names = frame["name_core"].fillna("").astype(str).str.strip()
    addrs = frame["addr_norm"].fillna("").astype(str).str.strip()
    return [f"{prefix}: {n} | {a}" for n, a in zip(names, addrs)]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model(model_name: str = MODEL_NAME, device: str = "cuda"):
    """Load the e5 sentence transformer, hiding an incompatible optional torchao."""
    import importlib.util

    original = importlib.util.find_spec
    importlib.util.find_spec = lambda name, *a, **kw: (
        None if name == "torchao" else original(name, *a, **kw)
    )
    try:
        from sentence_transformers import SentenceTransformer
    finally:
        importlib.util.find_spec = original

    import torch

    model = SentenceTransformer(
        model_name,
        device=device,
        model_kwargs={"torch_dtype": torch.float16 if str(device).startswith("cuda") else torch.float32},
    )
    model.max_seq_length = 256
    return model


def _encode(model, texts: list[str], batch_size: int) -> np.ndarray:
    values = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(values, dtype=np.float16)


# ---------------------------------------------------------------------------
# Float16 part cache
# ---------------------------------------------------------------------------

def _part_paths(split: str, source: str, country: str, index: int) -> tuple[Path, Path, Path]:
    EMBEDDINGS.mkdir(parents=True, exist_ok=True)
    stem = f"dense_{split}_{source}_country={country.lower()}"
    return (
        EMBEDDINGS / f"{stem}_part{index:04d}.npy",
        EMBEDDINGS / f"{stem}_part{index:04d}.ids.npy",
        EMBEDDINGS / f"{stem}.manifest.json",
    )


def encode_source(
    split: str,
    source: str,
    country: str,
    model,
    *,
    block_size: int = PART_ROWS,
    batch_size: int = 256,
    max_rows: int | None = None,
    force: bool = False,
    progress=print,
) -> dict:
    """Encode one split/source/country into float16 part files; return the manifest.

    Reuses an existing manifest when its recorded scope matches, so a rerun never
    recomputes embeddings.
    """
    _, _, manifest_path = _part_paths(split, source, country, 0)
    if manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("scope") == {"split": split, "source": source, "country": country.lower()}:
            progress(f"  cache hit: {manifest_path.name} ({manifest['total']:,} rows, {len(manifest['parts'])} parts)")
            return manifest

    parts: list[dict] = []
    total = 0
    started = time.perf_counter()
    for index, block in enumerate(iter_source_blocks(split, source, country, block_size, max_rows)):
        if block.empty:
            continue
        part_path, ids_path, _ = _part_paths(split, source, country, index)
        if part_path.exists() and not force:
            ids = np.load(ids_path, allow_pickle=False).astype(str).tolist()
        else:
            vectors = _encode(model, make_text(block, source), batch_size)
            ids = block["entity_id"].astype(str).tolist()
            if len(ids) != vectors.shape[0]:
                raise ValueError(f"id/vector row mismatch: {len(ids)} vs {vectors.shape[0]}")
            np.save(part_path, vectors)
            np.save(ids_path, np.asarray(ids, dtype=str))
        parts.append({"index": index, "path": str(part_path), "ids": str(ids_path), "rows": len(ids)})
        total += len(ids)
        progress(f"    part {index:04d} | {len(ids):>7,} rows | total {total:>9,}")

    manifest = {
        "scope": {"split": split, "source": source, "country": country.lower()},
        "model": MODEL_NAME,
        "dim": 384,
        "dtype": "float16",
        "total": total,
        "parts": parts,
        "encode_seconds": round(time.perf_counter() - started, 1),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_ids(manifest: dict) -> list[str]:
    """Load the candidate ids in part order (global position -> entity id)."""
    ids: list[str] = []
    for part in manifest["parts"]:
        ids.extend(np.load(part["ids"], allow_pickle=False).astype(str).tolist())
    return ids


def iter_part_vectors(manifest: dict, as_float32: bool = True) -> Iterator[np.ndarray]:
    """Yield each cached part's vectors, cast just in time and mmap-backed."""
    for part in manifest["parts"]:
        block = np.load(part["path"], mmap_mode="r")
        yield np.ascontiguousarray(block, dtype=np.float32) if as_float32 else block


# ---------------------------------------------------------------------------
# FAISS index (memory-bounded)
# ---------------------------------------------------------------------------

def build_index(manifest: dict, flat_limit: int = FLAT_LIMIT, nlist: int = 4096,
                pq_m: int = 48, nprobe: int = 32, progress=print):
    """Build an exact flat index for small pools or a compressed IVF-PQ index."""
    import faiss

    dim = int(manifest["dim"])
    total = int(manifest["total"])
    if total == 0:
        raise ValueError("empty manifest - nothing to index")

    if total <= flat_limit:
        index = faiss.IndexFlatIP(dim)
        for block in iter_part_vectors(manifest):
            index.add(block)
        progress(f"  index: IndexFlatIP | {index.ntotal:,} vectors (exact)")
        return index

    # IVF-PQ: train on a bounded sample, then add every part in turn.
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, pq_m, 8)
    train_cap = max(nlist * 64, 200_000)
    sample = []
    have = 0
    for block in iter_part_vectors(manifest):
        sample.append(block)
        have += block.shape[0]
        if have >= train_cap:
            break
    training = np.ascontiguousarray(np.concatenate(sample, axis=0)[:train_cap])
    del sample
    index.train(training)
    del training
    for block in iter_part_vectors(manifest):
        index.add(block)
    index.nprobe = nprobe
    progress(f"  index: IndexIVFPQ(nlist={nlist}, m={pq_m}) | {index.ntotal:,} vectors (compressed)")
    return index


def retrieve_streaming(
    model,
    index,
    candidate_ids: list[str],
    split: str,
    country: str,
    *,
    k: int = TOP_K,
    ref_block: int = PART_ROWS,
    query_batch: int = 512,
    max_refs: int | None = None,
    out_path: Path,
    progress=print,
) -> int:
    """Encode Source-1 in blocks, search, and stream the frozen parquet output."""
    import faiss

    out_path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([
        ("source1_entity_id", pa.string()),
        ("candidate_entity_ids", pa.list_(pa.string())),
        ("dense_scores", pa.list_(pa.float32())),
    ])
    take = min(k, len(candidate_ids)) if candidate_ids else 0
    written = 0
    with pq.ParquetWriter(str(out_path), schema, compression="zstd") as writer:
        for block in iter_source_blocks(split, "s1", country, ref_block, max_refs):
            if block.empty or take == 0:
                continue
            ref_ids = block["entity_id"].astype(str).tolist()
            vectors = _encode(model, make_text(block, "s1"), query_batch)
            vectors = np.ascontiguousarray(vectors, dtype=np.float32)
            scores, positions = index.search(vectors, take)
            rows_ids: list[list[str]] = []
            rows_scores: list[list[float]] = []
            for pos_row, score_row in zip(positions, scores):
                valid = pos_row >= 0
                rows_ids.append([candidate_ids[i] for i in pos_row[valid]])
                rows_scores.append([float(x) for x in score_row[valid]])
            writer.write_table(pa.table(
                {
                    "source1_entity_id": ref_ids,
                    "candidate_entity_ids": rows_ids,
                    "dense_scores": rows_scores,
                },
                schema=schema,
            ))
            written += len(ref_ids)
            if written % (ref_block * 20) < ref_block:
                progress(f"    retrieved {written:,} references")
    return written


# ---------------------------------------------------------------------------
# Evaluation helpers (train split, bounded)
# ---------------------------------------------------------------------------

def load_ground_truth(path: Path) -> dict[str, set[str]]:
    truth: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        header = stream.readline().rstrip("\r\n").split("\t")
        s1_col = header.index("s1_id") if "s1_id" in header else 0
        match_col = next((i for i, x in enumerate(header) if x.lower() not in {"s1_id", "source1_entity_id"}), 1)
        for line in stream:
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) <= max(s1_col, match_col):
                continue
            truth[fields[s1_col]] = {x for x in fields[match_col].split(",") if x}
    return truth


def f_beta(pred: set[str], true: set[str], b: float = 0.5) -> float:
    if not true:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    return (1 + b * b) * p * r / (b * b * p + r)


def evaluate(out_path: Path, truth: dict[str, set[str]], ks: Iterable[int]) -> None:
    """Recall@k and oracle macro F0.5 for the dense channel, plus union with L4."""
    ks = list(ks)
    hits = {k: 0 for k in ks}
    denominator = 0
    oracle = 0.0
    n_refs = 0
    table = pq.read_table(out_path, columns=["source1_entity_id", "candidate_entity_ids"]).to_pydict()
    sparse_path = BLOCKING / "l4_train_train_country=india.parquet"
    sparse_map: dict[str, list[str]] = {}
    if sparse_path.exists():
        s = pq.read_table(sparse_path, columns=["source1_entity_id", "candidate_entity_ids"]).to_pydict()
        sparse_map = dict(zip(s["source1_entity_id"], s["candidate_entity_ids"]))
    union_hit = 0
    for sid, cands in zip(table["source1_entity_id"], table["candidate_entity_ids"]):
        cands = [] if cands is None else cands
        true = truth.get(sid, set())
        n_refs += 1
        if true:
            denominator += len(true)
            for k in ks:
                hits[k] += len(true & set(cands[:k]))
        oracle += f_beta(set(cands) & true, true)
        if true and sparse_map:
            union_hit += len(true & (set(sparse_map.get(sid, [])) | set(cands)))
    print("Dense recall@k: " + ", ".join(f"{k}={hits[k]/max(1,denominator):.4f}" for k in ks))
    print(f"Dense oracle macro F0.5={oracle/max(1,n_refs):.4f} | non-singleton refs={n_refs:,}")
    if sparse_map and denominator:
        print(f"Sparse+dense union recall@{max(ks)}={union_hit/denominator:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Memory-bounded dense multilingual channel")
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--country", required=True)
    parser.add_argument("--k", type=int, default=TOP_K)
    parser.add_argument("--batch-size", type=int, default=256, help="Encode batch.")
    parser.add_argument("--part-rows", type=int, default=PART_ROWS, help="Rows per encoded block.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--force-encode", action="store_true")
    # Bounded proof runs: cap candidate rows (deterministic head) and refs.
    parser.add_argument("--candidate-sample", type=int, default=None)
    parser.add_argument("--ref-sample", type=int, default=None, help="Cap Source-1 refs (head order).")
    parser.add_argument("--flat-limit", type=int, default=FLAT_LIMIT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--evaluate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    country = args.country.lower()
    started = time.perf_counter()
    print(f"=== dense channel: split={args.split} country={country} ===")

    out_path = args.output or (BLOCKING / f"dense_{args.split}_{args.split}_country={country}.parquet")
    if out_path.exists() and not args.force_encode:
        print(f"  [dense] output already exists at {out_path}, skipping encoding!")
        return

    model = load_model(args.model, args.device)
    manifests = {}
    for source in ("s2", "s3"):
        print(f"encoding {source} ...")
        manifests[source] = encode_source(
            args.split, source, country, model,
            block_size=args.part_rows, batch_size=args.batch_size,
            max_rows=args.candidate_sample, force=args.force_encode,
        )

    # Concatenate candidates in s2-then-s3 order, keeping global positions aligned.
    merged_parts = []
    candidate_ids: list[str] = []
    offset = 0
    for source in ("s2", "s3"):
        for part in manifests[source]["parts"]:
            merged_parts.append({"path": part["path"], "rows": part["rows"], "offset": offset})
            offset += part["rows"]
        candidate_ids.extend(load_ids(manifests[source]))
    merged = {"dim": manifests["s2"]["dim"], "total": offset, "parts": merged_parts}
    print(f"candidates: {offset:,} across {len(merged_parts)} parts")

    index = build_index(merged, flat_limit=args.flat_limit)

    out_path = args.output or (BLOCKING / f"dense_{args.split}_{args.split}_country={country}.parquet")
    written = retrieve_streaming(
        model, index, candidate_ids, args.split, country,
        k=args.k, ref_block=args.part_rows, max_refs=args.ref_sample, out_path=out_path,
    )
    print(f"output: {out_path} | rows={written:,} | runtime={time.perf_counter()-started:.1f}s")

    # Clean up temporary cached .npy vectors to save Kaggle output disk quota
    for source in ("s2", "s3"):
        for part in manifests.get(source, {}).get("parts", []):
            try:
                Path(part["path"]).unlink(missing_ok=True)
                Path(part["ids"]).unlink(missing_ok=True)
            except Exception:
                pass
    print("  [dense] cleaned up temporary intermediate vector .npy files")

    if args.evaluate:
        if args.split != "train":
            raise ValueError("evaluation is train-only")
        truth = load_ground_truth(ROOT / "DATA SET" / "student_resource" / "dataset" / "train" / "train_ground_truth.tsv")
        evaluate(out_path, truth, [5, 10, 20, 50, 100, 200])


if __name__ == "__main__":
    main()
