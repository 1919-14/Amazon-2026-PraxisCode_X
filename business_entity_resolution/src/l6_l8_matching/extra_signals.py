"""Extra-channel signal sidecar (phonetic / dense) shared by L7 and L11.

The lexical RRF sidecar (``signals.py``) carries ``rrf_score`` and
``channel_agreement``. The phonetic (channel G) and dense (channel H) retrievers
add two more per-pair signals the matcher can use: their own scores. This module
writes those next to the candidate set and reads them back in lockstep, using the
same contract as the lexical sidecar:

* :func:`write_extra_signals` - streaming writer used by Layer 5.
* :func:`load_extra_signals_for_pairs` - keyed lookup used by Layer 7.
* :class:`ExtraSignalStream` - lockstep reader used by Layer 11.

The file is optional. When it is absent both training and inference see zeros, so
the two stay in parity by construction (never one-with, one-without).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, Mapping, Optional, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from config import PATH_ARTIFACTS_DIR

EXTRA_COLUMNS = ["s1_id", "cand_id", "phon_score", "dense_score"]
EXTRA_SCHEMA = pa.schema(
    [
        ("s1_id", pa.string()),
        ("cand_id", pa.string()),
        ("phon_score", pa.float32()),
        ("dense_score", pa.float32()),
    ]
)


def extra_signals_dir() -> Path:
    """Directory holding the extra-channel sidecars."""
    return PATH_ARTIFACTS_DIR / "signals"


def extra_signals_path(split: str, refs: str) -> Path:
    """Canonical extra-signal path for a candidate set."""
    return extra_signals_dir() / f"extra_{split}_{refs}.parquet"


def extra_signals_available(split: str, refs: str) -> bool:
    """True when the extra-channel sidecar for this candidate set exists."""
    return extra_signals_path(split, refs).exists()


class ExtraSignalWriter:
    """Streaming writer for the extra-channel sidecar."""

    def __init__(self, path: str | Path, chunk: int = 500_000) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.chunk = chunk
        self._writer = pq.ParquetWriter(str(self.path), EXTRA_SCHEMA)
        self._buffer: dict[str, list] = {name: [] for name in EXTRA_COLUMNS}
        self.rows = 0

    def add(
        self,
        s1_id: str,
        cand_ids: Sequence[str],
        phon_scores: Sequence[float] | None = None,
        dense_scores: Sequence[float] | None = None,
    ) -> None:
        """Append one reference's candidate extra-channel scores, in candidate order."""
        n = len(cand_ids)
        phon = list(phon_scores) if phon_scores is not None else [0.0] * n
        dense = list(dense_scores) if dense_scores is not None else [0.0] * n
        for i, cand_id in enumerate(cand_ids):
            self._buffer["s1_id"].append(s1_id)
            self._buffer["cand_id"].append(cand_id)
            self._buffer["phon_score"].append(float(phon[i]) if i < len(phon) else 0.0)
            self._buffer["dense_score"].append(float(dense[i]) if i < len(dense) else 0.0)
            self.rows += 1
        if len(self._buffer["s1_id"]) >= self.chunk:
            self.flush()

    def flush(self) -> None:
        """Write the buffered rows."""
        if not self._buffer["s1_id"]:
            return
        self._writer.write_table(pa.table(self._buffer, schema=EXTRA_SCHEMA))
        for values in self._buffer.values():
            values.clear()

    def close(self) -> None:
        """Flush and close the sidecar."""
        self.flush()
        self._writer.close()


def iter_extra_rows(path: str | Path, batch_size: int = 100_000) -> Iterator[dict]:
    """Stream the sidecar as ``{column: value}`` dicts, one per row."""
    parquet_file = pq.ParquetFile(str(path))
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=EXTRA_COLUMNS):
        data = batch.to_pydict()
        for i in range(batch.num_rows):
            yield {name: data[name][i] for name in EXTRA_COLUMNS}


def load_extra_signals_for_pairs(
    path: str | Path,
    needed: Optional[Iterable[tuple[str, str]]] = None,
    max_entries: int = 5_000_000,
) -> dict[tuple[str, str], dict[str, float]]:
    """Load ``(s1_id, cand_id) -> {phon_score, dense_score}`` for the pairs."""
    extra_path = Path(path)
    if not extra_path.exists():
        return {}

    wanted = set(needed) if needed is not None else None
    signals: dict[tuple[str, str], dict[str, float]] = {}
    for row in iter_extra_rows(extra_path):
        key = (row["s1_id"], row["cand_id"])
        if wanted is not None and key not in wanted:
            continue
        signals[key] = {
            "phon_score": float(row["phon_score"] or 0.0),
            "dense_score": float(row["dense_score"] or 0.0),
        }
        if len(signals) > max_entries:
            raise RuntimeError(
                f"extra signal join needs more than {max_entries:,} pairs "
                f"({extra_path.name}). Sample the training references first."
            )
    return signals


class ExtraSignalStream:
    """Lockstep reader: one extra-sidecar row per ``candidate_pairs.tsv`` row."""

    def __init__(self, path: str | Path | None, batch_size: int = 100_000) -> None:
        self.path = Path(path) if path is not None else None
        self.enabled = self.path is not None and self.path.exists()
        self._batch_size = batch_size
        self._iterator: Iterator[dict] | None = None
        self._pending: list[dict] = []
        self.rows_read = 0

    def _ensure_iterator(self) -> None:
        if self._iterator is None:
            self._iterator = iter_extra_rows(self.path, self._batch_size)

    def next_row(self, s1_id: str, cand_ids: Sequence[str]) -> list[dict[str, float]]:
        """Return per-candidate extra signals aligned with ``cand_ids``.

        Falls back to zeros when disabled. Raises ``ValueError`` on drift so a
        mismatched sidecar fails loudly instead of attaching wrong signals.
        """
        if not self.enabled:
            return [{"phon_score": 0.0, "dense_score": 0.0} for _ in cand_ids]

        self._ensure_iterator()
        out: list[dict[str, float]] = []
        for cand_id in cand_ids:
            while not self._pending:
                assert self._iterator is not None
                try:
                    self._pending.append(next(self._iterator))
                except StopIteration:
                    raise ValueError(
                        f"extra signal sidecar {self.path} ended early at row "
                        f"{self.rows_read}; it does not match this candidate file."
                    ) from None
            row = self._pending.pop(0)
            if row["s1_id"] != s1_id or row["cand_id"] != cand_id:
                raise ValueError(
                    f"extra signal sidecar {self.path} is out of sync at row "
                    f"{self.rows_read}: expected ({s1_id}, {cand_id}) but found "
                    f"({row['s1_id']}, {row['cand_id']})."
                )
            self.rows_read += 1
            out.append(
                {
                    "phon_score": float(row["phon_score"] or 0.0),
                    "dense_score": float(row["dense_score"] or 0.0),
                }
            )
        return out

    def summary(self) -> dict:
        """Row counts for the report."""
        return {
            "enabled": self.enabled,
            "path": str(self.path) if self.path else None,
            "rows": self.rows_read,
        }
