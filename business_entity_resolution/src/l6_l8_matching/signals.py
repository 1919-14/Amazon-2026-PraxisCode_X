"""Retrieval signals shared by training (L7) and inference (L11) - one artifact.

The matcher has six retrieval features. Four are derivable at both training and
inference time from the pair itself (rank, retrieved flag, rank inverse, exact-key
hit), but two are *not*: ``ret_rrf_score`` and ``ret_retriever_agreement`` come
from the Layer 3/4 blocking artifacts.

Layer 7 could join them via ``--l3-dir/--l4-dir``, yet Layer 11 always called
``compute_features(..., signals=None)``, hard-coding both features to ``0``. That
worked only while nobody passed the join flags - a fragile, undocumented
invariant: flipping one flag on the training run would have silently trained the
model on two features that are always zero at scoring time.

This module removes the invariant by writing the signals **once**, next to the
candidate set they describe (Layer 5 has them in hand), and by having both Layer 7
and Layer 11 read that same artifact. Parity is then structural, not accidental:

* :class:`SignalWriter` - streaming writer used by Layer 5.
* :func:`load_signals_for_pairs` - keyed lookup for Layer 7's pair batches.
* :class:`SignalStream` - lockstep reader for Layer 11 (the sidecar is written in
  exactly the same reference/candidate order as ``candidate_pairs.tsv``, so it is
  read row-by-row alongside the candidate file and never held in memory).

Each layer records which mode it used (``sidecar`` vs ``zeros``) in its report so
a mismatch is visible after the fact; Layer 11 also refuses to run against a
booster trained with signals when the sidecar is missing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, Mapping, Optional, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from config import PATH_ARTIFACTS_DIR

SIGNAL_COLUMNS = ["s1_id", "cand_id", "rrf_score", "channel_agreement"]
SIGNAL_SCHEMA = pa.schema(
    [
        ("s1_id", pa.string()),
        ("cand_id", pa.string()),
        ("rrf_score", pa.float32()),
        ("channel_agreement", pa.float32()),
    ]
)

MODE_SIDECAR = "sidecar"
MODE_ZEROS = "zeros"


class SignalsTooLargeError(RuntimeError):
    """Raised when a signal join would exceed the configured memory guard."""


def signals_dir() -> Path:
    """Directory holding candidate-signal sidecars."""
    return PATH_ARTIFACTS_DIR / "signals"


def signals_path(split: str, refs: str) -> Path:
    """Canonical sidecar path for a candidate set (must match the L5 run scope)."""
    return signals_dir() / f"candidates_{split}_{refs}.parquet"


def signals_available(split: str, refs: str) -> bool:
    """True when the sidecar for this candidate set exists."""
    return signals_path(split, refs).exists()


class SignalWriter:
    """Streaming writer for the candidate-signal sidecar."""

    def __init__(self, path: str | Path, chunk: int = 500_000) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.chunk = chunk
        self._writer = pq.ParquetWriter(str(self.path), SIGNAL_SCHEMA)
        self._buffer: dict[str, list] = {name: [] for name in SIGNAL_COLUMNS}
        self.rows = 0

    def add(
        self,
        s1_id: str,
        cand_ids: Sequence[str],
        rrf_scores: Sequence[float],
        channel_agreement: Sequence[float],
    ) -> None:
        """Append one reference's candidate signals in candidate order."""
        for cand_id, rrf, agreement in zip(cand_ids, rrf_scores, channel_agreement):
            self._buffer["s1_id"].append(s1_id)
            self._buffer["cand_id"].append(cand_id)
            self._buffer["rrf_score"].append(float(rrf))
            self._buffer["channel_agreement"].append(float(agreement))
            self.rows += 1
        if len(self._buffer["s1_id"]) >= self.chunk:
            self.flush()

    def flush(self) -> None:
        """Write the buffered rows."""
        if not self._buffer["s1_id"]:
            return
        self._writer.write_table(pa.table(self._buffer, schema=SIGNAL_SCHEMA))
        for values in self._buffer.values():
            values.clear()

    def close(self) -> None:
        """Flush and close the sidecar."""
        self.flush()
        self._writer.close()


def iter_signal_rows(path: str | Path, batch_size: int = 100_000) -> Iterator[dict]:
    """Stream the sidecar as ``{column: value}`` dicts, one per row."""
    parquet_file = pq.ParquetFile(str(path))
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=SIGNAL_COLUMNS):
        data = batch.to_pydict()
        for i in range(batch.num_rows):
            yield {name: data[name][i] for name in SIGNAL_COLUMNS}


def load_signals_for_pairs(
    path: str | Path,
    needed: Optional[Iterable[tuple[str, str]]] = None,
    max_entries: int = 5_000_000,
) -> dict[tuple[str, str], dict[str, float]]:
    """Load ``(s1_id, cand_id) -> signals`` for the requested pairs.

    Args:
        path: sidecar written by Layer 5.
        needed: pair keys to keep; ``None`` keeps every row.
        max_entries: memory guard - a bare dict of every pair in the full test set
            would be several GB, so exceeding this raises instead of swapping.

    Raises:
        SignalsTooLargeError: when the join would exceed ``max_entries``.
    """
    signal_path = Path(path)
    if not signal_path.exists():
        return {}

    wanted = set(needed) if needed is not None else None
    signals: dict[tuple[str, str], dict[str, float]] = {}
    for row in iter_signal_rows(signal_path):
        key = (row["s1_id"], row["cand_id"])
        if wanted is not None and key not in wanted:
            continue
        signals[key] = {
            "rrf_score": float(row["rrf_score"] or 0.0),
            "channel_agreement": float(row["channel_agreement"] or 0.0),
        }
        if len(signals) > max_entries:
            raise SignalsTooLargeError(
                f"signal join needs more than {max_entries:,} pairs "
                f"({signal_path.name}). Either sample the training references "
                f"(main_l3.py --ref-sample N) or use the streaming path."
            )
    return signals


class SignalStream:
    """Lockstep reader: one sidecar row per ``candidate_pairs.tsv`` row.

    Layer 5 writes both files in the same loop, so row *i* of the sidecar always
    describes row *i* of the candidate file. Reading them in lockstep keeps the
    join O(1) in memory even for the ~10M-pair test candidate set.
    """

    def __init__(self, path: str | Path | None, batch_size: int = 100_000) -> None:
        self.path = Path(path) if path is not None else None
        self.enabled = self.path is not None and self.path.exists()
        self._batch_size = batch_size
        self._iterator: Iterator[dict] | None = None
        self._pending: list[dict] = []
        self._exhausted = not self.enabled
        self.rows_read = 0

    def _ensure_iterator(self) -> None:
        if self._iterator is None:
            self._iterator = iter_signal_rows(self.path, self._batch_size)

    def next_row(self, s1_id: str, cand_ids: Sequence[str]) -> list[dict[str, float]]:
        """Return per-candidate signals for one reference, aligned with ``cand_ids``.

        Falls back to zeros when the sidecar is disabled. Raises ``ValueError`` if
        the sidecar drifts out of alignment with the candidate file - a loud failure
        instead of silently attaching the wrong reference's signals.
        """
        if not self.enabled:
            return [{"rrf_score": 0.0, "channel_agreement": 0.0} for _ in cand_ids]

        self._ensure_iterator()
        out: list[dict[str, float]] = []
        expected_ids = list(cand_ids)
        for cand_id in expected_ids:
            while not self._pending:
                assert self._iterator is not None
                try:
                    self._pending.append(next(self._iterator))
                except StopIteration:
                    self._exhausted = True
                    raise ValueError(
                        f"signal sidecar {self.path} ended early at row {self.rows_read}; "
                        "it does not match this candidate file (re-run main_l5.py)."
                    ) from None
            row = self._pending.pop(0)
            if row["s1_id"] != s1_id or row["cand_id"] != cand_id:
                raise ValueError(
                    f"signal sidecar {self.path} is out of sync at row {self.rows_read}: "
                    f"expected ({s1_id}, {cand_id}) but found "
                    f"({row['s1_id']}, {row['cand_id']}). Re-run main_l5.py to "
                    "regenerate the sidecar and the candidate file together."
                )
            self.rows_read += 1
            out.append(
                {
                    "rrf_score": float(row["rrf_score"] or 0.0),
                    "channel_agreement": float(row["channel_agreement"] or 0.0),
                }
            )
        return out

    def summary(self) -> dict:
        """Row counts for the report."""
        return {"enabled": self.enabled, "path": str(self.path) if self.path else None, "rows": self.rows_read}
