"""Shared report persistence: history-preserving JSON merges + markdown rendering.

Every layer used to write a single report file that the next run overwrote, so a
multi-country / multi-split sweep silently lost every result but the last one.
For example ``output/l3_blocking_report.json`` described only the most recent
country bucket (France), while the US/India recall measurements it replaced were
gone for good.

``merge_json_report`` keeps each run under ``runs[<run_key>]`` and additionally
mirrors the newest payload at the top level, so existing readers that expect
e.g. ``report["config"]["tau_match"]`` keep working unchanged.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping

RESERVED_KEYS = ("runs", "latest_run", "run_keys")


def load_json_report(path: str | Path) -> dict | None:
    """Load a JSON report, returning ``None`` when it is absent or unreadable."""
    report_path = Path(path)
    if not report_path.exists():
        return None
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def merge_json_report(
    path: str | Path,
    run_key: str,
    payload: Mapping[str, Any],
) -> dict:
    """Merge one run into a report file, keeping every previous run.

    Args:
        path: report path (created if missing).
        run_key: unique key for this run, e.g. ``"test_all_france"``.
        payload: JSON-serialisable run payload.

    Returns:
        The merged report (``runs`` mapping + newest payload mirrored on top).
    """
    report_path = Path(path)
    existing = load_json_report(report_path) or {}
    runs: dict[str, Any] = dict(existing.get("runs") or {})
    if not runs and existing:
        # Migrate a pre-history report (the old format overwrote on every run):
        # keep whatever it held instead of dropping the only record of that run.
        legacy = {key: value for key, value in existing.items() if key not in RESERVED_KEYS}
        if legacy:
            runs["legacy_previous_run"] = legacy
    runs[run_key] = dict(payload)

    merged: dict[str, Any] = {
        key: value for key, value in payload.items() if key not in RESERVED_KEYS
    }
    merged["runs"] = runs
    merged["latest_run"] = run_key
    merged["run_keys"] = sorted(runs)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = report_path.with_name(report_path.name + ".tmp")
    tmp_path.write_text(json.dumps(merged, indent=2, default=str), encoding="utf-8")
    os.replace(tmp_path, report_path)
    return merged


def run_history(runs: Mapping[str, Any]) -> dict[str, Any]:
    """Return the ``runs`` mapping of a report (empty when there is none)."""
    return dict(runs or {})


def render_markdown_table(
    runs: Mapping[str, Any],
    columns: Mapping[str, str],
    row_fn: Callable[[str, Mapping[str, Any]], Mapping[str, Any]],
    *,
    latest: str | None = None,
) -> str:
    """Render every run of a report as one markdown table row.

    Args:
        runs: ``run_key -> payload`` mapping.
        columns: ``column header -> payload field name`` (header order preserved).
        row_fn: ``(run_key, payload) -> {column header: value}`` extractor.
        latest: run key to mark as the latest run.
    """
    headers = ["run"] + list(columns.keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for run_key in sorted(runs):
        payload = runs[run_key] or {}
        values = row_fn(run_key, payload)
        cells = [f"`{run_key}`"]
        for header in columns:
            value = values.get(header, payload.get(columns[header], ""))
            cells.append("" if value is None else str(value))
        if latest == run_key:
            cells[0] = f"`{run_key}` (latest)"
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_text(path: str | Path, text: str) -> Path:
    """Write a text file, creating parent directories."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return out_path
