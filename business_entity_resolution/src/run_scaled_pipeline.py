"""Run the full scaled pipeline (L3 -> L11) in order, with memory guards.

This is the one-command driver for the scaled architecture:

    normalization (L2, done once)
      -> L3  multi-channel blocking (address text, topk)
      -> L4  RRF fusion
      -> L4x fuse phonetic (G) + dense (H) channels
      -> L5  truncate to K + write lexical & extra signal sidecars
      -> L6  training pairs
      -> L7  features v2 (45)
      -> L8  matcher (single LightGBM or the GBDT stack)
      -> L9  calibration
      -> L10 decision v2 (per-country + entity-level F0.5)
      -> L11 test inference -> output/matching_results.tsv

Each stage runs as a subprocess so a crash cannot take the driver down, and the
driver prints available RAM before every stage - the ramp that OOMed the earlier
attempts. Use ``--start`` / ``--stop`` to run a slice, and ``--dry-run`` to print
the commands.

Examples::

    # full train pass on India (tune the matcher)
    venv/Scripts/python.exe business_entity_resolution/src/run_scaled_pipeline.py \
        --mode train --countries india

    # test pass -> submission
    venv/Scripts/python.exe business_entity_resolution/src/run_scaled_pipeline.py \
        --mode test --countries us india france

    # just retrain the stack on existing features and re-infer
    venv/Scripts/python.exe business_entity_resolution/src/run_scaled_pipeline.py \
        --start l8 --stop l11 --model-kind stack
"""

from __future__ import annotations

import argparse
import ctypes
import subprocess
import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

STAGES = ["l3", "l4", "l4x", "l5", "l6", "l7", "l8", "l9", "l10", "l11"]


def available_ram_gb() -> float:
    """Return available physical RAM in GB (Windows/Linux), or -1 when unknown."""
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / 1e6
    except Exception:
        pass

    class _MemStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    try:
        status = _MemStatus()
        status.dwLength = ctypes.sizeof(_MemStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return status.ullAvailPhys / 1e9
    except Exception:
        return -1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scaled pipeline driver")
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--countries", nargs="+", default=["india"])
    parser.add_argument("--topk", type=int, default=200)
    parser.add_argument("--k-min", type=int, default=200)
    parser.add_argument("--k-max", type=int, default=200)
    parser.add_argument("--addr-text", action="store_true", default=True)
    parser.add_argument("--ref-sample", type=int, default=400000)
    parser.add_argument("--model-kind", choices=["lgbm", "stack"], default="lgbm")
    parser.add_argument("--bases", nargs="+", default=["lgbm", "xgboost"])
    parser.add_argument("--decision-v2", action="store_true", default=True)
    parser.add_argument("--phon-template", default=None,
                        help="Phonetic artifact template, e.g. artifacts/blocking/phon_{split}_{refs}_country={country}.parquet")
    parser.add_argument("--dense-template", default=None)
    parser.add_argument("--start", choices=STAGES, default="l3")
    parser.add_argument("--stop", choices=STAGES, default="l11")
    parser.add_argument("--dense", action="store_true", help="Also generate dense (H) artifacts first.")
    parser.add_argument("--phon", action="store_true", help="Also generate phonetic (G) artifacts first.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--min-ram-gb", type=float, default=1.5,
                        help="Abort before a stage if available RAM is below this.")
    return parser.parse_args()


def _py(script: str) -> list[str]:
    return [sys.executable, str(SRC_DIR / script)]


def build_commands(args: argparse.Namespace) -> dict[str, list[list[str]]]:
    """Return the ordered command list(s) for every stage."""
    split = "test" if args.mode == "test" else "train"
    refs = "all" if args.mode == "test" else "train"
    countries = args.countries
    cmds: dict[str, list[list[str]]] = {s: [] for s in STAGES}

    cmds["l3"] = [_py("main_l3.py") + (
        ["--split", split, "--refs", refs, "--countries", *countries,
         "--topk", str(args.topk)]
        + (["--addr-text"] if args.addr_text else [])
        + (["--ref-sample", str(args.ref_sample)] if args.mode == "train" else [])
    )]
    cmds["l4"] = [_py("main_l4.py") + ["--split", split, "--refs", refs, "--countries", *countries]]

    extras: list[str] = []
    for template in (args.phon_template, args.dense_template):
        if template:
            extras += ["--extra-template", template]
    cmds["l4x"] = [_py("main_l4_extra.py") + ["--split", split, "--refs", refs, "--countries", *countries, *extras]] if extras else []
    cmds["l5"] = [_py("main_l5.py") + (
        ["--split", split, "--refs", refs, "--countries", *countries,
         "--no-tune", "--ratio", "0.0", "--k-min", str(args.k_min), "--k-max", str(args.k_max)]
        + (["--l4-template", "artifacts/blocking/l4x_{split}_{refs}_country={country}.parquet"] if extras else [])
        + extras
    )]
    cmds["l6"] = [_py("main_l6.py") + ["--candidates", f"output/candidate_pairs_{refs}.tsv"]] if args.mode == "train" else []
    cmds["l7"] = [_py("main_l7.py") + ["--pairs", "artifacts/train_pairs/variant_a.parquet", "--signals-refs", refs]] if args.mode == "train" else []
    cmds["l8"] = [_py("main_l8.py") + ["--variant", "a", "--model-kind", args.model_kind, "--bases", *args.bases]] if args.mode == "train" else []
    cmds["l9"] = [_py("main_l9.py") + ["--variant", "a"]] if args.mode == "train" else []
    cmds["l10"] = [_py("main_l10.py") + ["--split", split, "--variant", "a", "--use-calibrated"]
                   + (["--decision-v2"] if args.decision_v2 else [])] if args.mode == "train" else []
    cmds["l11"] = [_py("main_l11.py") + ["--variant", "a", "--split", split, "--signals-refs", refs, "--coverage-refs", refs]
                   + (["--decision-v2"] if args.decision_v2 else [])
                   + (["--model-kind", args.model_kind] if args.model_kind == "stack" else [])
                   + ["--no-scores"]] if args.mode == "test" else []
    return cmds


def main() -> None:
    args = parse_args()
    commands = build_commands(args)
    started = STAGES.index(args.start)
    finished = STAGES.index(args.stop)
    if started > finished:
        raise SystemExit("--start must not come after --stop")

    print("=" * 70)
    print(f"SCALED PIPELINE | mode={args.mode} countries={args.countries} model={args.model_kind}")
    print(f"stages: {STAGES[started:finished+1]} | available RAM: {available_ram_gb():.2f} GB")
    print("=" * 70)

    for stage in STAGES[started:finished+1]:
        stage_cmds = commands.get(stage) or []
        if not stage_cmds:
            print(f"\n[{stage}] skipped (not applicable in {args.mode} mode)")
            continue
        ram = available_ram_gb()
        if 0 <= ram < args.min_ram_gb:
            raise SystemExit(
                f"\n❌ [{stage}] only {ram:.2f} GB RAM available (need >= {args.min_ram_gb}). "
                "Close other applications and retry; the earlier OOMs were exactly this."
            )
        for cmd in stage_cmds:
            print(f"\n{'─'*70}\n[{stage}] RAM {ram:.2f} GB\n$ {' '.join(cmd)}\n{'─'*70}", flush=True)
            if args.dry_run:
                continue
            t0 = time.perf_counter()
            code = subprocess.run(cmd, cwd=str(SRC_DIR.parent)).returncode
            elapsed = time.perf_counter() - t0
            print(f"[{stage}] exit={code} elapsed={elapsed/60:.1f} min", flush=True)
            if code != 0:
                raise SystemExit(f"❌ [{stage}] failed with exit code {code}")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE" + ("" if args.mode == "test" else " (train)"))
    if args.mode == "test":
        print("  submission: business_entity_resolution/output/matching_results.tsv")
    print("=" * 70)


if __name__ == "__main__":
    main()
