#!/usr/bin/env python3
"""Copy matching data files from /kaggle/input/** into expected repo paths.

Usage (from repo root, or any cwd — script finds repo via __file__):
  python scripts/kaggle_setup.py

Also accepts FLYBRAIN_INPUT=/path/to/fake/input for local smoke tests.
Reports OK / MISS for every required path; exits 1 if any MISS.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Destinations relative to repo root. Basename match from input tree.
NEEDED = [
    "regions/frankenstein/data/fly_cx_module.npz",
    "regions/frankenstein/data/mouse_module.npz",
    "regions/frankenstein/data/human_module.npz",
    "regions/hybrid_human/data/human_module.npz",
    "regions/hybrid_human/data/human_myelin.npz",
    "regions/hybrid_human/data/SOURCE_myelin.json",
    "regions/out/graphs/0_REAL_0.npz",
    "regions/empn/out/circuit.npz",
    "regions/empn/out/circuit_meta.npz",
]

# Optional extras (report only; do not fail the run)
OPTIONAL = [
    "regions/hybrid_human/data/SOURCE.json",
    "regions/mega_franken/data/human_myelin.npz",
    "regions/trained_bridges/data/human_myelin.npz",
]


def input_roots() -> list[Path]:
    roots: list[Path] = []
    env = os.environ.get("FLYBRAIN_INPUT")
    if env:
        roots.append(Path(env))
    kin = Path("/kaggle/input")
    if kin.exists():
        roots.append(kin)
    return roots


def index_by_name(roots: list[Path]) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file() and p.name not in found:
                found[p.name] = p
    return found


def main() -> int:
    roots = input_roots()
    index = index_by_name(roots)
    print(f"repo={ROOT}")
    print(f"input_roots={[str(r) for r in roots] or ['(none)']}")
    print(f"indexed_files={len(index)}")

    ok, miss = [], []
    for rel in NEEDED:
        dest = ROOT / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        src = index.get(Path(rel).name)
        if dest.exists() and dest.stat().st_size > 0:
            # Prefer existing repo file; still mark OK
            ok.append(rel)
            print(f"OK   {rel}  (already present, {dest.stat().st_size} bytes)")
            continue
        if src is None:
            miss.append(rel)
            print(f"MISS {rel}")
            continue
        shutil.copy2(src, dest)
        ok.append(rel)
        print(f"OK   {rel}  <- {src}  ({dest.stat().st_size} bytes)")

    for rel in OPTIONAL:
        dest = ROOT / rel
        if dest.exists():
            print(f"OPT  {rel}  (present)")
            continue
        src = index.get(Path(rel).name)
        if src is None:
            print(f"OPT  {rel}  (skip)")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        print(f"OPT  {rel}  <- {src}")

    print(f"summary ok={len(ok)} miss={len(miss)}")
    if miss:
        print("MISSING required files — attach dataset brain-data / kaggle_data_v2 or rebuild.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
