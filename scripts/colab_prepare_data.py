"""Prepare small module .npz files for Colab GPU runs.

The GitHub repo excludes connectome binaries. This script:
  1. Ensures /workspace/flybrain -> repo root (experiments hardcode that path).
  2. Rebuilds human_module.npz from HuggingFace edges when possible.
  3. Symlinks modules into experiment data/ folders.
  4. Tells you which tiny files to upload by hand if rebuild sources are missing.

Preferred rebuild (no secrets):
  - Human edges: HuggingFace NathanRoll/h01-cortex-snn → hf_edges.npz + hf_metadata.npz
    then: python3 regions/hybrid_human/build_human_module.py
  - Human myelin: python3 regions/hybrid_human/extract_h01_myelin.py  (public GCS Range GETs)
  - Fly CX / mouse: upload the small modules listed below (rebuild needs ~100MB parquet
    / multi-GB Zenodo which Colab free disks often cannot hold).

Upload-by-hand (total ~1 MB) if rebuild is skipped:
  regions/frankenstein/data/fly_cx_module.npz
  regions/frankenstein/data/mouse_module.npz
  regions/frankenstein/data/human_module.npz   (or rebuild)
  regions/hybrid_human/data/human_myelin.npz   (or extract)

Usage (from repo root on Colab):
  python3 scripts/colab_prepare_data.py
  python3 scripts/colab_prepare_data.py --download-hf
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def ensure_workspace_link(repo: Path):
    ws = Path("/workspace")
    target = ws / "flybrain"
    ws.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.is_symlink() and target.resolve() == repo.resolve():
            print(f"OK {target} -> {repo}")
            return
        if target.resolve() == repo.resolve():
            print(f"OK {target} is repo")
            return
        print(f"NOTE: {target} already exists ({target}); not replacing")
        return
    target.symlink_to(repo)
    print(f"linked {target} -> {repo}")


def download_hf(data: Path):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
        from huggingface_hub import hf_hub_download

    data.mkdir(parents=True, exist_ok=True)
    for name, dest in [
        ("edges.npz", data / "hf_edges.npz"),
        ("metadata.npz", data / "hf_metadata.npz"),
    ]:
        if dest.exists():
            print(f"exists {dest.name}")
            continue
        path = hf_hub_download("NathanRoll/h01-cortex-snn", filename=name)
        Path(dest).write_bytes(Path(path).read_bytes())
        print(f"downloaded {dest.name}")


def try_build_human():
    script = REPO / "regions/hybrid_human/build_human_module.py"
    data = REPO / "regions/hybrid_human/data"
    out = data / "human_module.npz"
    if out.exists():
        print(f"exists {out}")
        return True
    if not (data / "hf_edges.npz").exists():
        print("SKIP human build — missing hf_edges.npz (pass --download-hf)")
        return False
    print("building human_module.npz …")
    subprocess.check_call([sys.executable, str(script)])
    return out.exists()


def try_extract_myelin():
    script = REPO / "regions/hybrid_human/extract_h01_myelin.py"
    out = REPO / "regions/hybrid_human/data/human_myelin.npz"
    if out.exists():
        print(f"exists {out}")
        return True
    if not (REPO / "regions/hybrid_human/data/human_module.npz").exists():
        print("SKIP myelin extract — need human_module.npz first")
        return False
    print("extracting human_myelin.npz from H01 public GCS (Range GETs) …")
    try:
        subprocess.check_call([sys.executable, str(script)])
    except subprocess.CalledProcessError as e:
        print(f"myelin extract failed: {e}")
        return False
    return out.exists()


def symlink_modules():
    frank = REPO / "regions/frankenstein/data"
    human_src = REPO / "regions/hybrid_human/data/human_module.npz"
    myelin_src = REPO / "regions/hybrid_human/data/human_myelin.npz"
    mouse_src = REPO / "regions/hybrid/data/mouse_module.npz"
    # Prefer frankenstein copies as the canonical experiment inputs
    mapping = {
        REPO / "regions/frankenstein/data/human_module.npz": human_src,
        REPO / "regions/frankenstein/data/mouse_module.npz": mouse_src,
    }
    for dst, src in mapping.items():
        if dst.exists() or dst.is_symlink():
            continue
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.symlink_to(src)
            print(f"link {dst.relative_to(REPO)} -> {src.relative_to(REPO)}")

    # Experiment data folders that expect the modules
    need = {
        "fly_cx_module.npz": frank / "fly_cx_module.npz",
        "mouse_module.npz": frank / "mouse_module.npz",
        "human_module.npz": frank / "human_module.npz",
        "human_myelin.npz": myelin_src,
    }
    for exp in ("best_fly", "myelin_real_rep", "mega_franken", "trained_bridges"):
        d = REPO / "regions" / exp / "data"
        d.mkdir(parents=True, exist_ok=True)
        for name, src in need.items():
            dst = d / name
            if dst.exists() or dst.is_symlink():
                continue
            if src.exists():
                dst.symlink_to(src.resolve())
                print(f"link regions/{exp}/data/{name}")


REQUIRED = [
    "regions/frankenstein/data/fly_cx_module.npz",
    "regions/frankenstein/data/mouse_module.npz",
    "regions/frankenstein/data/human_module.npz",
    "regions/hybrid_human/data/human_myelin.npz",
]


def status():
    missing = []
    for rel in REQUIRED:
        p = REPO / rel
        ok = p.exists()
        print(f"{'OK ' if ok else 'MISS'} {rel}")
        if not ok:
            missing.append(rel)
    return missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download-hf", action="store_true", help="Download H01 HF edges/metadata")
    ap.add_argument("--extract-myelin", action="store_true", help="Run H01 myelin extractor")
    args = ap.parse_args()

    ensure_workspace_link(REPO)
    data = REPO / "regions/hybrid_human/data"
    if args.download_hf:
        download_hf(data)
    try_build_human()
    if args.extract_myelin:
        try_extract_myelin()
    symlink_modules()
    missing = status()
    if missing:
        print("\nUpload these by hand into the cloned repo (Colab Files panel), then re-run:")
        for m in missing:
            print(f"  {m}")
        print("\nOr rebuild: --download-hf (human), --extract-myelin (myelin).")
        print("Fly CX / mouse modules have no tiny public rebuild on Colab; upload the .npz files.")
        return 2
    print("\nAll required modules present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
