#!/usr/bin/env python3
"""Orchestrate DTM(+MPS+SKF) discrete + continuous."""
from __future__ import annotations
import argparse, os, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent

def run(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(HERE))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--skip-continuous", action="store_true")
    ap.add_argument("--skip-discrete", action="store_true")
    args = ap.parse_args()
    out = args.out_dir or os.environ.get("OUT_DIR") or str(HERE / "out")
    os.environ["OUT_DIR"] = out
    os.environ.setdefault("SBQA_GRIFFA_NPZ", str(HERE / "data" / "griffa70_scale68_group_minimal.npz"))
    Path(out).mkdir(parents=True, exist_ok=True)
    flag = ["--smoke"] if args.smoke else []
    common = flag + ["--device", args.device, "--out-dir", out]
    if not args.skip_discrete:
        run([sys.executable, "-u", "run_discrete.py", "--task", "narma", *common])
        if not args.smoke:
            run([sys.executable, "-u", "run_discrete.py", "--task", "xor", *common])
    if not args.skip_continuous:
        run([sys.executable, "-u", "run_continuous.py", "--task", "narma", *common])
    run([sys.executable, "-u", "analyze.py", "--out-dir", out])
    print("ALL DONE out=", out)

if __name__ == "__main__":
    main()
