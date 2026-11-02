#!/usr/bin/env python3
"""Feature ablation — resume-safe. Prereg: PREREG_v3_features.md"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "regions" / "callosum_scaled"))

from engine_v3.rate_network import OscRNNConfig, synthetic_wiring
from engine_v3.train import (
    DEFAULT_HP,
    append_jsonl,
    feature_cfg,
    load_done_keys,
    train_task,
)

OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
JSONL = OUT / "ablation_runs.jsonl"

FEATURES = ["BASE", "STP", "DUAL_TAU", "DALE", "HOMEO", "ALL"]
TASKS = ["XOR_HARD", "NARMA", "CHAR", "TRANSFER"]
SEEDS = list(range(3300, 3308))  # 8 declared
N_LABEL = 512

HP = dict(
    DEFAULT_HP,
    epochs=25,
    steps_per_epoch=10,
    batch_xor=32,
    batch_char=24,
    batch_narma=24,
    batch_xfer=32,
    seq_xor=48,
    seq_char=40,
    seq_narma=64,
    seq_xfer=48,
    char_train_chars=20000,
    emb_dim=32,
)


def make_wiring_safe(seed: int):
    try:
        import callosum_scaled as cs

        n_local = cs.N_LABEL_TO_LOCAL[N_LABEL]
        w = dict(cs.make_wiring("REAL_SC", seed, n_local))
        w["dmax"] = cs.DMAX
        w["_source"] = "callosum_REAL_SC"
        return w
    except Exception as e:
        w = synthetic_wiring(n=N_LABEL, seed=seed, p_intra=0.05, p_inter=0.01)
        w["_source"] = f"synthetic_fallback:{e}"
        return w


def _worker(feature, task, seed, hp):
    t0 = time.time()
    wiring = make_wiring_safe(seed)
    cfg = feature_cfg(feature, leak=hp["leak"])
    # align with callosum seed offset for REAL_SC reproducibility of Win init scale
    r = train_task(
        wiring, task, seed, hp, feature=feature, cfg=cfg, arm="REAL_SC", record_curve=True,
    )
    r["wall_s"] = time.time() - t0
    r["n_label"] = N_LABEL
    r["wiring_source"] = wiring.get("_source", "unknown")
    return r


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--features", nargs="*", default=None)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    args = ap.parse_args()

    features = args.features or FEATURES
    tasks = args.tasks or TASKS
    seeds = args.seeds or SEEDS
    jobs = [(f, t, s) for f in features for t in tasks for s in seeds]
    done = load_done_keys(JSONL)
    todo = [j for j in jobs if j not in done]
    print(f"ablation jobs={len(jobs)} todo={len(todo)} workers={args.workers} hp={HP}", flush=True)
    (OUT / "ablation_hparams.json").write_text(json.dumps(HP, indent=2))

    if args.workers <= 1:
        for f, t, s in todo:
            r = _worker(f, t, s, HP)
            append_jsonl(JSONL, r)
            m = r.get("metric")
            print(
                f"{r['feature']} {r['task']} seed={r['seed']} metric={m} "
                f"{r['wall_s']:.1f}s src={r.get('wiring_source')}",
                flush=True,
            )
    else:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
            futs = {ex.submit(_worker, f, t, s, HP): (f, t, s) for f, t, s in todo}
            for fut in as_completed(futs):
                r = fut.result()
                append_jsonl(JSONL, r)
                print(
                    f"{r['feature']} {r['task']} seed={r['seed']} metric={r.get('metric')} "
                    f"{r['wall_s']:.1f}s",
                    flush=True,
                )


if __name__ == "__main__":
    main()
