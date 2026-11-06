#!/usr/bin/env python3
"""Run hypothalamus motif tests A/B/C. Prereg: PREREG_ABC.md + HPARAMS_FROZEN.md.

Resume-safe jsonl. Single worker, torch threads=1. Does not touch callosum_scaled.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "regions" / "callosum_scaled"))

from engine_v3.rate_network import synthetic_wiring, count_params
from engine_v3 import tasks as T
from engine_v3.train import (
    DEFAULT_HP,
    append_jsonl,
    load_done_keys,
    set_seed,
    configure_threads,
    eval_metric,
)
from hypothalamus.hypo_model import HypoOscRNN, HypoModConfig

N_LABEL = 512
SEEDS = list(range(3400, 3408))
TASKS = ["XOR_HARD", "NARMA", "CHAR", "TRANSFER"]

# Arms by test — BASE shared by A and C (written once with feature=BASE)
ARMS_A = ["HOME_O_ERR", "RAND_MOD", "BASE"]
ARMS_B = ["RIVAL", "SINGLE", "SHUF_RIVAL"]
ARMS_C = ["VOL", "SYN"]  # BASE from A

ALL_ARMS = ["BASE", "HOME_O_ERR", "RAND_MOD", "RIVAL", "SINGLE", "SHUF_RIVAL", "VOL", "SYN"]

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
    wd=0.0,
    lr=5e-3,
    leak=0.4,
)

# Motif numerics — must match HPARAMS_FROZEN.md
HYPO_NUM = dict(
    beta=0.05,
    kappa=0.5,
    s_star=0.3,
    inhibit=0.5,
    temperature=1.0,
    n_pi=8,
    k_syn=32,
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


def train_hypo(wiring, task, seed, arm, hp):
    configure_threads(1)
    set_seed(seed, deterministic=True)
    hypo = HypoModConfig(arm=arm, seed=seed, **HYPO_NUM)
    n_in, out_dim, emb_dim, vocab, readout_mode, input_mode, d_task = T.task_io(task, hp)
    model = HypoOscRNN(
        wiring,
        n_in,
        out_dim,
        hypo=hypo,
        emb_dim=emb_dim,
        vocab=vocab,
        readout_mode=readout_mode,
        input_mode=input_mode,
        seed=seed,
        leak=hp["leak"],
    )
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp.get("wd", 0.0))
    curve = []
    rng = np.random.default_rng(seed + 17)
    best = None
    steps = int(hp["steps_per_epoch"])
    epochs = int(hp["epochs"])
    higher = task in T.HIGHER_BETTER
    aborted = False
    abort_reason = None

    for ep in range(epochs):
        model.train()
        last_loss = None
        for _step in range(steps):
            if task == "CHAR":
                u, y, mask = T.batch_for_task(task, hp, rng)
            else:
                u, y, mask = T.batch_for_task(task, hp, rng, d_task=d_task)
            if mask is None:
                mask = torch.ones(1)
            pred = model.forward_sequence(u)
            loss, _ = T.loss_and_metric_from_pred(task, pred, y, mask)
            if not torch.isfinite(loss):
                aborted = True
                abort_reason = f"nonfinite_loss_ep{ep}"
                break
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            # Test B drive update from loss (after step)
            model.update_drives_from_loss(float(loss.detach().item()))
            last_loss = loss
        if aborted:
            break
        metric = eval_metric(model, task, seed + 1000 + ep, hp, n_batches=2, device="cpu")
        if not np.isfinite(metric):
            aborted = True
            abort_reason = f"nonfinite_metric_ep{ep}"
            break
        curve.append({"epoch": ep, "train_loss": float(last_loss.item()), "metric": metric})
        if higher:
            best = metric if best is None else max(best, metric)
        else:
            best = metric if best is None else min(best, metric)

    if aborted:
        return dict(
            engine="v3_hypo",
            arm="REAL_SC",
            test_arm=arm,
            feature=arm,
            task=task,
            seed=seed,
            poisoned=True,
            nonfinite=True,
            abort_reason=abort_reason,
            metric=None,
            best_epoch_metric=best,
            n_params=count_params(model),
            nnz=int(wiring.get("nnz", 0)),
            n_hidden=int(wiring["n_hidden"]),
            hypo=dict(HYPO_NUM),
            curve=curve,
        )

    final = eval_metric(model, task, seed + 777, hp, n_batches=6, device="cpu")
    return dict(
        engine="v3_hypo",
        arm="REAL_SC",
        test_arm=arm,
        feature=arm,
        task=task,
        seed=seed,
        metric=final,
        best_epoch_metric=best,
        n_params=count_params(model),
        nnz=int(wiring.get("nnz", 0)),
        n_hidden=int(wiring["n_hidden"]),
        n_bridge=int(wiring.get("n_bridge", 0)),
        hypo=dict(HYPO_NUM),
        wiring_source=wiring.get("_source", "unknown"),
        curve=curve,
        poisoned=False,
        nonfinite=False,
    )


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--jsonl", type=str, default=str(HERE / "results" / "abc_runs.jsonl"))
    ap.add_argument("--smoke", action="store_true", help="1 epoch smoke")
    args = ap.parse_args()

    hp = dict(HP)
    if args.smoke:
        hp["epochs"] = 1
        hp["steps_per_epoch"] = 1

    arms = args.arms or ALL_ARMS
    tasks = args.tasks or TASKS
    seeds = args.seeds or SEEDS
    jsonl = Path(args.jsonl)
    jsonl.parent.mkdir(parents=True, exist_ok=True)

    jobs = [(a, t, s) for a in arms for t in tasks for s in seeds]
    done = load_done_keys(jsonl, key_fields=("feature", "task", "seed"))
    todo = [j for j in jobs if j not in done]
    print(
        f"hypo_abc jobs={len(jobs)} todo={len(todo)} done={len(done)} "
        f"jsonl={jsonl} hp_epochs={hp['epochs']}",
        flush=True,
    )
    (jsonl.parent / "abc_hparams.json").write_text(
        json.dumps({"train": HP, "hypo": HYPO_NUM, "seeds": SEEDS, "arms": ALL_ARMS}, indent=2)
    )

    for a, t, s in todo:
        # refuse to start if callosum parent vanished unexpectedly? no — just never kill it
        t0 = time.time()
        wiring = make_wiring_safe(s)
        r = train_hypo(wiring, t, s, a, hp)
        r["wall_s"] = time.time() - t0
        r["n_label"] = N_LABEL
        append_jsonl(jsonl, r)
        print(
            f"{r['feature']} {r['task']} seed={r['seed']} metric={r.get('metric')} "
            f"{r['wall_s']:.1f}s src={r.get('wiring_source')} poisoned={r.get('poisoned')}",
            flush=True,
        )


if __name__ == "__main__":
    main()
