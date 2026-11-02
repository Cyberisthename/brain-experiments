"""Adam train loop + resume-safe jsonl for engine v3."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn

from .rate_network import OscRNN, OscRNNConfig, count_params, synthetic_wiring
from . import tasks as T

ENGINE_TAG = "v3"

DEFAULT_HP = dict(
    lr=5e-3,
    leak=0.4,
    epochs=30,
    steps_per_epoch=12,
    batch_xor=32,
    batch_char=24,
    batch_narma=24,
    batch_xfer=32,
    seq_xor=48,
    seq_char=40,
    seq_narma=64,
    seq_xfer=48,
    char_train_chars=20000,
    wd=0.0,
    emb_dim=32,
)


def set_seed(s: int):
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def configure_threads(n: int = 1):
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    torch.set_num_threads(n)


def load_done_keys(jsonl: Path, key_fields=("arm", "task", "seed", "feature")):
    done = set()
    if not jsonl.exists():
        return done
    for line in jsonl.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("poisoned") or r.get("nonfinite"):
            continue
        done.add(tuple(r.get(k) for k in key_fields))
    return done


def append_jsonl(path: Path, record: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    line = json.dumps(record) + "\n"
    # atomic-ish append: write line via open append (jsonl append is standard)
    with open(path, "a") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def eval_metric(model, task, seed, hp, n_batches=4):
    rng = np.random.default_rng(seed)
    metrics = []
    _, _, _, _, _, _, d_task = T.task_io(task, hp)
    model.eval()
    with torch.no_grad():
        for _ in range(n_batches):
            if task == "CHAR":
                u, y, mask = T.batch_for_task(task, hp, rng)
                pred = model.forward_sequence(u)
                loss, m = T.loss_and_metric_from_pred(task, pred, y, mask)
                metrics.append(m)
            else:
                u, y, mask = T.batch_for_task(task, hp, rng, d_task=d_task)
                pred = model.forward_sequence(u)
                loss, m = T.loss_and_metric_from_pred(task, pred, y, mask)
                metrics.append(m)
    return float(np.mean(metrics))


def train_task(
    wiring: dict,
    task: str,
    seed: int,
    hp: dict,
    feature: str = "BASE",
    cfg: Optional[OscRNNConfig] = None,
    record_curve: bool = True,
    arm: str = "SYNTH",
) -> dict:
    configure_threads(1)
    set_seed(seed)
    cfg = cfg or OscRNNConfig(leak=hp.get("leak", 0.4))
    n_in, out_dim, emb_dim, vocab, readout_mode, input_mode, d_task = T.task_io(task, hp)
    model = OscRNN(
        wiring,
        n_in,
        out_dim,
        cfg=cfg,
        emb_dim=emb_dim,
        vocab=vocab,
        readout_mode=readout_mode,
        input_mode=input_mode,
        seed=seed,
    )
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp.get("wd", 0.0))
    curve = []
    rng = np.random.default_rng(seed + 17)
    best = None
    steps = int(hp.get("steps_per_epoch", 1))
    epochs = int(hp.get("epochs", 30))
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
            pred = model.forward_sequence(u)
            loss, _ = T.loss_and_metric_from_pred(task, pred, y, mask if mask is not None else torch.ones(1))
            if not torch.isfinite(loss):
                aborted = True
                abort_reason = f"nonfinite_loss_ep{ep}"
                break
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            last_loss = loss
        if aborted:
            break
        metric = eval_metric(model, task, seed + 1000 + ep, hp, n_batches=2)
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
            engine=ENGINE_TAG,
            arm=arm,
            task=task,
            seed=seed,
            feature=feature,
            poisoned=True,
            nonfinite=True,
            abort_reason=abort_reason,
            metric=None,
            best_epoch_metric=best,
            n_params=count_params(model),
            nnz=int(wiring.get("nnz", 0)),
            n_hidden=int(wiring["n_hidden"]),
            curve=curve if record_curve else [],
        )

    final = eval_metric(model, task, seed + 777, hp, n_batches=6)
    return dict(
        engine=ENGINE_TAG,
        arm=arm,
        task=task,
        seed=seed,
        feature=feature,
        metric=final,
        best_epoch_metric=best,
        n_params=count_params(model),
        nnz=int(wiring.get("nnz", 0)),
        n_hidden=int(wiring["n_hidden"]),
        n_bridge=int(wiring.get("n_bridge", 0)),
        cfg=dict(
            use_stp=cfg.use_stp,
            use_dual_tau=cfg.use_dual_tau,
            use_dale=cfg.use_dale,
            use_homeostasis=cfg.use_homeostasis,
            leak=cfg.leak,
        ),
        curve=curve if record_curve else [],
        poisoned=False,
        nonfinite=False,
    )


def feature_cfg(name: str, leak: float = 0.4) -> OscRNNConfig:
    name = name.upper()
    cfg = OscRNNConfig(leak=leak)
    if name in ("BASE", "NONE", "OFF"):
        return cfg
    if name in ("STP", "+STP", "STP_FAC"):
        cfg.use_stp = True
        return cfg
    if name in ("DUAL", "DUAL_TAU", "+DUAL"):
        cfg.use_dual_tau = True
        return cfg
    if name in ("DALE", "DALE_BAL", "+DALE"):
        cfg.use_dale = True
        return cfg
    if name in ("HOMEO", "HOMEOSTASIS", "+HOMEO"):
        cfg.use_homeostasis = True
        return cfg
    if name in ("ALL", "STP+DUAL+DALE+HOMEO"):
        cfg.use_stp = True
        cfg.use_dual_tau = True
        cfg.use_dale = True
        cfg.use_homeostasis = True
        return cfg
    raise KeyError(f"unknown feature arm {name}")


def run_job(
    arm: str,
    task: str,
    seed: int,
    feature: str,
    wiring_fn: Callable,
    hp: dict,
    out_jsonl: Path,
) -> dict:
    t0 = time.time()
    wiring = wiring_fn(arm, seed)
    cfg = feature_cfg(feature, leak=hp.get("leak", 0.4))
    r = train_task(wiring, task, seed, hp, feature=feature, cfg=cfg, arm=arm)
    r["wall_s"] = time.time() - t0
    if not r.get("poisoned"):
        append_jsonl(out_jsonl, r)
    else:
        # still log abort so resume skips / can retry manually
        append_jsonl(out_jsonl, r)
    return r
