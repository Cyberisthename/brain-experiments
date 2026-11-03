"""Shared tasks for engine v3 (NaN-safe NARMA)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

NARMA_ORDER = 10
XOR_HARD_D = 12
SPLIT_BIND_D = 8
D_MATCH = 12
DMAX = 5

HIGHER_BETTER = {"XOR_HARD", "TRANSFER", "SPLIT_BIND", "DELAYED_MATCH", "XOR2BIT"}

_CHAR_CACHE = {}


def load_alice(data_dir: Path | None = None):
    if "t" in _CHAR_CACHE:
        return _CHAR_CACHE
    candidates = []
    if data_dir is not None:
        candidates.append(Path(data_dir) / "alice.txt")
    here = Path(__file__).resolve().parent
    root = here.parent
    candidates += [
        root / "regions" / "callosum_scaled" / "data" / "alice.txt",
        root / "regions" / "human_wholebrain" / "data" / "alice.txt",
    ]
    text = None
    for p in candidates:
        if p.exists():
            text = p.read_text(errors="ignore")
            break
    if text is None:
        text = "alice was beginning to get very tired of sitting by her sister " * 200
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    _CHAR_CACHE.update(text=text, chars=chars, stoi=stoi, V=len(chars), t=True)
    return _CHAR_CACHE


def make_xor_hard_batch(d, seq_len, batch, rng):
    T = seq_len
    u = rng.integers(0, 2, size=(T, batch, 2)).astype(np.float32)
    y = np.zeros((T, batch, 1), dtype=np.float32)
    for t in range(d, T):
        y[t, :, 0] = np.logical_xor(u[t - d, :, 0] > 0.5, u[t - d, :, 1] > 0.5).astype(np.float32)
    mask = np.zeros((T, batch), dtype=np.float32)
    mask[d:] = 1.0
    return torch.from_numpy(u), torch.from_numpy(y), torch.from_numpy(mask)


def make_transfer_batch(d, seq_len, batch, rng):
    T = seq_len
    u = rng.integers(0, 2, size=(T, batch, 1)).astype(np.float32)
    y = np.zeros((T, batch, 1), dtype=np.float32)
    for t in range(d, T):
        y[t, :, 0] = u[t - d, :, 0]
    mask = np.zeros((T, batch), dtype=np.float32)
    mask[d:] = 1.0
    return torch.from_numpy(u), torch.from_numpy(y), torch.from_numpy(mask)


def make_split_bind_batch(d, seq_len, batch, rng):
    T = seq_len
    u = rng.integers(0, 2, size=(T, batch, 2)).astype(np.float32)
    y = np.zeros((T, batch, 1), dtype=np.float32)
    for t in range(d, T):
        y[t, :, 0] = np.logical_xor(u[t - d, :, 0] > 0.5, u[t - d, :, 1] > 0.5).astype(np.float32)
    mask = np.zeros((T, batch), dtype=np.float32)
    mask[d:] = 1.0
    return torch.from_numpy(u), torch.from_numpy(y), torch.from_numpy(mask)


def make_char_batch(seq_len, batch, start_limit, rng, data_dir=None):
    C = load_alice(data_dir)
    text = C["text"][:start_limit]
    stoi = C["stoi"]
    u = np.zeros((seq_len, batch), dtype=np.int64)
    y = np.zeros((seq_len, batch), dtype=np.int64)
    for b in range(batch):
        s = 0 if len(text) <= seq_len + 1 else int(rng.integers(0, len(text) - seq_len - 1))
        chunk = text[s : s + seq_len + 1]
        for t in range(seq_len):
            u[t, b] = stoi[chunk[t]]
            y[t, b] = stoi[chunk[t + 1]]
    return torch.from_numpy(u), torch.from_numpy(y)


def make_narma_batch(seq_len, batch, rng, max_abs=50.0, max_tries=50):
    """NARMA-10 with rejection of unstable targets."""
    T = seq_len
    u = rng.uniform(0, 0.5, size=(T, batch, 1)).astype(np.float32)
    y = np.zeros((T, batch, 1), dtype=np.float32)
    for b in range(batch):
        for attempt in range(max_tries):
            if attempt > 0:
                u[:, b, 0] = rng.uniform(0, 0.5, size=(T,)).astype(np.float32)
            y[:, b, 0] = 0.0
            ok = True
            for i in range(NARMA_ORDER, T - 1):
                val = (
                    0.3 * y[i, b, 0]
                    + 0.05 * y[i, b, 0] * y[i - 9 : i + 1, b, 0].sum()
                    + 1.5 * u[i - 9, b, 0] * u[i, b, 0]
                    + 0.1
                )
                if (not np.isfinite(val)) or abs(val) > max_abs:
                    ok = False
                    break
                y[i + 1, b, 0] = val
            if ok:
                break
        else:
            y[:, b, 0] = np.nan_to_num(y[:, b, 0], nan=0.0, posinf=max_abs, neginf=-max_abs)
            np.clip(y[:, b, 0], -max_abs, max_abs, out=y[:, b, 0])
    mask = np.zeros((T, batch), dtype=np.float32)
    mask[20:] = 1.0
    return torch.from_numpy(u), torch.from_numpy(y), torch.from_numpy(mask)


def task_io(task, hp, dmax=DMAX):
    xfer_d = max(8, dmax + 2)
    if task == "XOR_HARD":
        return 2, 1, 0, 0, "left_hemi", "hemi_split", XOR_HARD_D
    if task == "CHAR":
        C = load_alice()
        return hp["emb_dim"], C["V"], hp["emb_dim"], C["V"], "left_hemi", "left_drive", 0
    if task == "NARMA":
        return 1, 1, 0, 0, "left_hemi", "left_drive", 0
    if task == "TRANSFER":
        return 1, 1, 0, 0, "lh_motor", "transfer_rh_visual", xfer_d
    if task == "SPLIT_BIND":
        return 2, 1, 0, 0, "left_hemi", "hemi_split", SPLIT_BIND_D
    if task == "DELAYED_MATCH":
        return 2, 1, 0, 0, "left_hemi", "delayed_match", D_MATCH
    raise KeyError(task)


def batch_for_task(task, hp, rng, d_task=None):
    if task == "XOR_HARD":
        return make_xor_hard_batch(XOR_HARD_D, hp["seq_xor"], hp["batch_xor"], rng)
    if task == "CHAR":
        u, y = make_char_batch(hp["seq_char"], hp["batch_char"], hp["char_train_chars"], rng)
        return u, y, None
    if task == "NARMA":
        return make_narma_batch(hp["seq_narma"], hp["batch_narma"], rng)
    d = d_task if d_task is not None else max(8, DMAX + 2)
    if task == "TRANSFER":
        return make_transfer_batch(d, hp["seq_xfer"], hp["batch_xfer"], rng)
    if task == "SPLIT_BIND":
        return make_split_bind_batch(SPLIT_BIND_D, hp["seq_xfer"], hp["batch_xfer"], rng)
    raise KeyError(task)


def loss_and_metric_from_pred(task, pred, y, mask):
    """Return (loss_tensor, metric_float_or_None for single batch)."""
    if task in ("XOR_HARD", "TRANSFER", "SPLIT_BIND", "DELAYED_MATCH"):
        logits = pred.squeeze(-1) if pred.shape[-1] == 1 else pred
        target = y.squeeze(-1) if y.shape[-1] == 1 else y
        loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        loss = (loss * mask).sum() / mask.sum().clamp_min(1.0)
        with torch.no_grad():
            p = (logits > 0).float()
            acc = ((p == target).float() * mask).sum() / mask.sum().clamp_min(1.0)
        return loss, float(acc.item())
    if task == "CHAR":
        loss = F.cross_entropy(pred.reshape(-1, pred.shape[-1]), y.reshape(-1))
        return loss, float(loss.detach().item())
    if task == "NARMA":
        pr = pred.squeeze(-1)
        yt = y.squeeze(-1)
        mse = ((pr - yt) ** 2 * mask).sum() / mask.sum().clamp_min(1.0)
        with torch.no_grad():
            std = yt[mask > 0].std().clamp_min(1e-8)
            nrmse = torch.sqrt(mse) / std
        return mse, float(nrmse.item())
    raise KeyError(task)
