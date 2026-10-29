#!/usr/bin/env python3
"""Multi-region CTC / oscillation-gating rate RNN. Prereg: PREREG_oscillation_gating.md."""
from __future__ import annotations
import os, json, time, math, argparse
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"; OUT.mkdir(exist_ok=True)
FIGS = HERE / "figs"; FIGS.mkdir(exist_ok=True)
DATA = HERE / "data"

N_REG = 4
N_LOCAL = 32
N = N_REG * N_LOCAL
INTRA_DENS = 0.15
INTER_DENS = 0.025
PERIOD = 8
INTER_DELAY = 2  # shared myelin stand-in
GRAPH_SEED = 7
XOR_D = 5
TUNE_SEEDS = [90, 91]
EVAL_SEEDS = list(range(1100, 1106))  # fast pilot n=6
ARMS = ["COHERENT", "INCOHERENT", "NO_GATE", "HUB", "RAND_MATCH"]
PRIMARY_TASKS = ["XOR", "CHAR"]

DEFAULT_HP = dict(
    lr=5e-3, leak=0.5, epochs=25, steps_per_epoch=2, batch_xor=64,
    batch_char=48, seq_xor=24, seq_char=48, char_train_chars=30000,
    wd=0.0, emb_dim=24,
)

def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)

def region_of(idx):
    return int(idx) // N_LOCAL

def build_modular_masks(hub=False, seed=GRAPH_SEED):
    """Return (mask_intra, mask_inter) boolean NxN; diagonal False. Same intra always; hub only reallocates inter."""
    g = np.random.default_rng(seed)
    mask_intra = np.zeros((N, N), dtype=bool)
    mask_inter = np.zeros((N, N), dtype=bool)
    for r in range(N_REG):
        sl = slice(r * N_LOCAL, (r + 1) * N_LOCAL)
        block = g.random((N_LOCAL, N_LOCAL)) < INTRA_DENS
        np.fill_diagonal(block, False)
        mask_intra[sl, sl] = block
    # baseline inter edges (same for all modular arms)
    pairs = [(i, j) for i in range(N_REG) for j in range(N_REG) if i != j]
    seen = set(); uniq = []
    for i, j in pairs:
        pre = np.arange(i * N_LOCAL, (i + 1) * N_LOCAL)
        post = np.arange(j * N_LOCAL, (j + 1) * N_LOCAL)
        n_e = max(1, int(round(INTER_DENS * N_LOCAL * N_LOCAL)))
        rr = g.choice(pre, size=n_e * 3, replace=True)
        cc = g.choice(post, size=n_e * 3, replace=True)
        got = 0
        for r, c in zip(rr, cc):
            r, c = int(r), int(c)
            k = r * N + c
            if k in seen or r == c:
                continue
            seen.add(k); uniq.append((r, c)); got += 1
            if got >= n_e:
                break
    target = len(uniq)
    if hub:
        g2 = np.random.default_rng(seed + 4242)
        uniq = []
        seen = set()
        quotas = {
            "h2o": int(round(0.60 * target)),
            "o2h": int(round(0.25 * target)),
        }
        quotas["oo"] = target - quotas["h2o"] - quotas["o2h"]
        def add_edge(r, c):
            k = r * N + c
            if k in seen or r == c:
                return False
            seen.add(k); uniq.append((r, c)); return True
        hub_pre = np.arange(0, N_LOCAL)
        others = np.arange(N_LOCAL, N)
        for key, pre_pool, post_pool in (
            ("h2o", hub_pre, others),
            ("o2h", others, hub_pre),
        ):
            got = 0
            while got < quotas[key]:
                r = int(g2.choice(pre_pool)); c = int(g2.choice(post_pool))
                if add_edge(r, c):
                    got += 1
        nonhub_pairs = [(i, j) for i in range(1, N_REG) for j in range(1, N_REG) if i != j]
        got = 0
        while got < quotas["oo"]:
            i, j = nonhub_pairs[int(g2.integers(0, len(nonhub_pairs)))]
            r = int(g2.integers(i * N_LOCAL, (i + 1) * N_LOCAL))
            c = int(g2.integers(j * N_LOCAL, (j + 1) * N_LOCAL))
            if add_edge(r, c):
                got += 1
        while len(uniq) < target:
            r = int(g2.integers(0, N)); c = int(g2.integers(0, N))
            add_edge(r, c)
        uniq = uniq[:target]
    for r, c in uniq:
        mask_inter[r, c] = True
    return mask_intra, mask_inter

def build_rand_mask(nnz_target, seed):
    g = np.random.default_rng(seed + 12345)
    mask = np.zeros((N, N), dtype=bool)
    got = 0; seen = set()
    while got < nnz_target:
        r = int(g.integers(0, N)); c = int(g.integers(0, N))
        if r == c:
            continue
        k = r * N + c
        if k in seen:
            continue
        seen.add(k); mask[r, c] = True; got += 1
    return mask

def phases_for(arm):
    if arm == "INCOHERENT":
        return np.array([2 * math.pi * r / N_REG for r in range(N_REG)], dtype=np.float64)
    # COHERENT, HUB (coherent), NO_GATE/RAND unused
    return np.zeros(N_REG, dtype=np.float64)

def precompute_gates(phases, period=PERIOD):
    """Return (period, n_reg, n_reg) gates, each pair mean-normalized to 1."""
    t = np.arange(period, dtype=np.float64)
    env = 0.5 * (1.0 + np.cos(2 * math.pi * t[:, None] / period + phases[None, :]))  # (P, R)
    gates = np.zeros((period, N_REG, N_REG), dtype=np.float64)
    for i in range(N_REG):
        for j in range(N_REG):
            if i == j:
                gates[:, i, j] = 1.0
                continue
            raw = env[:, i] * env[:, j]
            m = raw.mean()
            gates[:, i, j] = raw / m if m > 1e-8 else 1.0
    return gates.astype(np.float32)

def make_wiring(arm, seed_eval):
    if arm == "RAND_MATCH":
        mi, mj = build_modular_masks(hub=False, seed=GRAPH_SEED)
        nnz = int(mi.sum() + mj.sum())
        mask = build_rand_mask(nnz, seed_eval)
        return dict(
            arm=arm, mask_intra=np.zeros((N, N), dtype=bool),
            mask_inter=mask,  # treat all as "inter" but ungated path uses full mask
            use_gate=False, phases=np.zeros(N_REG), nnz=nnz, kind="rand",
        )
    hub = arm == "HUB"
    mi, mj = build_modular_masks(hub=hub, seed=GRAPH_SEED)
    use_gate = arm in ("COHERENT", "INCOHERENT", "HUB")
    ph = phases_for(arm)
    return dict(
        arm=arm, mask_intra=mi, mask_inter=mj, use_gate=use_gate,
        phases=ph, nnz=int(mi.sum() + mj.sum()), kind="modular",
    )

class OscRNN(nn.Module):
    def __init__(self, wiring, n_in, leak, out_dim, emb_dim=0, vocab=0):
        super().__init__()
        self.leak = leak
        self.n = N
        self.use_emb = emb_dim > 0 and vocab > 0
        self.use_gate = bool(wiring["use_gate"])
        self.kind = wiring["kind"]
        self.inter_delay = INTER_DELAY
        if self.use_emb:
            self.emb = nn.Embedding(vocab, emb_dim)
            n_in_eff = emb_dim
        else:
            self.emb = None
            n_in_eff = n_in

        mi = wiring["mask_intra"].astype(np.float32)
        mj = wiring["mask_inter"].astype(np.float32)
        self.register_buffer("mask_intra", torch.from_numpy(mi))
        self.register_buffer("mask_inter", torch.from_numpy(mj))
        # region id per neuron
        reg = np.arange(N) // N_LOCAL
        self.register_buffer("reg_ids", torch.from_numpy(reg.astype(np.int64)))

        if self.use_gate:
            gates = precompute_gates(wiring["phases"])
            self.register_buffer("gates", torch.from_numpy(gates))  # (P,R,R)
        else:
            self.register_buffer("gates", torch.ones(PERIOD, N_REG, N_REG))

        # full weight matrices; masked in forward
        self.W_intra = nn.Parameter(0.05 * torch.randn(N, N))
        self.W_inter = nn.Parameter(0.05 * torch.randn(N, N))
        if self.kind == "rand":
            # single matrix under mask_inter (which holds all edges)
            self.W_intra = nn.Parameter(torch.zeros(N, N))  # unused
            self.W_inter = nn.Parameter(0.05 * torch.randn(N, N))

        self.Win = nn.Parameter(0.5 * torch.randn(N, n_in_eff))
        self.b = nn.Parameter(torch.zeros(N))
        self.readout = nn.Linear(N, out_dim)

    def _recurrent(self, x, x_delayed, t_step):
        """x, x_delayed: (B, N)."""
        if self.kind == "rand":
            W = self.W_inter * self.mask_inter
            return x @ W.t()

        Wi = self.W_intra * self.mask_intra
        rec = x @ Wi.t()
        Wj = self.W_inter * self.mask_inter
        # gated inter from delayed state
        src = x_delayed
        if self.use_gate:
            g = self.gates[t_step % PERIOD]  # (R,R)
            # build per-neuron gate: for each post neuron, scale contrib from each pre by g[reg_pre, reg_post]
            # out_j = sum_i W_ji * g[reg_i, reg_j] * src_i
            # = (src * scale) style: construct G_ij = g[reg_i, reg_j]
            rp = self.reg_ids  # (N,)
            # G[pre, post] = g[reg_pre, reg_post]
            G = g[rp][:, rp]  # wait: g[rp[i], rp[j]] -> need fancy
            # g is (R,R); we want G[i,j] = g[reg_i, reg_j]
            G = g[rp[:, None], rp[None, :]]  # (N,N)
            Wg = Wj * G
            rec = rec + src @ Wg.t()
        else:
            rec = rec + src @ Wj.t()
        return rec

    def forward_sequence(self, u):
        if self.use_emb:
            T, B = u.shape
            device = u.device
            dtype = torch.float32
        else:
            T, B, _ = u.shape
            device = u.device
            dtype = u.dtype
        x = torch.zeros(B, self.n, device=device, dtype=dtype)
        # delay ring for inter
        buf = torch.zeros(self.inter_delay + 1, B, self.n, device=device, dtype=dtype)
        outs = []
        for t in range(T):
            buf = torch.roll(buf, 1, dims=0)
            buf[0] = x
            x_del = buf[self.inter_delay]
            rec = self._recurrent(x, x_del, t)
            ut = self.emb(u[t]) if self.use_emb else u[t]
            drive = ut @ self.Win.t() + self.b
            x = (1 - self.leak) * x + self.leak * torch.tanh(rec + drive)
            outs.append(self.readout(x))
        return torch.stack(outs, 0)

def count_params(m):
    # count only nonzero-mask weights roughly + Win/b/readout/emb
    n = 0
    for name, p in m.named_parameters():
        if not p.requires_grad:
            continue
        if name == "W_intra":
            n += int(m.mask_intra.sum().item())
        elif name == "W_inter":
            n += int(m.mask_inter.sum().item())
        else:
            n += p.numel()
    return n

_CHAR_CACHE = {}
def load_alice():
    if "t" not in _CHAR_CACHE:
        text = (DATA / "alice.txt").read_text(errors="ignore")
        chars = sorted(set(text))
        stoi = {c: i for i, c in enumerate(chars)}
        _CHAR_CACHE.update(text=text, chars=chars, stoi=stoi, V=len(chars))
    return _CHAR_CACHE

def make_xor_batch(d, seq_len, batch, rng):
    T = seq_len
    u = rng.integers(0, 2, size=(T, batch, 1)).astype(np.float32)
    y = np.zeros((T, batch, 1), dtype=np.float32)
    for t in range(d, T):
        y[t] = np.logical_xor(u[t] > 0.5, u[t - d] > 0.5).astype(np.float32)
    mask = np.zeros((T, batch), dtype=np.float32); mask[d:] = 1.0
    return torch.from_numpy(u), torch.from_numpy(y), torch.from_numpy(mask)

def make_char_batch(seq_len, batch, start_limit, rng):
    C = load_alice()
    text = C["text"][:start_limit]
    stoi = C["stoi"]
    u = np.zeros((seq_len, batch), dtype=np.int64)
    y = np.zeros((seq_len, batch), dtype=np.int64)
    for b in range(batch):
        s = 0 if len(text) <= seq_len + 1 else int(rng.integers(0, len(text) - seq_len - 1))
        chunk = text[s:s + seq_len + 1]
        for t in range(seq_len):
            u[t, b] = stoi[chunk[t]]
            y[t, b] = stoi[chunk[t + 1]]
    return torch.from_numpy(u), torch.from_numpy(y)

def eval_metric(model, task, seed, hp, n_batches=4):
    rng = np.random.default_rng(seed)
    metrics = []
    model.eval()
    with torch.no_grad():
        for _ in range(n_batches):
            if task == "XOR":
                u, y, mask = make_xor_batch(XOR_D, hp["seq_xor"], hp["batch_xor"], rng)
                logits = model.forward_sequence(u).squeeze(-1)
                pred = (logits > 0).float(); target = y.squeeze(-1); m = mask
                acc = ((pred == target).float() * m).sum() / m.sum().clamp_min(1.0)
                metrics.append(float(acc.item()))
            else:
                u, y = make_char_batch(hp["seq_char"], hp["batch_char"], hp["char_train_chars"], rng)
                pred = model.forward_sequence(u)
                ce = F.cross_entropy(pred.reshape(-1, pred.shape[-1]), y.reshape(-1))
                metrics.append(float(ce.item()))
    return float(np.mean(metrics))

def train_task(arm, task, seed, hp, epochs=None, record_curve=True):
    set_seed(seed)
    epochs = epochs if epochs is not None else hp["epochs"]
    wiring = make_wiring(arm, seed)
    if task == "XOR":
        n_in, out_dim, emb_dim, vocab = 1, 1, 0, 0
    elif task == "CHAR":
        C = load_alice(); n_in, out_dim = hp["emb_dim"], C["V"]
        emb_dim, vocab = hp["emb_dim"], C["V"]
    else:
        raise KeyError(task)
    model = OscRNN(wiring, n_in, hp["leak"], out_dim, emb_dim=emb_dim, vocab=vocab)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    curve = []; rng = np.random.default_rng(seed + 17); best = None
    steps = int(hp.get("steps_per_epoch", 1))
    for ep in range(epochs):
        model.train()
        last_loss = None
        for _step in range(steps):
            if task == "XOR":
                u, y, mask = make_xor_batch(XOR_D, hp["seq_xor"], hp["batch_xor"], rng)
                logits = model.forward_sequence(u).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(logits, y.squeeze(-1), reduction="none")
                loss = (loss * mask).sum() / mask.sum().clamp_min(1.0)
            else:
                u, y = make_char_batch(hp["seq_char"], hp["batch_char"], hp["char_train_chars"], rng)
                pred = model.forward_sequence(u)
                loss = F.cross_entropy(pred.reshape(-1, pred.shape[-1]), y.reshape(-1))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            last_loss = loss
        metric = eval_metric(model, task, seed + 1000 + ep, hp, n_batches=2)
        curve.append({"epoch": ep, "train_loss": float(last_loss.item()), "metric": metric})
        if task == "XOR":
            best = metric if best is None else max(best, metric)
        else:
            best = metric if best is None else min(best, metric)
    final = eval_metric(model, task, seed + 777, hp, n_batches=6)
    return dict(
        arm=arm, task=task, seed=seed, metric=final, best_epoch_metric=best,
        n_params=count_params(model), nnz=int(wiring["nnz"]), n_hidden=N,
        curve=curve if record_curve else [],
    )

def tune():
    grid = [
        dict(DEFAULT_HP, lr=1e-3, leak=0.5, epochs=6),
        dict(DEFAULT_HP, lr=5e-3, leak=0.5, epochs=6),
        dict(DEFAULT_HP, lr=5e-3, leak=0.35, epochs=6),
    ]
    scores = []
    for i, hp in enumerate(grid):
        vals = []
        for arm in ("COHERENT", "INCOHERENT"):
            for s in TUNE_SEEDS:
                r_xor = train_task(arm, "XOR", s, hp, epochs=hp["epochs"], record_curve=False)
                r_ch = train_task(arm, "CHAR", s, hp, epochs=5, record_curve=False)
                vals.append(r_xor["metric"] - 0.05 * r_ch["metric"])
                print(f"  tune{i} {arm} seed{s} xor={r_xor['metric']:.3f} char={r_ch['metric']:.3f}", flush=True)
        scores.append((float(np.mean(vals)), i, hp))
        print("tune", i, "score", scores[-1][0], flush=True)
    scores.sort(reverse=True)
    best = dict(scores[0][2]); best["epochs"] = DEFAULT_HP["epochs"]
    (OUT / "hparams.json").write_text(json.dumps(best, indent=2))
    print("froze hparams", best, flush=True)
    return best

def load_hp():
    p = OUT / "hparams.json"
    return json.loads(p.read_text()) if p.exists() else dict(DEFAULT_HP)

def _worker(arm, task, seed, hp):
    t0 = time.time()
    r = train_task(arm, task, seed, hp, record_curve=True)
    r["wall_s"] = time.time() - t0
    # curves can be large; keep them
    return r

def run_eval(arms=None, tasks=None, seeds=None, workers=2):
    hp = load_hp()
    arms = arms or ARMS
    tasks = tasks or PRIMARY_TASKS
    seeds = seeds or EVAL_SEEDS
    jobs = [(a, t, s) for a in arms for t in tasks for s in seeds]
    jf = OUT / "runs.jsonl"
    done = set()
    if jf.exists():
        for line in jf.read_text().splitlines():
            if line.strip():
                r = json.loads(line); done.add((r["arm"], r["task"], r["seed"]))
    todo = [j for j in jobs if j not in done]
    print(f"jobs {len(jobs)} todo {len(todo)} workers {workers}", flush=True)
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        futs = {ex.submit(_worker, a, t, s, hp): (a, t, s) for a, t, s in todo}
        for fut in as_completed(futs):
            r = fut.result()
            # strip curve? keep for figure
            with open(jf, "a") as f:
                f.write(json.dumps(r) + "\n")
            print(f"{r['arm']} {r['task']} seed={r['seed']} metric={r['metric']:.4f} {r['wall_s']:.1f}s", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True); FIGS.mkdir(exist_ok=True)
    if args.smoke:
        hp = dict(DEFAULT_HP, epochs=2, batch_xor=16, batch_char=16)
        for arm in ARMS:
            for task in ("XOR", "CHAR"):
                r = train_task(arm, task, 90, hp, epochs=2)
                print("smoke", r["arm"], r["task"], f"metric={r['metric']:.4f}", f"nnz={r['nnz']}", f"npar={r['n_params']}", flush=True)
        # param match check
        nz = {}
        for arm in ARMS:
            w = make_wiring(arm, 90)
            nz[arm] = w["nnz"]
        print("nnz by arm", nz, flush=True)
        return
    if args.tune:
        tune()
    if args.eval:
        run_eval(args.arms, args.tasks, args.seeds, workers=args.workers)

if __name__ == "__main__":
    main()
