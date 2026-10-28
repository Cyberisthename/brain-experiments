#!/usr/bin/env python3
"""
braid_vs_gd_benchmark.py — structured-task ladder: braid-space trainer vs GD vs seed-opt.

agent-theoretical-physicist · 2026-10-09 · pure Python + numpy/scipy · deterministic.

The owner's direction: give the 3-phase braid-space anneal/evolution trainer a fair,
honest shot at beating plain gradient descent on INCREASINGLY STRUCTURED tasks, and
validate wins with the probe battery (no-LHV / CHSH) rather than raw loss curves.

HONEST GATE PROTOCOL
---------------------
Every contestant on every task gets the SAME evaluation budget (BUDGET forward
evaluations). Every number is a fresh, deterministic, committed run. Honest ceiling:
"deterministic anneal/evolution over braid space" — NO quantum tunneling, NO hardware,
NO universal-QC claim. If braids lose everywhere, that is reported straight.

TASK LADDER (increasing structure)
----------------------------------
  T0  smooth tanh regression        (GD's home turf; featureless, differentiable)
  T1  adapter-tagged multi-target   (rugged: K=3 teachers drawn from adapters/*.json tags)
  T2  braid-invariant matching      (discrete topological: Burau trace-vector distance)
  T3  no-LHV CHSH threshold         (probe-native: maximize CHSH of the FBSC braid state)

CONTESTANTS (equal budget; where a contestant does not apply it is marked N/A):
  gd       float-weight gradient descent (continuous)          [T0, T1]
  seedopt  Nelder-Mead over the 3-seed -> braid word/state      [T0, T1, T2, T3]
  trainer  min(anneal, evolve) over braid words (T0-T2); anneal [T0, T1, T2]
           over the 3-seed manifold (T3 = braid-state anneal)   [T3]

PROBE (owner's direction): the T3 objective IS the probe (CHSH). The no-LHV certificate
(lhv_feasible LP) is computed once per best solution, not inside the search loop.

Outputs: docs/artifacts/braid_vs_gd/braid_vs_gd_results.json + a markdown verdict table.
"""
import json
import math
import hashlib
import glob
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize

# ---- import the owned FBSC braid-state engine (for the T3 probe) -----------
# Vendored next to this file under flybrain/vendor/ (repo-relative; no chatbot-site-src).
_VENDOR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(_VENDOR, ".."))  # flybrain repo root (adapters/ optional)
sys.path.insert(0, _VENDOR)
from likeness_probes import fbsc_braid_state, partial_trace, PAULIS  # noqa: E402

# ============================================================================
# Determinism + budget
# ============================================================================
RNG_SEED = 20261005
N_STRANDS = 6
WORD_LEN = 8
T_PARAM = 2.0
N_SAMPLES = 64
BUDGET = 800          # equal evaluation budget for every contestant on every task
TOL = 1e-6

# ============================================================================
# Burau braid machinery (semantics identical to quantum_topological_trainer.py)
# ============================================================================
def burau_sigma(n, i, t, inverse=False):
    M = np.eye(n, dtype=np.complex128)
    if inverse:
        block = np.array([[0.0, 1.0], [1.0 / t, 1.0 - 1.0 / t]], dtype=np.complex128)
    else:
        block = np.array([[1.0 - t, t], [1.0, 0.0]], dtype=np.complex128)
    M[i - 1:i + 1, i - 1:i + 1] = block
    return M

def burau_matrix(word, n, t):
    M = np.eye(n, dtype=np.complex128)
    for (i, sign) in word:
        M = M @ burau_sigma(n, i, t, inverse=(sign < 0))
    return M

def weights_from_word(word, n=N_STRANDS, t=T_PARAM):
    return np.real(burau_matrix(word, n, t))

def burau_trace_vector(word, n, t_values):
    return [np.trace(burau_matrix(word, n, t)) for t in t_values]

def invariant_distance(v1, v2):
    return float(np.mean([abs(a - b) for a, b in zip(v1, v2)]))

def word_from_seed(seed, n=N_STRANDS, L=WORD_LEN):
    h = hashlib.sha256(("|".join(f"{x:.8f}" for x in seed)).encode()).digest()
    rng = np.random.RandomState(int.from_bytes(h[:4], 'big'))
    return [(int(rng.randint(1, n - 1)), int(1 if rng.rand() < 0.5 else -1))
            for _ in range(L)]

def random_word(rng, n=N_STRANDS, L=WORD_LEN):
    return [(int(rng.randint(1, n - 1)), int(1 if rng.rand() < 0.5 else -1))
            for _ in range(L)]

# ============================================================================
# Tasks
# ============================================================================
def make_regression_target(seed, n=N_STRANDS, n_samples=N_SAMPLES):
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n) / np.sqrt(n)
    w_teacher = random_word(rng)
    W_star = weights_from_word(w_teacher, n, T_PARAM)
    Y = np.tanh(X @ W_star)
    return X, Y, W_star, w_teacher

def tanh_loss(W, X, Y):
    A = np.tanh(X @ W)
    return float(np.mean((A - Y) ** 2))

def tanh_loss_grad(W, X, Y):
    Z = X @ W
    A = np.tanh(Z)
    err = A - Y
    return (2.0 / X.shape[0]) * (X.T @ (err * (1.0 - A ** 2)))

def word_loss(word, X, Y):
    return tanh_loss(weights_from_word(word), X, Y)

# ---------------------------------------------------------------------------
# T0 — smooth toy regression (GD native)
# ---------------------------------------------------------------------------
class T0:
    name = "T0_smooth_regression"
    desc = "single teacher, smooth tanh regression (featureless, differentiable)"
    gd_applicable = True
    def __init__(self):
        self.X, self.Y, self.W_star, self.w_teacher = make_regression_target(RNG_SEED)
    def cost_word(self, word):
        return word_loss(word, self.X, self.Y)
    def cost_seed(self, seed):
        return self.cost_word(word_from_seed(tuple(seed)))
    def cost_W(self, W):
        return tanh_loss(W, self.X, self.Y)
    def grad_W(self, W):
        return tanh_loss_grad(W, self.X, self.Y)

# ---------------------------------------------------------------------------
# T1 — adapter-tagged multi-target regression (rugged)
# ---------------------------------------------------------------------------
class T1:
    name = "T1_adapter_multitarget"
    desc = "K=3 teacher words derived from adapter task_tags -> multimodal landscape"
    gd_applicable = True
    def __init__(self, k=3):
        tags = []
        for fp in sorted(glob.glob(os.path.join(REPO, "adapters", "*.json"))):
            try:
                d = json.load(open(fp))
            except Exception:
                continue
            tags.extend(d.get("task_tags", []))
        # distinct tags -> deterministic teacher words (adapter-backed curriculum)
        distinct = list(dict.fromkeys(tags))
        self.tags_used = distinct[:k]
        self.X, _, self.W_star0, _ = make_regression_target(RNG_SEED)
        self.targets = []
        self.teacher_words = []
        for tag in self.tags_used:
            h = hashlib.sha256(tag.encode()).digest()
            rng = np.random.RandomState(int.from_bytes(h[:4], 'big'))
            w = random_word(rng)
            self.teacher_words.append(w)
            self.targets.append(np.tanh(self.X @ weights_from_word(w)))
    def cost_word(self, word):
        W = weights_from_word(word)
        return float(np.mean([tanh_loss(W, self.X, Y) for Y in self.targets]))
    def cost_seed(self, seed):
        return self.cost_word(word_from_seed(tuple(seed)))
    def cost_W(self, W):
        return float(np.mean([tanh_loss(W, self.X, Y) for Y in self.targets]))
    def grad_W(self, W):
        gs = [tanh_loss_grad(W, self.X, Y) for Y in self.targets]
        return np.mean(np.stack(gs), axis=0)

# ---------------------------------------------------------------------------
# T2 — braid-invariant matching (discrete topological)
# ---------------------------------------------------------------------------
class T2:
    name = "T2_invariant_matching"
    desc = "minimize Burau trace-vector distance to a reference word (discrete; GD N/A)"
    gd_applicable = False
    def __init__(self):
        self.ref_word = random_word(np.random.RandomState(RNG_SEED + 7))
        self.T_VALS = [-1.0, 2.0, 0.5 + 0.5j]
        self.target_inv = burau_trace_vector(self.ref_word, N_STRANDS, self.T_VALS)
    def cost_word(self, word):
        inv = burau_trace_vector(word, N_STRANDS, self.T_VALS)
        return invariant_distance(inv, self.target_inv)
    def cost_seed(self, seed):
        return self.cost_word(word_from_seed(tuple(seed)))

# ---------------------------------------------------------------------------
# T3 — no-LHV CHSH threshold (probe-native)
# ---------------------------------------------------------------------------
def chsh_horodecki(psi, n):
    """Closed-form Horodecki CHSH = 2 sqrt(s1^2 + s2^2) from the T-matrix SVD."""
    rho = partial_trace(psi, n, [0, 1])
    X, Y, Z = PAULIS['X'], PAULIS['Y'], PAULIS['Z']
    T = np.array([[np.real(np.trace(rho @ np.kron(si, sj))) for sj in (X, Y, Z)]
                  for si in (X, Y, Z)])
    S = np.linalg.svd(T, compute_uv=False)
    return 2.0 * math.sqrt(S[0] ** 2 + S[1] ** 2)

def _safe_seed(seed):
    """Clip a seed to the FBSC engine's valid box (keeps log/denominator finite)."""
    return tuple(float(np.clip(x, 0.05, 3.0)) for x in seed)

class T3:
    name = "T3_nolhv_chsh"
    desc = "maximize CHSH (probe) of the FBSC braid state; GD N/A (no gradient)"
    gd_applicable = False
    NQ = 6
    def cost_seed(self, seed):
        psi = fbsc_braid_state(_safe_seed(seed), self.NQ)
        return -chsh_horodecki(psi, self.NQ)   # minimize -CHSH == maximize CHSH

# ============================================================================
# Contestants (equal budget)
# ============================================================================
def gd_contest(task, budget=BUDGET, lr=0.5):
    """Float-weight gradient descent on cost_W."""
    n = task.X.shape[1]
    rng = np.random.RandomState(RNG_SEED)
    W = rng.randn(n, n) * 0.1
    evals = 0
    for _ in range(budget):
        W = W - lr * task.grad_W(W)
        evals += 1
    return task.cost_W(W), evals

def seedopt_contest(task, budget=BUDGET):
    def obj(seed):
        return task.cost_seed(tuple(seed))
    x0 = np.array([1.0, 1.0, 1.0])
    res = minimize(obj, x0, method='Nelder-Mead',
                   options={'maxfev': budget, 'xatol': 1e-3, 'fatol': 1e-6})
    return obj(res.x), int(res.nfev)

def anneal_word_contest(task, budget=BUDGET, T0=2.0, Tmin=1e-3):
    rng = np.random.RandomState(RNG_SEED + 1)
    w = random_word(rng)
    cur = task.cost_word(w)
    best = cur
    for s in range(budget):
        T = T0 * (Tmin / T0) ** (s / budget)
        cand = list(w)
        move = rng.choice(['flip', 'change_gen', 'insert_kink', 'delete_kink'])
        if move == 'flip' and cand:
            k = int(rng.randint(0, len(cand))); cand[k] = (cand[k][0], -cand[k][1])
        elif move == 'change_gen' and cand:
            k = int(rng.randint(0, len(cand))); cand[k] = (int(rng.randint(1, N_STRANDS - 1)), cand[k][1])
        elif move == 'insert_kink':
            i = int(rng.randint(1, N_STRANDS - 1)); pos = int(rng.randint(0, len(cand) + 1))
            cand[pos:pos] = [(i, 1), (i, -1)]
        elif move == 'delete_kink' and cand:
            del cand[int(rng.randint(0, len(cand)))]
        c = task.cost_word(cand)
        if c < cur or rng.rand() < np.exp(-(c - cur) / max(T, 1e-9)):
            w, cur = cand, c
        if cur < best:
            best = cur
    return best, budget

def evolve_word_contest(task, budget=BUDGET, pop=20):
    rng = np.random.RandomState(RNG_SEED + 2)
    gen = max(1, budget // pop)
    popw = [random_word(rng) for _ in range(pop)]
    fit = [task.cost_word(w) for w in popw]
    for _ in range(gen):
        order = np.argsort(fit)
        parents = [popw[i] for i in order[:pop // 2]]
        popw = []
        for _ in range(pop):
            p = parents[int(rng.randint(0, len(parents)))]
            child = list(p)
            for _ in range(int(rng.randint(1, 3))):
                k = int(rng.randint(0, len(child)))
                if rng.rand() < 0.7:
                    child[k] = (child[k][0], -child[k][1])
                else:
                    child[k] = (int(rng.randint(1, N_STRANDS - 1)), child[k][1])
            popw.append(child)
        fit = [task.cost_word(w) for w in popw]
    return float(min(fit)), pop * gen

def anneal_seed_contest(task, budget=BUDGET, T0=2.0, Tmin=1e-3):
    """Annealing over the 3-seed manifold (T3 = the braid-state anneal)."""
    rng = np.random.RandomState(RNG_SEED + 3)
    seed = np.array([0.5, 0.5, 0.5])
    cur = task.cost_seed(seed)
    best, best_seed = cur, _safe_seed(seed)
    for s in range(budget):
        T = T0 * (Tmin / T0) ** (s / budget)
        cand = seed + rng.randn(3) * 0.2
        c = task.cost_seed(cand)
        if c < cur or rng.rand() < np.exp(-(c - cur) / max(T, 1e-9)):
            seed, cur = cand, c
        if cur < best:
            best, best_seed = cur, _safe_seed(seed)
    return best, budget, best_seed

def trainer_contest(task, budget=BUDGET):
    """min(anneal, evolve) over braid words; for T3, seed-anneal (braid-state).
    Equal-budget rule: the trainer's TOTAL evaluations are capped at `budget`
    (anneal gets budget//2, evolve gets budget//2) so it gets no extra evals vs GD."""
    if isinstance(task, T3):
        c, ev, best_seed = anneal_seed_contest(task, budget)
        return c, ev, best_seed
    half = budget // 2
    a = anneal_word_contest(task, half)
    e = evolve_word_contest(task, half)
    return min(a[0], e[0]), a[1] + e[1]

# ============================================================================
# Pre-registered predictions (recorded BEFORE the run, committed with the code)
# ============================================================================
PRE_REGISTERED = {
    "T0_smooth_regression": {
        "prediction": "GD wins or ties (smooth differentiable landscape; gradient is the right tool)",
        "where_braid_advantage": "none expected — braid anneal gets stuck in local minima",
    },
    "T1_adapter_multitarget": {
        "prediction": "uncertain — multimodal landscape; anneal/evolve MAY escape local minima GD cannot",
        "where_braid_advantage": "possible — stochastic escape vs gradient descent into a local mode",
    },
    "T2_invariant_matching": {
        "prediction": "braids win (discrete topological objective; braid moves search the NATIVE space)",
        "where_braid_advantage": "expected — GD has no gradient through discrete braid moves (N/A)",
    },
    "T3_nolhv_chsh": {
        "prediction": "braids win or tie (probe is defined over the FBSC braid state manifold)",
        "where_braid_advantage": "expected — the objective is the probe; seed-anneal searches it natively",
    },
}

# ============================================================================
# Runner
# ============================================================================
def run_task(task):
    out = {"name": task.name, "desc": task.desc, "budget": BUDGET,
           "gd_applicable": task.gd_applicable}
    if hasattr(task, "tags_used"):
        out["adapter_tags_used"] = task.tags_used
    if isinstance(task, T2):
        out["reference_word"] = [[int(i), int(s)] for (i, s) in task.ref_word]
        out["t_values"] = [str(t) for t in task.T_VALS]
    # gd
    if task.gd_applicable:
        c, ev = gd_contest(task)
        out["gd"] = {"cost": round(c, 8), "evals": ev}
    else:
        out["gd"] = {"cost": None, "evals": 0, "note": "N/A — no gradient through discrete braid moves"}
    # seedopt
    c, ev = seedopt_contest(task)
    out["seedopt"] = {"cost": round(c, 8), "evals": ev}
    # trainer
    tr = trainer_contest(task)
    c, ev = tr[0], tr[1]
    out["trainer"] = {"cost": round(c, 8), "evals": ev}
    if isinstance(task, T3):
        out["trainer"]["best_seed"] = list(tr[2])
    # gate verdict (minimize cost)
    out["beats_gd"] = None if not task.gd_applicable else bool(
        out["trainer"]["cost"] < out["gd"]["cost"] - TOL)
    out["beats_seedopt"] = bool(out["trainer"]["cost"] < out["seedopt"]["cost"] - TOL)
    out["tie_seedopt"] = bool(abs(out["trainer"]["cost"] - out["seedopt"]["cost"]) <= TOL)
    return out

def main():
    t0 = time.time()
    results = {
        "meta": {
            "experiment": "braid_vs_gd_benchmark",
            "rng_seed": RNG_SEED, "n_strands": N_STRANDS, "word_len": WORD_LEN,
            "burau_t": T_PARAM, "budget_per_contestant": BUDGET, "tol": TOL,
            "honest_ceiling": ("deterministic anneal/evolution over braid space; no quantum "
                               "tunneling, no hardware, no universal-QC claim"),
            "probe": "no-LHV CHSH (Horodecki closed form) on the FBSC braid state (T3 objective)",
        },
        "pre_registered": PRE_REGISTERED,
    }
    tasks = [T0(), T1(k=3), T2(), T3()]
    for task in tasks:
        results[task.name] = run_task(task)

    # T3 probe certificate: full LHV feasibility on the trainer's BEST seed
    # (single call at the end, not in the search loop).
    try:
        from likeness_probes import lhv_feasible
        best_seed = tuple(results["T3_nolhv_chsh"]["trainer"]["best_seed"])
        psi = fbsc_braid_state(best_seed, 6)
        chsh = chsh_horodecki(psi, 6)
        lhv_ok, _ = lhv_feasible(psi, 6)
        results["T3_probe_certificate"] = {
            "seed": list(best_seed), "chsh_horodecki": round(chsh, 6),
            "lhv_bound": 2.0, "lhv_feasible": bool(lhv_ok),
            "verdict": ("no-LHV (provably outside every local-hidden-variable model)"
                        if (not lhv_ok and chsh > 2.0) else
                        ("LHV-compatible" if lhv_ok else "inconclusive")),
        }
    except Exception as exc:
        results["T3_probe_certificate"] = {"error": str(exc)}

    results["wall_clock_seconds"] = round(time.time() - t0, 3)

    outdir = os.path.join(REPO, "docs", "artifacts", "braid_vs_gd")
    os.makedirs(outdir, exist_ok=True)
    outp = os.path.join(outdir, "braid_vs_gd_results.json")
    with open(outp, "w") as f:
        json.dump(results, f, indent=2)

    # ---- console verdict table ----
    print("\n=== braid_vs_gd_benchmark — verdict table (equal budget %d) ===" % BUDGET)
    print(f"{'task':26s} {'gd':>12s} {'seedopt':>12s} {'trainer':>12s}  verdict")
    for task in tasks:
        r = results[task.name]
        g = f"{r['gd']['cost']:.6f}" if r['gd']['cost'] is not None else "N/A"
        s = f"{r['seedopt']['cost']:.6f}"
        t = f"{r['trainer']['cost']:.6f}"
        if r['gd']['cost'] is None:
            if r['beats_seedopt']:
                v = "trainer wins"
            elif r['tie_seedopt']:
                v = "tie"
            else:
                v = "trainer loses"
        else:
            if r['beats_gd']:
                v = "beats GD"
            elif r['beats_seedopt']:
                v = "beats seedopt (loses GD)"
            else:
                v = "loses"
        print(f"{task.name:26s} {g:>12s} {s:>12s} {t:>12s}  {v}")
    print(f"\nwrote {outp}")
    print(f"wall-clock: {results['wall_clock_seconds']}s")

if __name__ == "__main__":
    main()
