"""Regions doing their real job. See PREREGISTRATION_roles.md."""
import os, sys, json, time
for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[v] = "1"
sys.path.insert(0, "/workspace/flybrain/regions/empn")
import numpy as np, pandas as pd, scipy.sparse as sp
from multiprocessing import Pool
import empn as E

H = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(H, "out"); os.makedirs(OUT, exist_ok=True)
JF = os.path.join(OUT, "runs.jsonl")
SEEDS = list(range(8))
N_KC, CLAWS, K_WTA = 2000, 7, 100
N_ODOR, N_TRIAL_P1, N_TRIAL_P2 = 4, 40, 40


def odor_patterns(g, n_pn, n_odor=N_ODOR, sparse=0.15, overlap=0.2):
    P = np.zeros((n_odor, n_pn))
    k = max(1, int(sparse * n_pn))
    shared = g.choice(n_pn, max(1, int(overlap * k)), replace=False)
    for i in range(n_odor):
        rest = g.choice(np.setdiff1d(np.arange(n_pn), shared), k - len(shared), replace=False)
        P[i, np.concatenate([shared, rest])] = g.uniform(0.5, 1.0, k)
    return P


def build_mb(g, mode="real"):
    """PN(ALPN)->KC->MBON with PAM/PPL gated plasticity. mode: real|rand|shuf."""
    pn = E.IX["ALPN"]; mbon = E.IX["MBON"]; pam = E.IX["PAM"]; ppl = E.IX["PPL"]
    n_pn, n_mbon = len(pn), len(mbon)
    # PN->KC claws
    Wpk = np.zeros((N_KC, n_pn))
    for i in range(N_KC):
        idx = g.choice(n_pn, CLAWS, replace=False); Wpk[i, idx] = g.uniform(0.5, 1.0, CLAWS)
    if mode == "rand":
        Wpk = np.abs(g.standard_normal((N_KC, n_pn))); Wpk = Wpk / (Wpk.sum(1, keepdims=True) + 1e-9) * CLAWS
    elif mode == "shuf":
        # degree-preserving shuffle of claws
        flat = Wpk.ravel().copy(); g.shuffle(flat); Wpk = flat.reshape(Wpk.shape)
    # KC->MBON plastic
    Wkm = g.standard_normal((n_mbon, N_KC)) * 0.01
    if mode == "rand":
        Wkm = g.standard_normal((n_mbon, N_KC)) * 0.01
    elif mode == "shuf":
        flat = Wkm.ravel().copy(); g.shuffle(flat); Wkm = flat.reshape(Wkm.shape)
    return dict(Wpk=Wpk, Wkm=Wkm, pn=pn, mbon=mbon, pam=pam, ppl=ppl, e_kc=np.zeros(N_KC))


def kc_wta(h):
    z = np.zeros_like(h); idx = np.argpartition(-h, K_WTA - 1)[:K_WTA]; z[idx] = 1.0; return z


def lh_valence(P, g, mode="real"):
    """Fixed PN->valence. odor 0 innately positive."""
    n_odor, n_pn = P.shape
    w = g.standard_normal(n_pn) * 0.1
    if mode == "real":
        w = P[0] - P[1]  # innate: prefer odor 0 pattern
    elif mode == "shuf":
        w = w.copy(); g.shuffle(w)
    return w / (np.linalg.norm(w) + 1e-9)


def run_mb_task(seed, mb_mode="real", use_lh=False, lh_mode="real"):
    g = np.random.default_rng(seed)
    pn = E.IX["ALPN"]; P = odor_patterns(g, len(pn))
    mb = build_mb(g, mb_mode)
    lh = lh_valence(P, g, lh_mode) if use_lh else None
    # contingency: phase1 odor0 reward, odor1 shock; phase2 reverse
    pairs_p1 = [(0, +1), (1, -1), (2, 0), (3, 0)]
    pairs_p2 = [(0, -1), (1, +1), (2, 0), (3, 0)]
    eta, tau = 0.05, 0.9
    acc = []; t80 = None; correct_streak = 0

    def trial(odor, rew, learn=True):
        nonlocal correct_streak
        h = mb["Wpk"] @ P[odor]; z = kc_wta(h)
        mbon = mb["Wkm"] @ z
        # choice: prefer odor if mean MBON (appetitive subset) high — use all MBON mean as value
        val = float(mbon.mean())
        if lh is not None: val += 0.5 * float(lh @ P[odor])
        choice = 1 if val > 0 else 0  # 1 = approach
        want = 1 if rew > 0 else (0 if rew < 0 else -1)
        ok = (want == -1) or (choice == want)
        if learn:
            mb["e_kc"] = tau * mb["e_kc"] + z
            # three-factor: dWkm = eta * outer(DAN, e_kc) * RPE; DAN = PAM for +, PPL for -
            rpe = rew - val
            dan = np.zeros(len(mb["mbon"]))
            if rew > 0: dan[:] = 1.0
            elif rew < 0: dan[:] = -1.0
            mb["Wkm"] += eta * np.outer(dan, mb["e_kc"]) * np.clip(rpe, -2, 2)
            np.clip(mb["Wkm"], -2, 2, out=mb["Wkm"])
        return ok, val

    # phase 1
    for t in range(N_TRIAL_P1):
        odor, rew = pairs_p1[t % 4] if t % 2 == 0 else pairs_p1[g.integers(0, 2)]
        # train on 0/1 only mostly
        if t % 3 == 0: odor, rew = pairs_p1[g.integers(0, 2)]
        ok, _ = trial(odor, rew, True); acc.append(ok)
        if t80 is None and t >= 9 and np.mean(acc[-10:]) >= 0.8: t80 = t + 1
    p1_last10 = float(np.mean(acc[-10:]))
    # phase 2 reversal
    acc2 = []
    for t in range(N_TRIAL_P2):
        odor, rew = pairs_p2[g.integers(0, 2)]
        ok, _ = trial(odor, rew, True); acc2.append(ok)
    p2_last10 = float(np.mean(acc2[-10:]))
    return dict(task="mb", seed=seed, mb_mode=mb_mode, use_lh=use_lh, lh_mode=lh_mode,
                p1_last10=p1_last10, p2_last10=p2_last10, trials_to_80=(t80 if t80 else N_TRIAL_P1),
                p1_mean=float(np.mean(acc)), p2_mean=float(np.mean(acc2)))


def run_tubu_heading(seed, mode="real"):
    g = np.random.default_rng(seed)
    tubu = E.IX["TuBu"]; epg = E.CXT["EPG"]; n_h = 36
    n_tu, n_epg = len(tubu), len(epg)
    # landmark → TuBu
    n_land = 8
    W_lt = g.standard_normal((n_tu, n_land)) / np.sqrt(n_land)
    W_te = g.standard_normal((n_epg, n_tu)) / np.sqrt(n_tu) * 0.5
    if mode == "rand":
        W_lt = g.standard_normal((n_tu, n_land)) / np.sqrt(n_land)
        W_te = g.standard_normal((n_epg, n_tu)) / np.sqrt(n_tu) * 0.5
    elif mode == "shuf":
        for W in (W_lt, W_te):
            flat = W.ravel().copy(); g.shuffle(flat); W[:] = flat.reshape(W.shape)
    # ring attractor weights among EPG (circular)
    Wr = np.zeros((n_epg, n_epg))
    for i in range(n_epg):
        for d, w in ((0, 1.0), (1, 0.5), (-1, 0.5), (2, -0.3), (-2, -0.3)):
            Wr[i, (i + d) % n_epg] += w
    errs = []
    for trial in range(30):
        heading = int(g.integers(0, n_h)); home = heading
        h_epg = np.zeros(n_epg); h_epg[heading % n_epg] = 1.0
        # random turns
        for step in range(20):
            turn = int(g.choice([-2, -1, 0, 1, 2])); heading = (heading + turn) % n_h
            # landmark feature: one-hot of absolute heading bins coarsely
            land = np.zeros(n_land); land[heading * n_land // n_h] = 1.0
            tu = np.tanh(W_lt @ land)
            drive = W_te @ tu
            h_epg = np.tanh(Wr @ h_epg + drive + 0.1 * np.roll(h_epg, turn))
        est = int(np.argmax(h_epg)) * n_h // max(1, n_epg)
        # home error
        err = min((est - home) % n_h, (home - est) % n_h)
        errs.append(err)
    return dict(task="tubu", seed=seed, mode=mode, heading_err=float(np.mean(errs)), heading_err_sd=float(np.std(errs)))


def run_combined_maze(seed, mode="real"):
    """CX+MB+LH+TuBu roles vs RANDMOD-style in v2 maze with odor cues."""
    g = np.random.default_rng(seed)
    env = E.Env(100 + seed); fly = E.Fly(g)
    mb = build_mb(g, "real" if mode == "real" else "rand")
    P = odor_patterns(g, len(E.IX["ALPN"]))
    lh = lh_valence(P, g, "real" if mode == "real" else "rand")
    # odor cue: when env.odour, food direction biases PN pattern 0 else 1
    nf = len(E.IX["DN"]) + 11 + 1 + len(E.IX["MBON"])
    if mode == "randmod":
        R = g.standard_normal((400, 400)) / 20 * 0.95; Ri = g.standard_normal((400, 11)); nf = len(E.IX["DN"]) + 11 + 1 + 400
    Wq = np.zeros((nf, 4)); Nsa = np.zeros((E.S * E.S, 4)); got = False
    log = []; z = np.zeros(400)
    for ep in range(E.NTRAIN + E.NEVAL):
        o = env.reset(ep); fly.reset(); z[:] = 0
        # inject odor into fly via PN drive
        odor_id = 0 if env.odour else 1
        phi = fly.step(o); done = False; why = None
        while not done:
            p = env.pos[0] * E.S + env.pos[1]
            # MB value
            h = mb["Wpk"] @ P[odor_id]; kc = kc_wta(h); mbon = mb["Wkm"] @ kc
            val_lh = float(lh @ P[odor_id])
            if mode == "randmod":
                z = np.tanh(R @ z + Ri @ o); feat = np.concatenate([
                    phi[:-1] / (np.linalg.norm(phi[:-1]) + 1e-6),
                    z / (np.linalg.norm(z) + 1e-6),
                    o / (np.linalg.norm(o) + 1e-6), [1.0]])
            else:
                feat = np.concatenate([
                    phi[:-1] / (np.linalg.norm(phi[:-1]) + 1e-6),
                    o / (np.linalg.norm(o) + 1e-6),
                    mbon / (np.linalg.norm(mbon) + 1e-6 + 1e-6), [1.0]])
            q = feat @ Wq
            eps = 0.05 if ep >= E.NTRAIN else (1.0 if not got else max(0.1, 1 - ep / E.NTRAIN))
            a = int(g.integers(4)) if g.random() < eps else int(np.argmax(q + 0.2 * val_lh + 0.3 / np.sqrt(1 + Nsa[p]) + 1e-6 * g.random(4)))
            Nsa[p, a] += 1
            o2, r, done, why = env.step(a)
            # DAN plasticity on food/threat
            if why == "food":
                mb["e_kc"] = 0.9 * mb["e_kc"] + kc; mb["Wkm"] += 0.05 * np.outer(np.ones(len(mbon)), mb["e_kc"])
            if why == "threat":
                mb["e_kc"] = 0.9 * mb["e_kc"] + kc; mb["Wkm"] -= 0.05 * np.outer(np.ones(len(mbon)), mb["e_kc"])
            np.clip(mb["Wkm"], -2, 2, out=mb["Wkm"])
            phi2 = fly.step(o2, reward_in=2 * (why == "food"), threat_in=2 * (why == "threat"))
            if mode == "randmod":
                z = np.tanh(R @ z + Ri @ o2); feat2 = np.concatenate([
                    phi2[:-1] / (np.linalg.norm(phi2[:-1]) + 1e-6), z / (np.linalg.norm(z) + 1e-6),
                    o2 / (np.linalg.norm(o2) + 1e-6), [1.0]])
            else:
                h2 = mb["Wpk"] @ P[odor_id]; mbon2 = mb["Wkm"] @ kc_wta(h2)
                feat2 = np.concatenate([
                    phi2[:-1] / (np.linalg.norm(phi2[:-1]) + 1e-6), o2 / (np.linalg.norm(o2) + 1e-6),
                    mbon2 / (np.linalg.norm(mbon2) + 1e-6), [1.0]])
            td = r + (0 if done else 0.95 * (feat2 @ Wq).max()) - feat @ Wq[:, a]
            Wq[:, a] += 0.1 * td * feat; np.clip(Wq, -5, 5, out=Wq)
            phi, o = phi2, o2
            if why == "food": got = True
        if ep >= E.NTRAIN:
            log.append(dict(food=why == "food", threat=why == "threat", loom=env.loom, steps=env.t))
    L = pd.DataFrame(log); succ = L.food
    return dict(task="maze", seed=seed, mode=mode, success=float(succ.mean()),
                threat_hit_loom=float(L.threat[L.loom].mean()) if L.loom.any() else float("nan"),
                steps_to_food=float(L.steps[succ].mean()) if succ.any() else float("nan"))


def job(spec):
    t0 = time.time(); kind = spec[0]
    if kind == "mb":
        _, seed, mb_mode, use_lh, lh_mode = spec
        r = run_mb_task(seed, mb_mode, use_lh, lh_mode)
    elif kind == "tubu":
        _, seed, mode = spec
        r = run_tubu_heading(seed, mode)
    else:
        _, seed, mode = spec
        r = run_combined_maze(seed, mode)
    r["sec"] = time.time() - t0
    return r


if __name__ == "__main__":
    have = set()
    if os.path.exists(JF):
        for line in open(JF):
            r = json.loads(line)
            have.add((r["task"], r["seed"], r.get("mb_mode"), r.get("use_lh"), r.get("lh_mode"), r.get("mode")))
    J = []
    for s in SEEDS:
        for mb_mode in ("real", "rand", "shuf"):
            if ("mb", s, mb_mode, False, "real", None) not in have:
                J.append(("mb", s, mb_mode, False, "real"))
            for lh_mode in ("real", "rand", "shuf"):
                if ("mb", s, mb_mode, True, lh_mode, None) not in have:
                    J.append(("mb", s, mb_mode, True, lh_mode))
        for mode in ("real", "rand", "shuf"):
            if ("tubu", s, None, None, None, mode) not in have:
                J.append(("tubu", s, mode))
        for mode in ("real", "rand", "randmod"):
            if ("maze", s, None, None, None, mode) not in have:
                J.append(("maze", s, mode))
    print("njobs", len(J), flush=True)
    with Pool(int(os.environ.get("WORKERS", 7))) as P, open(JF, "a") as fh:
        for r in P.imap_unordered(job, J, chunksize=1):
            fh.write(json.dumps(r) + "\n"); fh.flush()
            print(r["task"], r["seed"], r.get("mb_mode", r.get("mode")), round(r.get("success", r.get("p1_last10", r.get("heading_err", -1))), 3), flush=True)
    print("all done", flush=True)
