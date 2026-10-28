"""EMPN v2b: organ-fix rerun. See PREREGISTRATION_empn_v2_organ_fix.md.

Root causes (v2 final + seed 900/901):
1. CFS scored risk-only (goals gated) and its bias was added 1:1 to Q, so it
   overrode food-seeking; EMPN never found food (success 0% vs noCFS 50%).
2. +TR *replaced* TD on Wq and applied unnormalized episodic e·RPE → Wq norm
   exploded (~10–14 vs healthy ~0.7); FLY+TR 2.5% vs FLY 46%.

Shared fixes (ALL arms):
- CFS: expected-reward − risk with reward weight; fall back to policy when WM
  uncertain (seen[p]<2); tanh-scale bias; score = q + 0.3*conf*bias; fall back if CFS flips policy under low WM conf; decay rew (food non-stationary).
- TR: KEEP online TD always; +TR adds small L2-normalized episodic bonus
  (eta_tr=0.02); do not replace TD; no SCNC V in plasticity RPE.
- CTB gated by food stores≥5; NTRAIN×2.
"""
import os, sys, json, time
for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[v] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from multiprocessing import Pool
import empn as E

H = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(H, "out")
os.makedirs(OUT, exist_ok=True)
JF = os.path.join(OUT, "runs_v2b.jsonl")
CF = os.path.join(OUT, "curves_v2b.jsonl")
NTRAIN_MUL = 2  # shared budget boost
BASE = ["FLY", "RANDMOD", "EMPN", "EMPN+SCNC", "EMPN+SCNC+CTB", "EMPN+SCNC+MGM",
        "EMPN+SCNC+CVB", "EMPN+SCNC+ALL3", "EMPN+SCNC+ALL3RAND", "EMPN-noCFS"]
SEEDS = list(range(8))
TAU_GRID = [0.85, 0.9, 0.95]
TUNE = [900, 901]


def _feat(phi_dn, o, z=None):
    """SHARED readout features for ALL arms.
    Diagnosis: DN-only phi is nearly flat (std~0.004) and after a single
    L2-norm drowns the Markov obs (11-d). Block-normalize DN, optional
    reservoir z, and obs separately so each block keeps equal voice, then
    concat a bias. Applied identically to FLY/RANDMOD/EMPN*.
    """
    dn = np.asarray(phi_dn[:-1], float)
    parts = [dn / (float(np.linalg.norm(dn)) + 1e-6)]
    if z is not None:
        zz = np.asarray(z, float)
        parts.append(zz / (float(np.linalg.norm(zz)) + 1e-6))
    oo = np.asarray(o, float)
    parts.append(oo / (float(np.linalg.norm(oo)) + 1e-6))
    parts.append(np.array([1.0]))
    return np.concatenate(parts)



def cfs_er(env, model, seen, rew, risk, loom_now, Hh=12, w_r=1.0, w_risk=1.0):
    """Expected reward minus risk. Fall back (zeros) when WM uncertain at pos.

    Unseen transitions are NOT treated as self-loops (that inflated V and made
    CFS prefer staying / weird detours over food). Only seen (s,a) bootstrap.
    """
    p = env.pos[0] * E.S + env.pos[1]
    # Fall back to policy unless we have a usable local model + some reward signal
    if int(seen[p].sum()) < 2 or float(rew.max()) <= 0.0:
        return np.zeros(4)
    n = E.S * E.S
    V = rew.copy()
    loom = float(loom_now)
    for _ in range(Hh):
        Q = np.full((n, 4), -1e9)
        for a in range(4):
            sm = seen[:, a]
            if not sm.any():
                continue
            nxt = model[:, a]
            Qa = np.zeros(n)
            Qa[sm] = (w_r * rew[nxt[sm]] - w_risk * risk[nxt[sm]] * loom
                      + 0.9 * V[nxt[sm]])
            # unseen actions: no opinion (leave at -1e9 so max ignores them)
            Q[:, a] = Qa
            Q[~sm, a] = -1e9
        # V update: max over seen actions; default to rew
        have = seen.any(1)
        mx = Q.max(1)
        V = rew.copy()
        V[have] = np.maximum(rew[have], mx[have])
    # action values at p from seen model only
    q = np.zeros(4)
    for a in range(4):
        if not seen[p, a]:
            q[a] = np.nan
            continue
        nxt = int(model[p, a])
        q[a] = w_r * rew[nxt] - w_risk * risk[nxt] * loom + 0.9 * V[nxt]
    if np.all(np.isnan(q)):
        return np.zeros(4)
    fill = float(np.nanmean(q))
    q = np.where(np.isnan(q), fill, q)
    bias = q - q.mean()
    return np.tanh(bias)


def run_fixed(agent, seed, use_tr=False, tau=0.9, diagnose=False):
    g = np.random.default_rng(seed)
    env = E.Env(100 + seed)
    fly = E.Fly(g)
    # v2b: 2x episode budget (shared)
    _NTRAIN = int(getattr(E, 'NTRAIN', 200) * int(globals().get('NTRAIN_MUL', 2)))
    _NEVAL = int(getattr(E, 'NEVAL', 60))
    nf = len(E.IX["DN"]) + 11 + 1  # DN + obs + bias; block-normed in _feat
    base = agent.replace("+TR", "")
    use_tr = use_tr or agent.endswith("+TR")
    use_tbu = base.startswith("EMPN") and base != "EMPN-noTBU"
    use_vtm = base.startswith("EMPN") and base != "EMPN-noVTM"
    use_cfs = base.startswith("EMPN") and base != "EMPN-noCFS"
    empn = base.startswith("EMPN")
    scnc = E.SCNC(g) if "SCNC" in base else None
    ecode = None
    if scnc is not None:
        e0 = E.Env(100 + seed); e0.reset(0)
        e0.odour = e0.loom = e0.door = False
        C = []
        for p in range(E.S * E.S):
            e0.pos = divmod(p, E.S); C.append(scnc.ec(e0.obs()))
        ecode = np.array(C)
    tbu = E.TBU(g, plastic=True, tgt="EPG", ecode=ecode) if use_tbu else None
    ctb = "+CTB" in base or "+ALL3" in base
    mgm = "+MGM" in base or "+ALL3" in base
    cvb = "+CVB" in base or "+ALL3" in base
    rnd = base.endswith("ALL3RAND")
    visits = np.zeros(E.S * E.S)
    habit = np.zeros((E.S * E.S, 4))
    Nsa = np.zeros((E.S * E.S, 4))  # shared count-based exploration
    if rnd:
        Rc = g.standard_normal((250, 11)); Rco = g.standard_normal((4, 250)) / np.sqrt(250)
        Rm = g.standard_normal((100, 11)); Rmo = g.standard_normal(100) / np.sqrt(100)
        Rv = g.standard_normal((150, 4)); Rvo = g.standard_normal((4, 150)) / np.sqrt(150)
    if base == "RANDMOD":
        R = g.standard_normal((400, 400)) / 20 * 0.95
        Ri = g.standard_normal((400, 11)); nf += 400
    Wq = np.zeros((nf, 4))
    e_q = np.zeros_like(Wq)
    Rbar = 0.0
    alpha = 0.1  # slightly higher than v1 0.05; shared
    eps0 = 1.0
    beta_c = 0.3  # count bonus — shared for all arms
    model = np.tile(np.arange(E.S * E.S)[:, None], (1, 4))
    seen = np.zeros((E.S * E.S, 4), bool)
    risk = np.zeros(E.S * E.S)
    rew = np.zeros(E.S * E.S)  # expected-reward table for CFS
    log = []; curve = []; dW_hist = []
    R0 = tbu.R.copy() if tbu is not None else None
    mgm_hi = []; tsf = 0; allV = []; got_food_ever = False; n_food_stores = 0

    for ep in range(_NTRAIN + _NEVAL):
        o = env.reset(ep); fly.reset()
        phi = fly.step(o)
        z = np.zeros(400) if base == "RANDMOD" else None
        if z is not None:
            z = np.tanh(R @ z + Ri @ o)
        # food location alternates every 10 eps — decay stale ER/risk so CFS cannot lock onto the previous food
        rew *= 0.9
        risk *= 0.95
        path = [6 * E.S + 3]; vread = []; done = False
        recalled = tbu.recall(path[0]) if use_tbu else []
        ep_ret = 0.0; e_q *= 0.0
        # TBU eligibility buffers (accumulated over episode when +TR)
        if tbu is not None and use_tr:
            e_R = np.zeros_like(tbu.R); e_MF = np.zeros_like(tbu.MF); e_W31 = np.zeros_like(tbu.W31)
            e_out = np.zeros(tbu.out.shape) if hasattr(tbu.out, 'shape') else None
        Vs = []; phis = []; acts = []; steps_phi = []
        while not done:
            p = env.pos[0] * E.S + env.pos[1]
            pn = _feat(phi, o, z)
            q = pn @ Wq
            bias = np.zeros(4)
            if empn:
                goal = np.zeros(E.S * E.S)
                # Recall goals only after enough food stores (legacy channel for CTB)
                if recalled and n_food_stores >= 5:
                    gscale = min(1.0, (n_food_stores - 4) / 10.0)
                    goal[recalled] = gscale * np.linspace(0.1, 0.5, len(recalled)); goal[recalled[-1]] = gscale
                if use_cfs:
                    # v2b CFS FIX: expected-reward − risk (w_r=1), fall back if WM uncertain
                    bias = cfs_er(env, model, seen, rew,
                                  risk if use_vtm else risk * 0, o[6], w_r=1.0, w_risk=1.0)
            if empn and tbu is not None and (ctb or cvb):
                opens = [a_ for a_ in range(4) if o[a_] == 0]
                # v2b FIX: gate CTB until enough food stores (WM proxy)
                if ctb and len(opens) >= 3 and n_food_stores >= 5:
                    if rnd:
                        sc_ = Rco @ np.tanh(Rc @ o)
                    else:
                        sc_ = np.zeros(4)
                        for a_ in range(4):
                            if o[a_]: sc_[a_] = -1; continue
                            cur = model[p, a_]; traj = [cur]
                            for _ in range(2):
                                cur = tbu.step_pos(cur)[0]; traj.append(cur)
                            sc_[a_] = max(goal[traj].max() if recalled else 0, 0) - 1.0 * o[6] * risk[traj].max()
                    bias = sc_ - sc_.mean() if not use_cfs else 0.5 * bias + (sc_ - sc_.mean())
                if cvb:
                    fear = float(o[6]); cur_ = (1 - fear) * float((scnc.V(tsf, 6, 0, 0, 0) if scnc else 0) < 0.1)
                    urg = min(1.0, tsf / E.MAXT)
                    if rnd:
                        cb = Rvo @ np.tanh(Rv @ np.array([fear, cur_, urg, 1.0]))
                    else:
                        nov = np.array([1.0 / (1 + visits[model[p, a_]]) if seen[p, a_] else 1.0 for a_ in range(4)])
                        hb = habit[p] / (habit[p].sum() + 1e-9)
                        cb = cur_ * (1 - urg) * (nov - nov.mean()) + fear * (hb - hb.mean()) * 2
                    bias = bias + cb
            visits[p] += 1
            # SHARED: higher exploration + count bonus (all arms)
            if ep >= _NTRAIN:
                eps = 0.05
            else:
                eps = 1.0 if not got_food_ever else max(0.1, 1.0 * (1 - ep / max(1, _NTRAIN)))
            bonus = beta_c / np.sqrt(1.0 + Nsa[p])
            # v2b: CFS assist-only — never flip policy argmax; single RNG draw (aligned w/ noCFS)
            base_score = q + bonus
            a_pol = int(np.argmax(base_score))
            conf = float(seen[p].mean())
            tie = 1e-6 * g.random(4)
            score = base_score + (0.3 * conf) * bias + tie
            if int(np.argmax(score)) != a_pol:
                score = base_score + tie  # fall back to policy, same tie noise
            a = int(g.integers(4)) if g.random() < eps else int(np.argmax(score))
            habit[p, a] += 1; Nsa[p, a] += 1
            o2, r, done, why = env.step(a)
            p2 = env.pos[0] * E.S + env.pos[1]
            model[p, a] = p2; seen[p, a] = True; path.append(p2); ep_ret += r
            phi2 = fly.step(o2, reward_in=2.0 * (why == "food"), threat_in=2.0 * (why == "threat"),
                            extra=tbu.cx_drive(p2) if tbu is not None and tbu.pl else None)
            vread.append(fly.x[E.IX["PAM"]].mean() - 0.5 * (fly.x[E.IX["PPL"]].mean() + fly.x[E.IX["OA"]].mean()))
            if z is not None:
                z = np.tanh(R @ z + Ri @ o2)
            pn2 = _feat(phi2, o2, z)

            # v2b TR FIX: ALWAYS keep online TD (was replaced by pure episodic TR → Wq explode)
            target = r + (0 if done else 0.95 * (pn2 @ Wq).max())
            td = target - pn @ Wq[:, a]
            Wq[:, a] += alpha * td * pn / (eps0 + 1.0)
            np.clip(Wq, -5, 5, out=Wq)
            if use_tr:
                e_q *= tau
                e_q[:, a] += pn

            # TBU eligibility bump each transition when +TR
            if tbu is not None and use_tr and p != p2:
                pre = tbu.ca3(p); d = tbu.dg(p2); post = np.tanh(tbu.MF @ d / 4 + tbu.R @ pre)
                e_R *= tau; e_MF *= tau; e_W31 *= tau
                e_R += np.outer(post, pre) * tbu.Rm
                e_MF += np.outer(post, d) * tbu.MFm
                c3 = tbu.ca3(p2); plus = np.tanh(tbu.W31 @ c3 + tbu.Wec1[:, p2]); minus = tbu.ca1(c3)
                e_W31 += np.outer(plus - minus, c3) * tbu.Rm

            steps_phi.append(float(np.linalg.norm(phi)))
            phi = phi2; o = o2; tsf += 1
            if scnc is not None:
                dthr = min(abs(env.pos[0] - zr) + abs(env.pos[1] - zc) for zr, zc in E.ZONE)
                Vs.append(scnc.V(tsf, dthr, float(why == "food"), float(why == "threat"), float(env.loom)))
            if why == "food":
                tsf = 0; got_food_ever = True; n_food_stores += 1
                # credit recent tail of path more (food switched every 10 eps)
                for i, pp in enumerate(path):
                    w = 0.05 + 0.25 * (i / max(1, len(path) - 1))
                    rew[pp] = min(1.0, 0.85 * rew[pp] + w)
                rew[p2] = min(1.0, rew[p2] + 0.5)
            if why == "threat":
                risk[p2] += 1.0

        # RPE baseline
        Rbar = 0.95 * Rbar + 0.05 * ep_ret
        rpe = ep_ret - Rbar
        # episodic three-factor: dW = eta * e * (R - Rbar)
        if use_tr:
            # v2b: small L2-normalized episodic bonus ON TOP of TD (eta_tr << alpha)
            en = float(np.linalg.norm(e_q)) + 1e-8
            eta_tr = 0.02
            Wq += eta_tr * np.clip(rpe, -2, 2) * (e_q / en)
            np.clip(Wq, -5, 5, out=Wq)

        # TBU store / three-factor
        if empn and tbu is not None:
            if use_tr:
                # apply e * RPE (SCNC V as broadcast when present)
                # v2b FIX: do NOT mix SCNC V into plasticity return RPE
                mod = np.clip(rpe, -1, 1)
                tbu.R += tbu.eta * 0.01 * mod * e_R / 400
                tbu.MF += tbu.eta * 0.01 * mod * e_MF / 2000
                tbu.W31 += tbu.eta * 0.01 * mod * e_W31 / 400
                tbu.norm()
                # also classic store with RPE weight so path structure lands
                if scnc is not None and len(Vs):
                    Vs_a = np.array(Vs); el = np.zeros(len(Vs_a)); acc = 0.0
                    for i in range(len(Vs_a) - 1, -1, -1):
                        acc = Vs_a[i] + 0.9 * acc; el[i] = acc
                    el = el * np.sign(mod + 1e-9) * min(abs(mod), 1.0)
                    ks = None
                    if mgm:
                        if rnd:
                            ks = [100 if Rmo @ np.tanh(Rm @ np.r_[g.standard_normal(9), v_, 1.0]) > 0 else 10 for v_ in Vs_a]
                        else:
                            ks = [100 if (visits[b_] <= 1 or v_ > 0.3) else 10 for b_, v_ in zip(path[1:], Vs_a)]
                        mgm_hi.extend([k_ == 100 for k_ in ks])
                    tbu.store(path, np.clip(el, 0, 3), ks)
                    if why == "threat": risk[p2] += 1
                else:
                    tbu.store(path, w=max(0.1, abs(mod)))
                    if why == "threat": risk[p2] += 1
            else:
                # no-trace: same store logic as v1 but with RPE-weighted fallback
                if scnc is not None and len(Vs):
                    Vs_a = np.array(Vs); el = np.zeros(len(Vs_a)); acc = 0.0
                    for i in range(len(Vs_a) - 1, -1, -1):
                        acc = Vs_a[i] + 0.9 * acc; el[i] = acc
                    ks = None
                    if mgm:
                        if rnd:
                            ks = [100 if Rmo @ np.tanh(Rm @ np.r_[g.standard_normal(9), v_, 1.0]) > 0 else 10 for v_ in Vs_a]
                        else:
                            ks = [100 if (visits[b_] <= 1 or v_ > 0.3) else 10 for b_, v_ in zip(path[1:], Vs_a)]
                        mgm_hi.extend([k_ == 100 for k_ in ks])
                    tbu.store(path, np.clip(el, 0, 3), ks)
                    if why == "threat": risk[p2] += 1
                elif use_vtm:
                    val = vread[-1] - np.mean(vread[:-1]) if len(vread) > 1 else 0.0
                    if why == "food" and val > 0: tbu.store(path)
                    if why == "threat" and val < 0: risk[p2] += 1
                elif use_tbu:
                    tbu.store(path)

        if ep < _NTRAIN and ep % 20 == 19:
            curve.append(dict(ep=ep, ret=float(ep_ret), food=bool(why == "food"), rpe=float(rpe),
                              wq_norm=float(np.linalg.norm(Wq)),
                              mean_phi_norm=float(np.mean(steps_phi)) if steps_phi else 0.0))
        if ep >= _NTRAIN:
            log.append(dict(food=why == "food", threat=why == "threat", steps=env.t,
                            odour=env.odour, loom=env.loom, door=env.door, zone=env.entered_zone))
        dW_hist.append(float(np.linalg.norm(Wq)))

    L = pd.DataFrame(log); succ = L.food
    clutter = float(np.abs(tbu.R - R0).sum() / (np.abs(R0).sum() + 1e-9)) if tbu is not None else float("nan")
    out = dict(
        agent=agent, seed=seed, use_tr=use_tr, tau=tau,
        success=float(succ.mean()),
        success_no_odour=float(succ[~L.odour].mean()) if (~L.odour).any() else float("nan"),
        steps_to_food=float(L.steps[succ].mean()) if succ.any() else float("nan"),
        threat_hit_loom=float(L.threat[L.loom].mean()) if L.loom.any() else float("nan"),
        short_route_when_safe=float(L.zone[~L.loom].mean()) if (~L.loom).any() else float("nan"),
        short_route_when_loom=float(L.zone[L.loom].mean()) if L.loom.any() else float("nan"),
        ca3_clutter_relL1=clutter,
        wq_final=float(dW_hist[-1]), wq_start=float(dW_hist[0]),
        train_food_rate=float(np.mean([c["food"] for c in curve])) if curve else float("nan"),
        train_food_last40=float(np.mean([c["food"] for c in curve[-2:]])) if len(curve) >= 2 else float("nan"),
    )
    out["curve"] = curve  # always keep learning curve for RESULTS_empn_v2.md
    return out


def job(a):
    agent, seed, use_tr, tau = a
    t = time.time()
    r = run_fixed(agent, seed, use_tr, tau)
    r["sec"] = time.time() - t
    return r

# CF set above for v2b


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "diag":
        for a in ["FLY", "RANDMOD", "EMPN"]:
            r = run_fixed(a, 0, False, diagnose=True)
            open(os.path.join(OUT, f"diag_{a}.json"), "w").write(
                json.dumps({k: r[k] for k in r if k != "curve"}) + "\n")
            open(os.path.join(OUT, f"diag_{a}_curve.json"), "w").write(json.dumps(r["curve"]))
            print(a, "success", round(r["success"], 3), "train_food", r["train_food_rate"],
                  "wq", round(r["wq_start"], 4), "->", round(r["wq_final"], 4),
                  "last40", r["train_food_last40"], flush=True)
        sys.exit()
    if cmd == "tune":
        # tau on TUNE seeds for RANDMOD+TR (has learning signal; EMPN was flat at 0)
        rows = []
        for tau in TAU_GRID:
            for s in TUNE:
                r = run_fixed("RANDMOD", s, True, tau)
                rows.append(dict(tau=tau, seed=s, success=r["success"], train_food=r["train_food_rate"]))
                print("tune", tau, s, "succ", round(r["success"], 3), "train_food", r["train_food_rate"], flush=True)
        pd.DataFrame(rows).to_json(os.path.join(OUT, "tau_tune.json"), orient="records")
        print("tune done", flush=True); sys.exit()
    # full run
    agents = list(BASE) + [a + "+TR" for a in BASE]
    have = set()
    if os.path.exists(JF):
        for line in open(JF):
            r = json.loads(line); have.add((r["agent"], r["seed"]))
    tau = 0.9
    tp = os.path.join(OUT, "tau_tune.json")
    if os.path.exists(tp):
        T = pd.read_json(tp)
        if len(T):
            tau = float(T.groupby("tau").success.mean().idxmax())
    print("using tau", tau, "jobs left", flush=True)
    J = [(a, s, a.endswith("+TR"), tau) for s in SEEDS for a in agents if (a, s) not in have]
    print("njobs", len(J), flush=True)
    with Pool(int(os.environ.get("WORKERS", 7))) as P, open(JF, "a") as fh, open(CF, "a") as fc:
        for r in P.imap_unordered(job, J, chunksize=1):
            curve = r.pop("curve", [])
            fh.write(json.dumps(r) + "\n"); fh.flush()
            fc.write(json.dumps(dict(agent=r["agent"], seed=r["seed"], use_tr=r["use_tr"], tau=r["tau"], curve=curve)) + "\n"); fc.flush()
            print(r["agent"], r["seed"], round(r["success"], 3), "train_food", round(r.get("train_food_rate", -1), 3), flush=True)
    open(os.path.join(OUT, "v2b.done"), "w").write("ok\n")
    print("all done", flush=True)
