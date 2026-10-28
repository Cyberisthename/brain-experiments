"""EMPN gridworld experiment. See PREREGISTRATION_empn.md. Resumable: out/runs.jsonl"""
import os, sys, json, time
for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]: os.environ[v] = "1"
import numpy as np, scipy.sparse as sp
from multiprocessing import Pool
H = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(H, "out"); JF = os.path.join(OUT, "runs.jsonl")
W0 = sp.load_npz(os.path.join(OUT, "circuit.npz")).tocsr() * 0.95; META = np.load(os.path.join(OUT, "circuit_meta.npz"), allow_pickle=True)
GRP = META["group"]; NN = W0.shape[0]; IX = {k: np.where(GRP == k)[0] for k in np.unique(GRP)}
AGENTS = ["FLY", "EMPN", "RANDMOD", "EMPN-noTBU", "EMPN-noVTM", "EMPN-noCFS", "EMPN-TBUstatic", "EMPN-FBT", "EMPN-RCX", "EMPN+SCNC", "EMPN+SCNC-Vmean", "EMPN+SCNC-Vshuf", "EMPN+RAND500", "EMPN+SCNC+CTB", "EMPN+SCNC+MGM", "EMPN+SCNC+CVB", "EMPN+SCNC+ALL3", "EMPN+SCNC+ALL3RAND"]; SEEDS = list(range(8))
NTRAIN, NEVAL, MAXT, S = 200, 60, 80, 7
MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]
ZONE = {(2, 3), (3, 3), (4, 3)}

class Env:
    def __init__(s, seed):
        s.g = np.random.default_rng(seed); s.foods = [(0, int(s.g.integers(4, 7))), (0, int(s.g.integers(0, 3)))]
        s.wall = {(3, c) for c in range(S)} - {(3, 3), (3, 6), (3, 0)}  # gaps: threat corridor c3, safe c6, door c0
    def reset(s, ep):
        s.food = s.foods[(ep // 10) % 2]; s.odour = ep % 10 < 3; s.loom = s.g.random() < 0.5; s.door = s.g.random() < 0.3
        s.pos = (6, 3); s.t = 0; s.entered_zone = False; s.short = False; return s.obs()
    def blocked(s, p):
        r, c = p
        return not (0 <= r < S and 0 <= c < S) or p in s.wall or (p == (3, 0) and not s.door)
    def obs(s):
        r, c = s.pos; o = [float(s.blocked((r + dr, c + dc))) for dr, dc in MOVES]
        o += [np.sign(s.food[0] - r) * s.odour, np.sign(s.food[1] - c) * s.odour]
        near = min(abs(r - zr) + abs(c - zc) for zr, zc in ZONE) <= 2
        o += [float(s.loom and near), float(s.door and abs(r - 3) + abs(c - 0) <= 2), r / 6, c / 6, 1.0]
        return np.array(o)
    def step(s, a):
        p = (s.pos[0] + MOVES[a][0], s.pos[1] + MOVES[a][1]); s.t += 1
        if not s.blocked(p): s.pos = p
        if s.pos in ZONE: s.short = True; s.entered_zone = True
        if s.pos in ZONE and s.loom: return s.obs(), -1.0, True, "threat"
        if s.pos == s.food: return s.obs(), 1.0, True, "food"
        return s.obs(), -0.01, s.t >= MAXT, "timeout" if s.t >= MAXT else None

class Fly:
    def __init__(s, g):
        s.Wi = sp.lil_matrix((NN, 11)); orn, vpn = IX["ORN"], IX["VPN"]
        for ch in range(11):
            tgt = g.choice(orn, 100, replace=False) if ch in (4, 5) else g.choice(vpn, 100, replace=False)
            s.Wi[tgt, ch] = g.uniform(0.5, 1.0, 100)
        s.Wi = s.Wi.tocsr() * 3.0; s.b = g.uniform(-.1, .1, NN); s.mu = 0; s.sd = 1
        C = []
        for _ in range(150):
            s.reset(); o = np.r_[(g.random(4) < .3).astype(float), g.integers(-1, 2, 2), g.random(2) < .3, g.random(2), 1.0]; C.append(s.step(o)[:-1])
        C = np.array(C); s.mu = C.mean(0); s.sd = C.std(0) + 1e-4
    def reset(s): s.x = np.zeros(NN)
    def step(s, o, reward_in=0.0, threat_in=0.0, extra=None):
        u = s.Wi @ o + s.b
        if extra is not None: u = u + extra
        if reward_in: u[IX["PAM"]] += reward_in
        if threat_in: u[IX["PPL"]] += threat_in; u[IX["OA"]] += threat_in
        for _ in range(6): s.x = 0.5 * s.x + 0.5 * np.tanh(W0 @ s.x + u)
        f = np.clip((s.x[IX["DN"]] - s.mu) / s.sd, -5, 5) / np.sqrt(len(IX["DN"]))
        return np.concatenate([f, [1.0]])

import pandas as _pd
_ct = _pd.read_csv("/workspace/flybrain/data/ann.tsv", sep="\t", low_memory=False, usecols=["root_id", "cell_type"]).drop_duplicates("root_id").set_index("root_id").cell_type.astype(str)
_cts = _ct.reindex(META["root_id"]).values
CXT = {"EPG": np.where(np.isin(_cts, ["EPG", "PEG", "EPGt"]) & (GRP == "CX"))[0], "FBT": np.where(_pd.Series(_cts).str.match(r"^FB\d").values & (GRP == "CX"))[0], "RCX": IX["CX"]}
class TBU:  # plastic hippocampus: EC(pos) -> DG kWTA -> CA3 (plastic mossy + plastic sparse recurrents, covariance+BCM) -> CA1 (contrastive delta) -> CX targets
    def __init__(s, g, plastic=True, tgt="EPG", eta=15.0, ecode=None):
        s.g, s.pl, s.eta = g, plastic, eta; P = S * S
        s.ec = np.eye(P) if ecode is None else ecode  # (P, nEC): EC code per place; SCNC layer-1 output when present
        s.EC_DG = ((g.random((2000, s.ec.shape[1])) < 0.25) * g.lognormal(0, .5, (2000, s.ec.shape[1]))) @ s.ec.T
        s.MF = (sp.random(400, 2000, density=0.005, random_state=int(g.integers(1e9))).toarray() * 4.0); s.MFm = s.MF > 0
        s.R = np.zeros((400, 400)); s.Rm = np.zeros((400, 400), bool)
        for i in range(400): s.Rm[i, g.choice(400, 40, replace=False)] = True
        s.R[s.Rm] = g.lognormal(0, .5, s.Rm.sum()) * 0.01
        s.Wec1 = (g.standard_normal((400, s.ec.shape[1])) / 2 * (1 if ecode is None else np.sqrt(P / s.ec.shape[1]))) @ s.ec.T; s.W31 = g.standard_normal((400, 400)) * s.Rm / 20
        s.n0 = {k: np.abs(getattr(s, k)).sum(1) + 1e-9 for k in ("R", "MF", "W31")}; s.theta = np.full(400, 0.05)
        s.init = {k: getattr(s, k).copy() for k in ("R", "MF", "W31")}; s.tmpl = np.tanh(s.Wec1.T); s.out = np.zeros((NN, 400))
        tp = CXT[tgt]
        for i in range(400): s.out[g.choice(tp, 30, replace=len(tp) < 30), i] = g.choice([-1, 1], 30) * 0.2
        s.out = sp.csr_matrix(s.out)
    def dg(s, p, kk=40):
        h = s.EC_DG[:, p]; k = np.zeros(2000); k[np.argsort(-h)[:kk]] = 1; return k
    def kw(s, h, k=20):
        z = np.zeros(400); z[np.argsort(-h)[:k]] = 1; return z
    def ca3(s, p): return s.kw(s.MF @ s.dg(p))
    def ca1(s, c3): return np.tanh(s.W31 @ c3)
    def norm(s):
        for k in ("R", "MF", "W31"):
            W = np.clip(getattr(s, k), -3, 3); W *= (s.n0[k] / (np.abs(W).sum(1) + 1e-9))[:, None]; setattr(s, k, W)
    def store(s, path, w=1.0, ks=None):
        if not s.pl: return
        ws = np.broadcast_to(np.asarray(w, float), (len(path) - 1,)); ks = [40] * (len(path) - 1) if ks is None else ks
        for a, b, w, kk in zip(path[:-1], path[1:], ws, ks):
            if a == b or w == 0: continue
            pre = s.ca3(a); d = s.dg(b, kk); post = np.tanh(s.MF @ d / 4 + s.R @ pre)
            s.theta = 0.9 * s.theta + 0.1 * post ** 2; phi = post * (post - s.theta)
            s.R += s.eta * w * np.outer(phi, pre - pre.mean()) * s.Rm / 400
            s.MF += s.eta * w * np.outer(phi, d - d.mean()) * s.MFm / 2000
            c3 = s.ca3(b); plus = np.tanh(s.W31 @ c3 + s.Wec1[:, b]); minus = s.ca1(c3)
            s.W31 += s.eta * w * np.outer(plus - minus, c3) * s.Rm / 400
        s.norm()
    def step_pos(s, p):
        c = s.kw(s.R @ s.ca3(p)); a1 = s.ca1(c); return int(np.argmax(s.tmpl @ a1)), a1
    def recall(s, p, n=30):
        out = []; cur = p
        if not s.pl: return out
        for _ in range(n):
            nxt, _ = s.step_pos(cur)
            if nxt in out or nxt == cur: break
            out.append(nxt); cur = nxt
        return out
    def cx_drive(s, p): return s.out @ s.step_pos(p)[1]

class SCNC:  # 400 fixed sparse random ELM filter (obs -> EC 200) + 100-unit valence engine -> scalar V (three-factor gate)
    def __init__(s, g):
        s.A = sp.random(400, 11, density=0.3, random_state=int(g.integers(1e9))).toarray() * 3 - 0.0; s.A[s.A != 0] = g.standard_normal((s.A != 0).sum()) * 1.5
        s.E = (g.random((200, 400)) < 0.1) * g.standard_normal((200, 400)) / np.sqrt(40); s.Wv = g.standard_normal((100, 5)); s.bv = g.standard_normal(100) * 0.1
        s.c = g.standard_normal(100) / 10
        base = np.tanh(s.Wv @ np.zeros(5) + s.bv); ev = [np.tanh(s.Wv @ e + s.bv) for e in np.eye(5)[[2, 3]]]
        s.v0 = s.c @ base; s.vs = max(abs(s.c @ e - s.v0) for e in ev) + 1e-9
    def ec(s, o): return np.tanh(s.E @ np.tanh(s.A @ o))
    def V(s, t_since_food, dist_threat, food, threat, loom):
        f = np.array([t_since_food / MAXT * 0.05, 0.05 * max(0, 2 - dist_threat) * loom, food, threat, 0.0])
        return float(abs(s.c @ np.tanh(s.Wv @ f + s.bv) - s.v0) / s.vs)  # ~0 normally, ~1 at food or danger

def cfs(env, model, goal_val, risk, loom_now, Hh=12):
    V = np.zeros(S * S)
    for _ in range(Hh):
        Q = goal_val[model] - 2.0 * risk[model] * loom_now + 0.9 * V[model]; V = np.maximum(goal_val, Q.max(1))
    p = env.pos[0] * S + env.pos[1]; q = goal_val[model[p]] - 2.0 * risk[model[p]] * loom_now + 0.9 * V[model[p]]
    return q - q.mean()

def run(agent, seed):
    g = np.random.default_rng(seed); env = Env(100 + seed); fly = Fly(g); nf = len(IX["DN"]) + 1
    use_tbu = agent.startswith("EMPN") and agent != "EMPN-noTBU"; use_vtm = agent.startswith("EMPN") and agent != "EMPN-noVTM"
    use_cfs = agent.startswith("EMPN") and agent != "EMPN-noCFS"; empn = agent.startswith("EMPN")
    scnc = SCNC(g) if "SCNC" in agent else None; ecode = None
    if scnc is not None:
        e0 = Env(100 + seed); e0.reset(0); e0.odour = e0.loom = e0.door = False; C = []
        for p in range(S * S): e0.pos = divmod(p, S); C.append(scnc.ec(e0.obs()))
        ecode = np.array(C)
    tbu = TBU(g, plastic=agent != "EMPN-TBUstatic", tgt={"EMPN-FBT": "FBT", "EMPN-RCX": "RCX"}.get(agent, "EPG"), ecode=ecode) if use_tbu else None
    allV = []; tsf = 0; visits = np.zeros(S * S); habit = np.zeros((S * S, 4))
    ctb = "+CTB" in agent or "+ALL3" in agent; mgm = "+MGM" in agent or "+ALL3" in agent; cvb = "+CVB" in agent or "+ALL3" in agent; rnd = agent.endswith("ALL3RAND")
    if rnd:  # same-size random modules: CTB 250, MGM 100, CVB 150 random tanh units driven by obs/SCNC features, outputs of matched scale
        Rc = g.standard_normal((250, 11)); Rco = g.standard_normal((4, 250)) / np.sqrt(250)
        Rm = g.standard_normal((100, 11)); Rmo = g.standard_normal(100) / np.sqrt(100)
        Rv = g.standard_normal((150, 4)); Rvo = g.standard_normal((4, 150)) / np.sqrt(150)
    R0 = tbu.R.copy() if tbu is not None else None; mgm_hi = []
    if agent == "EMPN+RAND500":  # same-size random add-on: 500 random recurrent tanh units feeding the Q readout
        R5 = g.standard_normal((500, 500)) / np.sqrt(500) * 0.95 * (g.random((500, 500)) < 0.1) / np.sqrt(0.1); Ri5 = g.standard_normal((500, 11)); nf += 500
    if agent == "RANDMOD":  # same param count as TBU's CA3 store (400x400)
        R = g.standard_normal((400, 400)) / 20 * 0.95; Ri = g.standard_normal((400, 11)); nf += 400
    Wq = np.zeros((nf, 4)); model = np.tile(np.arange(S * S)[:, None], (1, 4)); seen = np.zeros((S * S, 4), bool); risk = np.zeros(S * S); log = []
    for ep in range(NTRAIN + NEVAL):
        o = env.reset(ep); fly.reset(); phi = fly.step(o); z = np.zeros(400) if agent == "RANDMOD" else None
        if z is not None: z = np.tanh(R @ z + Ri @ o); phi = np.concatenate([phi, z])
        z5 = None
        if agent == "EMPN+RAND500": z5 = np.tanh(Ri5 @ o); phi = np.concatenate([phi, z5])
        Vs = []
        path = [6 * S + 3]; vread = []; done = False; recalled = tbu.recall(path[0]) if use_tbu else []
        while not done:
            p = env.pos[0] * S + env.pos[1]; q = phi @ Wq; bias = np.zeros(4)
            if empn:
                goal = np.zeros(S * S)
                if recalled: goal[recalled] = np.linspace(0.1, 0.5, len(recalled)); goal[recalled[-1]] = 1.0
                if env.odour: pass  # odour handled by fly Q-readout
                if use_cfs: bias = cfs(env, model, goal, risk if use_vtm else risk * 0, o[6])
                elif recalled:
                    rp = tbu.recall(p, 1)
                    if rp: bias = np.array([float(model[p, a] == rp[0]) for a in range(4)])
            if empn and tbu is not None and (ctb or cvb):
                opens = [a_ for a_ in range(4) if o[a_] == 0]
                if ctb and len(opens) >= 3:  # CTB: pause, simulate 3 steps per action through CA3 recall (no env step taken)
                    if rnd: sc_ = Rco @ np.tanh(Rc @ o)
                    else:
                        sc_ = np.zeros(4)
                        for a_ in range(4):
                            if o[a_]: sc_[a_] = -1; continue
                            cur = model[p, a_]; traj = [cur]
                            for _ in range(2): cur = tbu.step_pos(cur)[0]; traj.append(cur)
                            sc_[a_] = max(goal[traj].max() if recalled else 0, 0) - 1.0 * o[6] * risk[traj].max() - 0.5 * float(all(model[traj[0], b_] == traj[0] for b_ in range(4) if seen[traj[0], b_]) and seen[traj[0]].sum() >= 3)
                    bias = sc_ - sc_.mean() if not use_cfs else 0.5 * bias + (sc_ - sc_.mean())
                if cvb:  # curiosity / fear / urgency vector -> bias toward novel moves when safe, habitual home route when threatened
                    fear = float(o[6]); cur_ = (1 - fear) * float(scnc.V(tsf, 6, 0, 0, 0) < 0.1); urg = min(1.0, tsf / MAXT)
                    if rnd: cb = Rvo @ np.tanh(Rv @ np.array([fear, cur_, urg, 1.0]))
                    else:
                        nov = np.array([1.0 / (1 + visits[model[p, a_]]) if seen[p, a_] else 1.0 for a_ in range(4)]); hb = habit[p] / (habit[p].sum() + 1e-9)
                        cb = cur_ * (1 - urg) * (nov - nov.mean()) + fear * (hb - hb.mean()) * 2
                    bias = bias + cb
            visits[p] += 1
            eps = max(0.05, 0.3 * (1 - ep / NTRAIN)); a = int(g.integers(4)) if g.random() < eps else int(np.argmax(q + 1.0 * bias + 1e-6 * g.random(4)))
            habit[p, a] += 1; o = None
            o2, r, done, why = env.step(a); p2 = env.pos[0] * S + env.pos[1]; model[p, a] = p2; seen[p, a] = True; path.append(p2)
            phi2 = fly.step(o2, reward_in=2.0 * (why == "food"), threat_in=2.0 * (why == "threat"), extra=tbu.cx_drive(p2) if tbu is not None and tbu.pl else None)
            vread.append(fly.x[IX["PAM"]].mean() - 0.5 * (fly.x[IX["PPL"]].mean() + fly.x[IX["OA"]].mean()))
            if z is not None: z = np.tanh(R @ z + Ri @ o2); phi2 = np.concatenate([phi2, z])
            if z5 is not None: z5 = np.tanh(R5 @ z5 + Ri5 @ o2); phi2 = np.concatenate([phi2, z5])
            tsf += 1
            if scnc is not None:
                dthr = min(abs(env.pos[0] - zr) + abs(env.pos[1] - zc) for zr, zc in ZONE)
                Vs.append(scnc.V(tsf, dthr, float(why == "food"), float(why == "threat"), float(env.loom)))
            if why == "food": tsf = 0
            target = r + (0 if done else 0.95 * (phi2 @ Wq).max())
            Wq[:, a] += 0.05 / (phi @ phi) * (target - phi @ Wq[:, a]) * phi; phi = phi2; o = o2
        if empn:
            val = vread[-1] - np.mean(vread[:-1]) if len(vread) > 1 else 0.0  # VTM: valence from PAM vs PPL/OA response at episode end
            if scnc is not None:  # three-factor: eta * eligibility(trace 0.9) of V; SCNC replaces the episode-end VTM gate for storage
                Vs = np.array(Vs); allV.extend(Vs.tolist())
                if agent == "EMPN+SCNC-Vmean": Vs = np.full(len(Vs), np.mean(allV))
                if agent == "EMPN+SCNC-Vshuf": Vs = g.permutation(np.array(allV))[:len(Vs)]
                el = np.zeros(len(Vs)); acc = 0.0
                for i in range(len(Vs) - 1, -1, -1): acc = Vs[i] + 0.9 * acc; el[i] = acc
                ks = None
                if mgm:  # novelty-gated DG sparsity: 0.5% (10/2000) familiar, 5% (100/2000) on novelty or reward/danger
                    if rnd: ks = [100 if Rmo @ np.tanh(Rm @ np.r_[g.standard_normal(9), v_, 1.0]) > 0 else 10 for v_ in Vs]
                    else: ks = [100 if (visits[b_] <= 1 or v_ > 0.3) else 10 for b_, v_ in zip(path[1:], Vs)]
                    mgm_hi.extend([k_ == 100 for k_ in ks])
                tbu.store(path, np.clip(el, 0, 3), ks)
                if why == "threat" and val < 0: risk[p2] += 1
            elif use_vtm:
                if why == "food" and val > 0 and tbu is not None: tbu.store(path)
                if why == "threat" and val < 0: risk[p2] += 1
            elif use_tbu: tbu.store(path)  # no valence gating: store every episode
        if ep >= NTRAIN:
            log.append(dict(food=why == "food", threat=why == "threat", steps=env.t, odour=env.odour, loom=env.loom, door=env.door, zone=env.entered_zone))
    import pandas as pd
    L = pd.DataFrame(log); succ = L.food
    clutter = float(np.abs(tbu.R - R0).sum() / np.abs(R0).sum()) if tbu is not None else float("nan")
    act = float((np.abs(tbu.R - R0).sum(1) > 1e-3).mean()) if tbu is not None else float("nan")
    return dict(agent=agent, seed=seed, ca3_clutter_relL1=clutter, ca3_rows_changed=act, mgm_high_k_frac=float(np.mean(mgm_hi)) if mgm_hi else float("nan"), success=succ.mean(), success_no_odour=succ[~L.odour].mean(), steps_to_food=L.steps[succ].mean() if succ.any() else float("nan"),
                threat_hit_loom=L.threat[L.loom].mean(), short_route_when_safe=L.zone[~L.loom].mean(), short_route_when_loom=L.zone[L.loom].mean())

def job(a):
    t = time.time(); r = run(*a); r["sec"] = time.time() - t; return r
if __name__ == "__main__":
    if len(sys.argv) > 1: print(job((sys.argv[1], 0))); sys.exit()
    have = {(r["agent"], r["seed"]) for r in map(json.loads, open(JF))} if os.path.exists(JF) else set()
    J = [(a, s) for s in SEEDS for a in AGENTS if (a, s) not in have]
    with Pool(int(os.environ.get("WORKERS", 3))) as P, open(JF, "a") as fh:
        for r in P.imap_unordered(job, J, chunksize=1): fh.write(json.dumps(r) + "\n"); fh.flush()
    print("all done")
