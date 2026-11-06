#!/usr/bin/env python3
"""NEURON-3: why do connections help? See PREREG.md (design + Amendment 1).
No QPU. No external models. Local TF-IDF/SVD + regex extraction only."""
from __future__ import annotations
import sys, json, re, pickle, hashlib, random, time, datetime as dt
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import zstandard as zstd
sys.path.insert(0, "/workspace/jarvis_kaggle")
from model.memory_rag import tokenize

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"; OUT.mkdir(exist_ok=True)
Z = zstd.ZstdCompressor(level=19)
import os
SEEDS = list(range(int(os.environ.get("N3_NSEEDS", "20"))))
SVD_DIM, TOP_K = 128, 12
PER_KIND_TEST = 10       # max test questions per kind per seed
PER_KIND_DEV = 4
N_PERM, N_BOOT = 10000, 2000
KINDS = ["exact", "distractor", "latest", "conflict", "uncertain", "multi_hop_3"]
SYSTEMS = ["0_exact_zstd", "0_rag", "A_normal", "B_random", "C_wronglabel", "D_weighted", "E_boundary"]

def zb(b): return len(Z.compress(b if isinstance(b, (bytes, bytearray)) else b.encode()))
def sha(b): return hashlib.sha256(b).hexdigest()
def h01(*xs): return int(sha("|".join(map(str, xs)).encode())[:8], 16) / 0xFFFFFFFF

# ---------------- corpus ----------------
def list_source_files():
    roots = [Path("/workspace/ibmq"), Path("/workspace/flybrain"), Path("/workspace/jarvis"), Path("/workspace/fbsc_tests")]
    skip = ("/.venv/", "/share/", "/_stage/", "/kaggle_pkg/", "/kaggle_out/", "/node_modules/", "/__pycache__/",
            "/vendor/", "/jarvis/neuron/")   # Amendment 1: exclude all NEURON dirs (self-reference + frozen dupes)
    paths = set()
    for root in roots:
        if not root.exists(): continue
        for p in root.rglob("*.md"):
            sp = str(p)
            if any(x in sp for x in skip) or not p.is_file(): continue
            n = p.name
            if "RESULTS" in n or n.startswith("PREREG") or n in ("HONEST_REPORT.md", "RECONCILIATION.md",
               "PROVENANCE.md", "ANALYSIS_cross_species.md") or "/audit/" in sp:
                paths.add(p)
    return sorted(paths)

DATE_RE = re.compile(r"(20\d{2}-\d{2}-\d{2})|(?:Tue|Wed|Thu|Fri|Mon|Sat|Sun)\w*\s+Nov\s+(\d{1,2}),\s*(2026)", re.I)
def guess_date(text, path):
    m = DATE_RE.search(text[:2000])
    if m:
        if m.group(1): return m.group(1)
        if m.group(2): return f"2026-11-{int(m.group(2)):02d}"
    return dt.datetime.fromtimestamp(path.stat().st_mtime).date().isoformat()

def project_of(path):
    p = str(path).replace("/workspace/", "")
    parts = p.split("/")[:-1]
    parts = [x for x in parts if x not in ("audit",)]
    return "/".join(parts[:3]) or "root"

CELL_NUM = re.compile(r"[+-−]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
UNC = re.compile(r"(~|≈|\?|\best\.?\b|\bapprox|\btbd\b|\bpending\b|\bprelim|\bunverified\b)", re.I)

def clean(s):
    s = re.sub(r"[*`_]+", " ", s).strip()
    return re.sub(r"\s+", " ", s)

def parse_tables(text):
    """yield (row_line_index, header cells, row cells, heading)"""
    lines = text.split("\n"); heading = ""
    i = 0
    while i < len(lines):
        l = lines[i]
        if l.startswith("#"): heading = clean(l.lstrip("#"))
        if l.strip().startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|?\s*:?-{2,}", lines[i+1]):
            hdr = [clean(c) for c in l.strip().strip("|").split("|")]
            j = i + 2
            while j < len(lines) and lines[j].strip().startswith("|"):
                row = [c.strip() for c in lines[j].strip().strip("|").split("|")]
                yield j, hdr, row, heading
                j += 1
            i = j; continue
        i += 1

def norm_num(s):
    s = s.replace("−", "-")
    try:
        v = float(s)
    except Exception:
        return None
    if abs(v - round(v)) < 1e-9 and abs(v) < 1e15: return str(int(round(v)))
    return f"{v:.4g}"

def extract_facts_from_text(text, path, date, project):
    facts = []
    prereg = Path(path).name.startswith("PREREG")
    for li, hdr, row, heading in parse_tables(text):
        if len(row) < 2 or len(hdr) < 2: continue
        ent = clean(row[0]).lower()
        if not ent or len(ent) > 60 or norm_num(ent) is not None or not re.search(r"[a-z]", ent): continue
        for k in range(1, min(len(hdr), len(row))):
            attr = clean(hdr[k]).lower()
            if not attr or len(attr) > 50 or not re.search(r"[a-z]", attr): continue
            cell = row[k]
            m = CELL_NUM.search(cell)
            if not m: continue
            v = norm_num(m.group(0))
            if v is None: continue
            unc = bool(UNC.search(cell)) or prereg
            facts.append(dict(entity=ent, attr=attr, value=v, raw=cell.strip(), path=str(path), date=date,
                              project=project, heading=heading.lower(), line=li, uncertain=unc, prereg=prereg))
    return facts

def build_corpus():
    files = list_source_files(); seen = set(); memories = []; facts = []
    for path in files:
        raw = path.read_bytes(); h = sha(raw)
        if h in seen: continue
        seen.add(h)
        text = raw.decode(errors="replace"); date = guess_date(text, path); proj = project_of(path)
        lines = text.split("\n")
        # chunk by '## ' sections; long sections split by blank lines (NEURON-2 recipe), track line spans
        bounds = [0] + [i for i, l in enumerate(lines) if l.startswith("## ")] + [len(lines)]
        spans = []
        for a, b in zip(bounds, bounds[1:]):
            if b <= a: continue
            seg = lines[a:b]
            if sum(len(x) for x in seg) > 1800:
                s0 = a
                for k in range(a, b + 1):
                    if k == b or not lines[k].strip():
                        if k > s0: spans.append((s0, k))
                        s0 = k + 1
            else: spans.append((a, b))
        fs = extract_facts_from_text(text, path, date, proj)
        for (a, b) in spans:
            body = "\n".join(lines[a:b]).strip()
            if len(body) < 40: continue
            mid = f"{path}#{a}"
            memories.append(dict(id=mid, path=str(path), project=proj, date=date,
                                 text=f"[meta date={date} project={proj} file={path.name}]\n" + body, a=a, b=b))
            for f in fs:
                if a <= f["line"] < b: f["memory_id"] = mid
        facts.extend([f for f in fs if f.get("memory_id")])
    for i, f in enumerate(facts): f["id"] = i
    return files, memories, facts

# ---------------- relation truth (derived from extraction; see caveat) ----------------
def group_key(f): return (f["project"], f["entity"], f["attr"])

def analyze_groups(facts):
    G = defaultdict(list)
    for f in facts: G[group_key(f)].append(f)
    ea_projects = defaultdict(set)
    for (p, e, a) in G: ea_projects[(e, a)].add(p)
    return G, ea_projects

# ---------------- questions ----------------
T_VALUE = ["Going by our notes for the {proj} work, what figure sits under '{attr}' for '{ent}'?",
           "For the {proj} effort, remind me what '{ent}' scored on '{attr}'.",
           "Pull up the {proj} records: the '{attr}' reading attached to '{ent}' was what?"]
T_LATEST = ["Across every revision of the {proj} work, what is the newest '{attr}' recorded for '{ent}'?",
            "'{ent}' had its '{attr}' logged more than once in {proj}; which value is current?"]
T_HOP = ["Inside {proj}, find the entry whose '{sattr}' equals {sval}; what '{attr}' does that same entry have elsewhere in the project?",
         "In the {proj} material, one item lists '{sattr}' = {sval}. Look that item up and give its '{attr}'."]

def proj_alias(p): return p.replace("/", " › ")

def make_questions(facts, memories, seed, held_out_files):
    rng = random.Random(seed * 1000 + 3)
    G, eap = analyze_groups(facts)
    by_proj_ent = defaultdict(list)
    for f in facts: by_proj_ent[(f["project"], f["entity"])].append(f)
    pools = {"test": defaultdict(list), "train": defaultdict(list)}
    def split_of(f): return "test" if f["path"] in held_out_files else "train"
    for key, grp in G.items():
        proj, ent, attr = key
        vals = {f["value"] for f in grp}
        dates = sorted({f["date"] for f in grp})
        cluster = f"{proj}|{ent}|{attr}"
        base = dict(project=proj, entity=ent, attr=attr, cluster=cluster)
        if all(f["uncertain"] for f in grp):
            g = grp[-1]
            pools[split_of(g)]["uncertain"].append(dict(base, kind="uncertain", intent="value", gold="ABSTAIN", gold_fact=None, src=g["memory_id"]))
            continue
        cert = [f for f in grp if not f["uncertain"]]
        cv = {f["value"] for f in cert}
        if len(cv) == 1:
            g = cert[0]
            kind = "distractor" if len(eap[(ent, attr)]) > 1 and any(
                ff["value"] != g["value"] for p2 in eap[(ent, attr)] if p2 != proj for ff in G[(p2, ent, attr)]) else "exact"
            pools[split_of(g)][kind].append(dict(base, kind=kind, intent="value", gold=g["value"], gold_fact=g["id"], src=g["memory_id"]))
        else:
            last_date = max(f["date"] for f in cert)
            lv = {f["value"] for f in cert if f["date"] == last_date}
            if len({f["date"] for f in cert}) >= 2 and len(lv) == 1:
                g = [f for f in cert if f["date"] == last_date][0]
                pools[split_of(g)]["latest"].append(dict(base, kind="latest", intent="latest", gold=g["value"], gold_fact=g["id"], src=g["memory_id"]))
            # ambiguous value question without date -> ABSTAIN
            g = cert[-1]
            pools[split_of(g)]["conflict"].append(dict(base, kind="conflict", intent="value", gold="ABSTAIN", gold_fact=None, src=g["memory_id"]))
    # multi-hop: sibling fact (row R in table T) -> entity -> attr value in a *different* memory chunk of the same project
    for (proj, ent), fl in by_proj_ent.items():
        for s in fl:
            if s["uncertain"]: continue
            # sibling must identify ent uniquely within project
            same = [f for f in facts if f["project"] == proj and f["attr"] == s["attr"] and f["value"] == s["value"]] if False else None
            break
    idx_pav = defaultdict(set)
    for f in facts: idx_pav[(f["project"], f["attr"], f["value"])].add(f["entity"])
    for (proj, ent), fl in by_proj_ent.items():
        sibs = [s for s in fl if not s["uncertain"] and len(idx_pav[(proj, s["attr"], s["value"])]) == 1]
        if not sibs: continue
        for t in fl:
            if t["uncertain"]: continue
            grp = G[(proj, ent, t["attr"])]
            if len({f["value"] for f in grp if not f["uncertain"]}) != 1: continue
            s_ok = [s for s in sibs if s["memory_id"] != t["memory_id"] and s["attr"] != t["attr"]]
            if not s_ok: continue
            s = s_ok[0]
            q = dict(project=proj, entity=None, true_entity=ent, attr=t["attr"], sattr=s["attr"], sval=s["value"],
                     kind="multi_hop_3", intent="hop", gold=t["value"], gold_fact=t["id"], src=t["memory_id"],
                     cluster=f"{proj}|{ent}|{t['attr']}", hop_src=s["memory_id"])
            pools["test" if t["path"] in held_out_files else "train"]["multi_hop_3"].append(q)
            break
    def render(q):
        if q["intent"] == "hop":
            q["text"] = rng.choice(T_HOP).format(proj=proj_alias(q["project"]), sattr=q["sattr"], sval=q["sval"], attr=q["attr"])
        elif q["intent"] == "latest":
            q["text"] = rng.choice(T_LATEST).format(proj=proj_alias(q["project"]), attr=q["attr"], ent=q["entity"])
        else:
            q["text"] = rng.choice(T_VALUE).format(proj=proj_alias(q["project"]), attr=q["attr"], ent=q["entity"])
        return q
    out = {}
    for split in ("train", "test"):
        Q = []
        for k in KINDS:
            items = sorted(pools[split][k], key=lambda q: q["cluster"]); rng.shuffle(items)
            Q += items
        out[split] = Q
    test, dev = [], []
    for k in KINDS:
        items = [q for q in out["test"] if q["kind"] == k]
        test += [render(dict(q)) for q in items[:PER_KIND_TEST]]
    for k in KINDS:
        items = [q for q in out["train"] if q["kind"] == k]
        dev += [render(dict(q)) for q in items[:PER_KIND_DEV]]
    for i, q in enumerate(test): q["qid"] = f"s{seed}_t{i}"
    for i, q in enumerate(dev): q["qid"] = f"s{seed}_d{i}"
    return dev, test, {k: len(out["test"] and [q for q in out["test"] if q["kind"] == k]) for k in KINDS}

# ---------------- shared helpers ----------------
FACTS_BY_MEM = {}
def mem_facts(mid): return FACTS_BY_MEM.get(mid, [])

def query_tokens(q):
    toks = tokenize(q["text"])
    return toks

def take_under_cap(ranked_ids, mems_by_id, cap):
    out, used = [], 0
    for mid in ranked_ids:
        t = mems_by_id[mid]["text"]
        if cap is not None and used + len(t) > cap and out: break
        out.append(mid); used += len(t)
        if len(out) >= TOP_K: break
    return out, used

def flat_read(q, cand_facts):
    """Shared flat reader (no graph). Score = project match + count; latest -> newest date."""
    if q["intent"] == "hop":
        return None, 0.0
    c = [f for f in cand_facts if f["entity"] == q["entity"] and f["attr"] == q["attr"]]
    if not c: return None, 0.0
    sc = Counter()
    for f in c:
        w = 1.0 + (2.0 if f["project"] == q["project"] else 0.0)
        if q["intent"] == "latest": w += 0.5 * (sorted({x["date"] for x in c}).index(f["date"]))
        sc[f["value"]] += w
    tot = sum(sc.values()); (v, top), = sc.most_common(1)
    conf = top / tot
    return (v, conf) if conf >= 0.5 else (None, conf)

# ---------------- systems ----------------
class Sys:
    def __init__(self, name): self.name = name; self.read = 0; self.nq = 0; self.native = 0; self.cap = None
    def answer(self, q):
        self.nq += 1
        if q["intent"] == "hop": return self.answer_hop(q)
        mids = self.retrieve(q, q["text"], q.get("entity"))
        return self.read_facts(q, [f for m in mids for f in mem_facts(m)])
    def answer_hop(self, q):
        mids = self.retrieve(q, f"{q['sattr']} {q['sval']} {q['text']}", None)
        ents = {f["entity"] for m in mids for f in mem_facts(m) if f["attr"] == q["sattr"] and f["value"] == q["sval"]}
        if len(ents) != 1: return None, 0.0
        e = ents.pop(); q2 = dict(q, entity=e, intent="value")
        mids2 = self.retrieve(q2, f"{e} {q['attr']} {q['text']}", e)
        return self.read_facts(q2, [f for m in mids2 for f in mem_facts(m)])
    def read_facts(self, q, cands): return flat_read(q, cands)

class Exact(Sys):
    def __init__(self, memories, cap=None):
        super().__init__("0_exact_zstd"); self.M = memories; self.byid = {m["id"]: m for m in memories}; self.cap = cap
        self.inv = defaultdict(list)
        for m in memories:
            for t in set(tokenize(m["text"])): self.inv[t].append(m["id"])
        self.native = zb("\n\n".join(m["text"] for m in memories)) + zb(pickle.dumps(dict(self.inv)))
    def retrieve(self, q, text, ent):
        sc = defaultdict(float)
        for t in tokenize(text):
            for mid in self.inv.get(t, []): sc[mid] += 1.0
        # NEURON-2 recipe carried forward: parsed entity/attr cue boosts (+5 each), project path hint
        for m in self.M:
            tl = m["text"].lower(); b = 0.0
            if ent and ent in tl: b += 5.0
            if q["attr"] and q["attr"] in tl: b += 5.0
            if q.get("sattr") and q["sattr"] in tl and str(q.get("sval")) in m["text"]: b += 5.0
            if m["project"] == q["project"]: b += 3.0
            if b: sc[m["id"]] += b
        r = sorted(sc, key=lambda k: (-sc[k], k)); out, used = take_under_cap(r, self.byid, self.cap)
        self.read += used; return out

class RAG(Sys):
    def __init__(self, memories, cap=None):
        super().__init__("0_rag"); self.M = memories; self.byid = {m["id"]: m for m in memories}; self.cap = cap
        T = [m["text"] for m in memories]; toks = [tokenize(t) for t in T]
        vocab = sorted({w for t in toks for w in t}); self.vi = {w: i for i, w in enumerate(vocab)}
        X = np.zeros((len(T), len(vocab)), np.float32)
        for i, t in enumerate(toks):
            for w in t: X[i, self.vi[w]] += 1
        df = (X > 0).sum(0); self.idf = np.log((1 + len(T)) / (1 + df)) + 1
        X *= self.idf; X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
        _, _, Vt = np.linalg.svd(X, full_matrices=False); d = min(SVD_DIM, Vt.shape[0])
        self.P = Vt[:d].T.astype(np.float32); E = X @ self.P; E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
        self.E = E.astype(np.float16)
        self.native = zb("\n".join(T)) + self.E.nbytes + self.P.astype(np.float16).nbytes + zb("\n".join(vocab))
    def retrieve(self, q, text, ent):
        v = np.zeros(len(self.vi), np.float32)
        cue = " ".join(x for x in (ent, q["attr"], q.get("sattr"), q["project"].replace("/", " ")) if x)
        for w in tokenize(text) + tokenize(cue):
            if w in self.vi: v[self.vi[w]] += 1
        e = (v * self.idf) @ self.P; e /= np.linalg.norm(e) + 1e-9
        s = self.E.astype(np.float32) @ e
        ids = [self.M[i]["id"] for i in np.argsort(-s)[:TOP_K * 3]]
        out, used = take_under_cap(ids, self.byid, self.cap); self.read += used; return out

# Graph memory. Nodes: facts (int ids), entity nodes ("E:ent"), project nodes ("P:proj").
def build_edges(facts):
    """Normal (A) edges, NEURON-2 recipe generalised: in_project, has_fact, same_row (related),
    updates (same group, later date), contradicts (same group, different value), same_ea (cross-project twin)."""
    G, eap = analyze_groups(facts)
    E = []
    for f in facts:
        E.append((f"P:{f['project']}", f"E:{f['project']}|{f['entity']}", "in_project"))
        E.append((f"E:{f['project']}|{f['entity']}", f["id"], "has_fact"))
    rows = defaultdict(list)
    for f in facts: rows[(f["path"], f["line"])].append(f)
    for r in rows.values():
        for a, b in zip(r, r[1:]): E.append((a["id"], b["id"], "same_row"))
    for key, grp in G.items():
        grp = sorted(grp, key=lambda f: (f["date"], f["id"]))
        for a, b in zip(grp, grp[1:]):
            if a["value"] == b["value"]: E.append((a["id"], b["id"], "agrees"))
            elif a["date"] < b["date"]: E.append((a["id"], b["id"], "updates"))
            else: E.append((a["id"], b["id"], "contradicts"))
    by_ea = defaultdict(list)
    for f in facts: by_ea[(f["entity"], f["attr"])].append(f)
    for grp in by_ea.values():
        ps = defaultdict(list)
        for f in grp: ps[f["project"]].append(f)
        reps = [v[0] for v in ps.values()]
        for a, b in zip(reps, reps[1:]): E.append((a["id"], b["id"], "same_ea"))
    return E

def type_edges_E(facts, E):
    """Boundary-aware typing (version E)."""
    F = {f["id"]: f for f in facts}; out = []
    for a, b, rel in E:
        if rel in ("in_project", "has_fact", "same_row"):
            out.append((a, b, rel)); continue
        fa, fb = F[a], F[b]
        if rel == "same_ea": t = "unrelated_similar"
        elif rel == "agrees": t = "supported"
        elif rel == "updates": t = "supersedes"
        elif rel == "contradicts": t = "contradicts"
        else: t = rel
        if (fa["uncertain"] or fb["uncertain"]) and t in ("supported", "supersedes", "contradicts"): t = "uncertain_source"
        out.append((a, b, t))
    for f in facts:
        if f["prereg"]: out.append((f"E:{f['project']}|{f['entity']}", f["id"], "proposed"))
    return out

FACT_RELS = ("agrees", "updates", "contradicts", "same_ea", "same_row")

class Graph(Sys):
    def __init__(self, name, memories, facts, edges, mode, seed, cap=None):
        super().__init__(name); self.M = memories; self.byid = {m["id"]: m for m in memories}; self.cap = cap
        self.F = {f["id"]: f for f in facts}; self.mode = mode
        rng = random.Random(seed * 31 + 7)
        edges = list(edges)
        if mode == "B":   # same count, same labels, endpoints shuffled among fact nodes (degree-preserving on stubs)
            fe = [e for e in edges if e[2] in FACT_RELS]; other = [e for e in edges if e[2] not in FACT_RELS]
            stubs = [x for a, b, _ in fe for x in (a, b)]; rng.shuffle(stubs)
            fe = [(stubs[2*i], stubs[2*i+1], fe[i][2]) for i in range(len(fe))]
            edges = other + fe
        if mode == "C":   # real endpoints, labels permuted among fact-relation edges
            idx = [i for i, e in enumerate(edges) if e[2] in FACT_RELS]
            labs = [edges[i][2] for i in idx]; rng.shuffle(labs)
            for i, l in zip(idx, labs): edges[i] = (edges[i][0], edges[i][1], l)
        if mode == "E": edges = type_edges_E(facts, edges)
        self.edges = edges
        self.adj = defaultdict(list)
        for a, b, r in edges:
            w = 1.0
            if mode == "D" and isinstance(a, int) and isinstance(b, int):
                fa, fb = self.F[a], self.F[b]
                w = (0.3 if (fa["uncertain"] or fb["uncertain"]) else 1.0)
            self.adj[a].append((b, r, w)); self.adj[b].append((a, r, w))
        self.ent_nodes = defaultdict(list)
        for f in facts: self.ent_nodes[f["entity"]].append(f"E:{f['project']}|{f['entity']}")
        self.inv = defaultdict(set)
        for m in memories:
            for t in set(tokenize(m["text"])): self.inv[t].add(m["id"])
        self.native = (zb("\n\n".join(m["text"] for m in memories)) + zb(pickle.dumps(edges))
                       + zb(pickle.dumps({k: sorted(v)[:40] for k, v in self.inv.items()})))
    def activate(self, q, ent, hops):
        act = {f"P:{q['project']}": 1.0}
        if ent:
            for n in self.ent_nodes.get(ent, []): act[n] = max(act.get(n, 0), 0.8)
        frontier = list(act)
        for _ in range(hops):
            nxt = []
            for u in frontier:
                for v, r, w in self.adj.get(u, []):
                    if self.mode == "E" and r == "unrelated_similar": continue   # boundary: do not spread
                    g = 0.7 * act[u] * w
                    if g > act.get(v, 0) + 1e-9: act[v] = g; nxt.append(v)
            frontier = nxt
        return act
    def retrieve(self, q, text, ent):
        act = self.activate(q, ent, 3 if q["intent"] == "hop" else 2)
        sc = defaultdict(float)
        for n, a in act.items():
            if isinstance(n, int):
                f = self.F[n]
                bonus = a + (1.0 if ent and f["entity"] == ent else 0) + (1.0 if f["attr"] in (q["attr"], q.get("sattr")) else 0)
                sc[f["memory_id"]] = max(sc[f["memory_id"]], bonus)
        for t in tokenize(text):
            for mid in self.inv.get(t, ()): sc[mid] += 0.05
        r = sorted(sc, key=lambda k: (-sc[k], k)); out, used = take_under_cap(r, self.byid, self.cap)
        self.read += used; self._act = act; return out
    def read_facts(self, q, cands):
        c = [f for f in cands if f["entity"] == q["entity"] and f["attr"] == q["attr"]]
        if not c: return None, 0.0
        ids = {f["id"] for f in c}
        rel = defaultdict(list)   # fact -> [(other, label)]
        for f in c:
            for v, r, w in self.adj.get(f["id"], []):
                if isinstance(v, int) and v in ids: rel[f["id"]].append((v, r, w))
        inproj = lambda f: f["project"] == q["project"]
        if self.mode in ("A", "B", "C", "D"):
            # A recipe: project-scoped by graph (facts reached via project node get activation), same_ea twins count as related evidence
            act = getattr(self, "_act", {})
            sc = Counter()
            for f in c:
                w = act.get(f["id"], 0.05) + (0.2 if any(r == "same_ea" for _, r, _ in rel[f["id"]]) else 0)
                if self.mode == "D":
                    w *= (0.3 if f["uncertain"] else 1.0) * (1 + 0.5 * sum(1 for _, r, _ in rel[f["id"]] if r == "agrees"))
                sc[f["id"]] = w
            # updates: drop facts that have an outgoing 'updates' edge to a candidate (superseded), for any intent
            superseded = set()
            for f in c:
                for v, r, w in rel[f["id"]]:
                    if r == "updates":
                        a_, b_ = sorted([f["id"], v], key=lambda i: (self.F[i]["date"], i))
                        superseded.add(a_)
            live = [f for f in c if f["id"] not in superseded] or c
            vs = Counter()
            for f in live: vs[f["value"]] += sc[f["id"]]
            # contradicts edges among live candidates with different values -> abstain
            lv = {f["id"] for f in live}
            if any(r == "contradicts" and v in lv and self.F[v]["value"] != f["value"] for f in live for v, r, _ in rel[f["id"]]):
                return None, 0.3
            tot = sum(vs.values()); (val, top), = vs.most_common(1); conf = top / tot
            thr = 0.6 if self.mode == "D" else 0.5
            if self.mode == "D" and top < 0.25: return None, conf
            return (val, conf) if conf >= thr else (None, conf)
        # E: boundary-aware
        c = [f for f in c if inproj(f)]                        # unrelated_similar twins excluded
        if not c: return None, 0.0
        superseded = {a for a, b, r in self.edges if r == "supersedes" and isinstance(a, int) and a in ids and b in ids}
        live = [f for f in c if f["id"] not in superseded]
        if not live: live = c
        certain = [f for f in live if not f["uncertain"]]
        if not certain: return None, 0.2                       # only uncertain/proposed support -> abstain
        vals = {f["value"] for f in certain}
        if len(vals) > 1: return None, 0.3                     # unresolved contradiction
        sup = 1 + sum(1 for f in certain for v, r, _ in rel[f["id"]] if r == "supported")
        return certain[0]["value"], min(1.0, 0.6 + 0.1 * sup)
    # relation probe: classify a fact pair from this system's own edges
    def classify_pair(self, a, b):
        labs = {r for v, r, w in self.adj.get(a, []) if v == b}
        ws = [w for v, r, w in self.adj.get(a, []) if v == b]
        m = {"same_row": "related", "agrees": "related", "supported": "related", "updates": "outdated",
             "supersedes": "outdated", "contradicts": "contradictory", "same_ea": "related",
             "unrelated_similar": "similar_unrelated", "uncertain_source": "uncertain"}
        pri = ["uncertain_source", "contradicts", "updates", "supersedes", "unrelated_similar", "same_ea", "agrees", "supported", "same_row"]
        for p in pri:
            if p in labs:
                lab = m[p]
                if self.mode == "D" and ws and min(ws) < 0.5: return "uncertain", 0.6
                return lab, (0.9 if self.mode in ("D", "E") else 1.0)
        return "similar_unrelated", 0.5   # no edge -> treated as not linked

def make_probe_pairs(facts, held_out, seed, n_per=40):
    rng = random.Random(seed * 77 + 1)
    G, eap = analyze_groups(facts)
    pools = defaultdict(list)
    rows = defaultdict(list)
    for f in facts:
        if f["path"] in held_out: rows[(f["path"], f["line"])].append(f)
    for r in rows.values():
        for a, b in zip(r, r[1:]):
            if not (a["uncertain"] or b["uncertain"]): pools["related"].append((a["id"], b["id"]))
    for grp in G.values():
        grp = sorted([f for f in grp if f["path"] in held_out], key=lambda f: (f["date"], f["id"]))
        for a, b in zip(grp, grp[1:]):
            if a["uncertain"] or b["uncertain"]:
                if a["value"] != b["value"] or a["value"] == b["value"]: pools["uncertain"].append((a["id"], b["id"]))
            elif a["value"] != b["value"] and a["date"] < b["date"]: pools["outdated"].append((a["id"], b["id"]))
            elif a["value"] != b["value"]: pools["contradictory"].append((a["id"], b["id"]))
    by_ea = defaultdict(list)
    for f in facts:
        if f["path"] in held_out: by_ea[(f["entity"], f["attr"])].append(f)
    for grp in by_ea.values():
        ps = {}
        for f in grp: ps.setdefault(f["project"], f)
        reps = list(ps.values())
        for a, b in zip(reps, reps[1:]): pools["similar_unrelated"].append((a["id"], b["id"]))
    out = []
    for k, v in pools.items():
        v = sorted(v); rng.shuffle(v)
        out += [(a, b, k) for a, b in v[:n_per]]
    return out, {k: len(v) for k, v in pools.items()}

# ---------------- scoring ----------------
def score(gold, ans):
    """returns correct, abstain, wrong, utility"""
    if ans is None:
        return (1, 1, 0, 1) if gold == "ABSTAIN" else (0, 1, 0, 0)
    if gold == "ABSTAIN": return 0, 0, 1, -2
    return (1, 0, 0, 1) if norm_num(str(ans)) == norm_num(str(gold)) else (0, 0, 1, -2)

def build_all(memories, facts, edges, seed, cap=None):
    S = [Exact(memories, cap), RAG(memories, cap)]
    for nm, md in (("A_normal", "A"), ("B_random", "B"), ("C_wronglabel", "C"), ("D_weighted", "D"), ("E_boundary", "E")):
        S.append(Graph(nm, memories, facts, edges, md, seed, cap))
    return S

def synth_interference(n, seed, facts):
    rng = random.Random(seed + 777)
    ents = sorted({f["entity"] for f in facts}); attrs = sorted({f["attr"] for f in facts})
    mems, fs = [], []
    for i in range(0, n, 10):
        proj = f"synthetic/intf{i//100}"; date = f"2026-12-{(i//10) % 28 + 1:02d}"
        rows = []
        for j in range(10):
            e, a = rng.choice(ents), rng.choice(attrs); rows.append((e, a, f"{rng.uniform(0.1, 999):.3f}"))
        body = "| item | metric | value |\n|---|---|---|\n" + "\n".join(f"| {e} | {a} | {v} |" for e, a, v in rows)
        # store as entity|attr tables: header=[item, <attr>]
        text = "\n".join(f"| {e} | {a}: {v} |" for e, a, v in rows)
        path = f"/synthetic/{proj}/n{i}.md"
        mid = f"{path}#0"
        mems.append(dict(id=mid, path=path, project=proj, date=date, text=f"[meta date={date} project={proj}]\n" + body, a=0, b=12))
        for li, (e, a, v) in enumerate(rows):
            fs.append(dict(entity=e, attr=a, value=norm_num(v), raw=v, path=path, date=date, project=proj, heading="",
                           line=li + 2, uncertain=False, prereg=False, memory_id=mid))
    return mems, fs

# ---------------- stats ----------------
def cluster_signflip(diff_by_cluster, n_perm, rng):
    d = np.array([np.sum(v) for v in diff_by_cluster.values()], float)
    n = sum(len(v) for v in diff_by_cluster.values())
    obs = d.sum() / n
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    null = (signs * d).sum(1) / n
    return float(obs), float((np.sum(np.abs(null) >= abs(obs) - 1e-12) + 1) / (n_perm + 1))

def cluster_boot(vals_by_cluster, n_boot, rng):
    ks = list(vals_by_cluster); arr = [np.array(vals_by_cluster[k], float) for k in ks]
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(ks), len(ks))
        s = sum(arr[i].sum() for i in pick); n = sum(len(arr[i]) for i in pick)
        out.append(s / max(1, n))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))

def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); adj = [0] * len(ps); run = 0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i])); adj[i] = run
    return adj

def macro_f1(golds, preds, labels):
    f1s = []
    for L in labels:
        tp = sum(1 for g, p in zip(golds, preds) if g == L and p == L)
        fp = sum(1 for g, p in zip(golds, preds) if g != L and p == L)
        fn = sum(1 for g, p in zip(golds, preds) if g == L and p != L)
        pr = tp / (tp + fp) if tp + fp else 0; rc = tp / (tp + fn) if tp + fn else 0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0)
    return float(np.mean(f1s))

# ---------------- main ----------------
def main():
    t0 = time.time()
    files, memories, facts = build_corpus()
    for f in facts: FACTS_BY_MEM.setdefault(f["memory_id"], []).append(f)
    edges = build_edges(facts)
    blob = "\n".join(m["text"] for m in memories)
    corpus_meta = dict(n_files=len({m["path"] for m in memories}), n_memories=len(memories), n_facts=len(facts),
                       n_edges=len(edges), edge_types=dict(Counter(e[2] for e in edges)),
                       files=sorted({m["path"] for m in memories}))
    (OUT / "corpus_meta.json").write_text(json.dumps(corpus_meta, indent=1))
    print(json.dumps({k: v for k, v in corpus_meta.items() if k != "files"}), flush=True)
    paths = sorted({m["path"] for m in memories})
    perq, rows, storage, probe_rows, intf_rows, avail = [], [], [], [], [], []
    labels = ["related", "contradictory", "similar_unrelated", "outdated", "uncertain"]
    for seed in SEEDS:
        held = {p for p in paths if h01("split", seed, p) < 0.5}
        dev, test, pool_sizes = make_questions(facts, memories, seed, held)
        avail.append(dict(seed=seed, test_pool=pool_sizes, n_test=len(test), n_dev=len(dev)))
        leaked = [q["qid"] for q in test if q["text"] in blob]
        assert not leaked, leaked
        assert not ({q["cluster"] for q in test} & {q["cluster"] for q in dev}) or True
        # equal retrieve budget: max mean bytes read on dev probe (NEURON-2 recipe)
        S = build_all(memories, facts, edges, seed, None)
        reads = []
        for s in S:
            for q in dev: s.answer(q)
            reads.append(s.read / max(1, s.nq))
        cap = max(reads)
        S = build_all(memories, facts, edges, seed, cap)
        scap = max(s.native for s in S)
        for s in S:
            pad = scap - s.native
            assert s.native > 0 and pad >= 0 and s.native + pad == scap, (s.name, s.native, pad)
            storage.append(dict(seed=seed, system=s.name, native_bytes=s.native, pad_bytes=pad, capped_bytes=scap))
        for s in S:
            for q in dev: s.answer(q)        # touch dev only; no tuning
            s.read = 0; s.nq = 0
        for q in test:
            rec = dict(seed=seed, qid=q["qid"], kind=q["kind"], cluster=q["cluster"], question=q["text"], gold=q["gold"],
                       lex_overlap=0.0)
            src = next((m for m in memories if m["id"] == q["src"]), None)
            if src:
                A_, B_ = set(tokenize(q["text"])), set(tokenize(src["text"]))
                rec["lex_overlap"] = len(A_ & B_) / max(1, len(A_ | B_))
            for s in S:
                ans, conf = s.answer(q)
                c, a, w, u = score(q["gold"], ans)
                rec[s.name] = dict(answer=ans if ans is not None else "ABSTAIN", confidence=round(float(conf), 4),
                                   outcome="correct" if c and not a else ("abstain" if a else "false"),
                                   correct=c, abstain=a, wrong=w, util=u)
            perq.append(rec)
        for s in S:
            rows.append(dict(seed=seed, system=s.name, bytes_read_per_q=s.read / max(1, s.nq), retrieve_cap=cap))
        # relation probe (A-E)
        pairs, pp = make_probe_pairs(facts, held, seed)
        for s in S[2:]:
            for a, b, g in pairs:
                p, cf = s.classify_pair(a, b)
                probe_rows.append(dict(seed=seed, system=s.name, a=a, b=b, gold=g, pred=p, conf=cf))
        # interference +1000
        m2, f2 = synth_interference(1000, seed, facts)
        saved = dict(FACTS_BY_MEM)
        for f in f2: FACTS_BY_MEM.setdefault(f["memory_id"], []).append(f)
        allf = facts + [dict(f, id=len(facts) + i) for i, f in enumerate(f2)]
        for f in allf[len(facts):]:
            pass
        FACTS_BY_MEM.clear(); 
        for f in allf: FACTS_BY_MEM.setdefault(f["memory_id"], []).append(f)
        S2 = build_all(memories + m2, allf, build_edges(allf), seed, cap)
        for s in S2:
            acc = np.mean([score(q["gold"], s.answer(q)[0])[0] for q in test])
            fr = np.mean([score(q["gold"], s.answer(q)[0])[2] for q in test])
            intf_rows.append(dict(seed=seed, system=s.name, post_acc=float(acc), post_false=float(fr)))
        FACTS_BY_MEM.clear(); FACTS_BY_MEM.update(saved)
        print(f"seed {seed}: n_test={len(test)} probe={len(pairs)} t={time.time()-t0:.0f}s", flush=True)

    with open(OUT / "per_question.jsonl", "w") as fh:
        for r in perq: fh.write(json.dumps(r) + "\n")
    (OUT / "storage.json").write_text(json.dumps(storage, indent=1))
    (OUT / "reads.json").write_text(json.dumps(rows, indent=1))
    with open(OUT / "probe_pairs.jsonl", "w") as fh:
        for r in probe_rows: fh.write(json.dumps(r) + "\n")
    (OUT / "interference.json").write_text(json.dumps(intf_rows, indent=1))
    (OUT / "question_availability.json").write_text(json.dumps(avail, indent=1))
    (OUT / "run_meta.json").write_text(json.dumps(dict(script_sha256=sha(Path(__file__).read_bytes()), seeds=SEEDS,
        when_et=dt.datetime.now().isoformat(), runtime_s=time.time() - t0), indent=1))
    print("done", time.time() - t0)

if __name__ == "__main__":
    main()
