#!/usr/bin/env python3
"""Cognitive routers for the Y/Z/G experiment (classical only).

Configs:
  existing   — JARVIS pack_router
  graph_ppr  — personalized PageRank over concept↔pack graph
  yz         — YBit confidence + ZBit bias
  yzg        — yz + GGraph influence features
  yz_zrand   — control: Z replaced by matched-variance deterministic scalar
  softmax    — control: plain softmax over same evidence as yz
"""
from __future__ import annotations

import hashlib
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path("/workspace/jarvis/chat_rag")))

from lib.primitives import GGraph, YBit, ZBit  # noqa: E402

PACKS = ("dictionary", "books", "uploads")
INTENTS = ("debug", "explain", "rewrite")

# Mirror chat_rag pack sources / keywords (imported lightly to avoid circular deps)
PACK_SOURCES = {
    "dictionary": {"wordnet"},
    "books": {"literature", "history", "cyber_ai"},
    "uploads": {"novel", "upload", "fact_card", "pack_fact"},
}

PACK_KEYWORDS = {
    "dictionary": ("mean", "means", "meaning", "define", "definition", "synonym", "word for"),
    "books": ("passage", "chapter", "according to the book", "in the novel"),
    "uploads": (
        "zorblax", "atlantis", "poseidonis", "gleeb", "quill", "aerogel",
        "xylith", "amberfall", "kestrel", "nexora", "helixwind", "marlowe",
        "glaucite", "northmere", "starlight wren", "capital of",
        "who wrote", "who invented", "who discovered", "who founded",
        "what year", "what color", "how many",
    ),
}

INTENT_KEYWORDS = {
    "debug": ("debug", "bug", "error", "crash", "stack trace", "traceback", "failing", "exception", "race condition", "fix"),
    "explain": ("explain", "what does this", "help me understand", "walk me through", "in simple terms", "step by step", "how does", "difference between"),
    "rewrite": ("rewrite", "refactor", "restructure", "convert", "make a cleaner", "more readable", "simpler english"),
}


def _tokenize(query: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", query.lower())


def _stable_seed(*parts: str, salt: int = 0) -> int:
    h = hashlib.sha256(("||".join(parts) + f"|{salt}").encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") % (2**31 - 1)


def _softmax(xs: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    t = max(1e-6, float(temperature))
    z = xs / t
    z = z - z.max()
    e = np.exp(z)
    s = e.sum()
    if s <= 0 or not np.isfinite(s):
        return np.ones_like(xs) / len(xs)
    return e / s


@dataclass
class Decision:
    pack: str
    scores: Dict[str, float]
    confidence: float
    reason: str
    intent: Optional[str] = None
    intent_scores: Optional[Dict[str, float]] = None


# ---------- evidence features shared by yz / softmax / controls ----------

def keyword_pack_scores(query: str) -> Dict[str, float]:
    q = query.lower()
    out = {p: 0.0 for p in PACKS}
    for pack, kws in PACK_KEYWORDS.items():
        for kw in kws:
            if kw in q:
                out[pack] += 1.0 + 0.2 * len(kw.split())
    if re.search(r"\bdefine\b|\bmeaning\b|\bmeans\b|\bwhat does\b", q):
        out["dictionary"] += 3.0
    return out


def keyword_intent_scores(query: str) -> Dict[str, float]:
    q = query.lower()
    out = {i: 0.0 for i in INTENTS}
    for intent, kws in INTENT_KEYWORDS.items():
        for kw in kws:
            if kw in q:
                out[intent] += 1.0 + 0.15 * len(kw.split())
    return out


def bm25_pack_scores(mem, query: str, per_pack_k: int = 5) -> Dict[str, float]:
    from model.memory_rag import tokenize

    scores = mem.bm25.get_scores(tokenize(query))
    idx = np.argsort(scores)[::-1]
    out = {p: 0.0 for p in PACKS}
    counts = {p: 0 for p in PACKS}
    for i in idx:
        sc = float(scores[int(i)])
        if sc <= 0:
            break
        src = (mem.chunks[int(i)].get("source") or "").strip()
        for pack, sources in PACK_SOURCES.items():
            if src in sources and counts[pack] < per_pack_k:
                out[pack] += sc
                counts[pack] += 1
        if all(counts[p] >= per_pack_k for p in PACKS):
            break
    return out


def evidence_vector(mem, query: str, alpha_kw: float = 1.5) -> Dict[str, float]:
    kw = keyword_pack_scores(query)
    bm = bm25_pack_scores(mem, query)
    return {p: alpha_kw * kw.get(p, 0.0) + bm.get(p, 0.0) for p in PACKS}


# ---------- C1 existing ----------

class ExistingRouter:
    name = "existing"

    def __init__(self, mem):
        self.mem = mem
        from pack_router import route_query
        self._route_query = route_query

    def route(self, query: str) -> Decision:
        dec = self._route_query(self.mem, query)
        scores = {p: float(dec.scores.get(p, 0.0)) for p in PACKS}
        total = sum(max(0.0, v) for v in scores.values()) or 1.0
        conf = max(0.0, scores.get(dec.pack, 0.0)) / total
        intent_sc = keyword_intent_scores(query)
        intent = max(intent_sc, key=intent_sc.get) if max(intent_sc.values()) > 0 else None
        return Decision(dec.pack, scores, conf, f"existing:{dec.reason}", intent, intent_sc)


# ---------- C2 graph PPR ----------

class GraphPPRRouter:
    name = "graph_ppr"

    def __init__(self, mem, damping: float = 0.85, steps: int = 20, seed: int = 0):
        self.mem = mem
        self.damping = damping
        self.steps = steps
        self.seed = seed
        self.nodes, self.adj, self.pack_nodes = self._build_graph(mem, seed)

    @staticmethod
    def _build_graph(mem, seed: int):
        """Nodes: packs + frequent tokens from chunks; edges from co-occurrence."""
        rng = np.random.default_rng(seed)
        pack_nodes = {p: i for i, p in enumerate(PACKS)}
        # Collect token→pack mass from a sample of chunks
        token_pack: Dict[str, Dict[str, float]] = {}
        n_chunks = len(mem.chunks)
        # Stratified sample up to 4000 chunks for speed
        idxs = np.linspace(0, n_chunks - 1, num=min(4000, n_chunks), dtype=int)
        for i in idxs:
            c = mem.chunks[int(i)]
            src = (c.get("source") or "").strip()
            pack = None
            for p, sources in PACK_SOURCES.items():
                if src in sources:
                    pack = p
                    break
            if pack is None:
                continue
            toks = set(_tokenize(c.get("text") or "")[:40])
            for t in toks:
                if len(t) < 3:
                    continue
                token_pack.setdefault(t, {pp: 0.0 for pp in PACKS})
                token_pack[t][pack] += 1.0
        # Keep top tokens by total mass
        ranked = sorted(token_pack.items(), key=lambda kv: sum(kv[1].values()), reverse=True)[:800]
        nodes = list(PACKS) + [t for t, _ in ranked]
        index = {n: i for i, n in enumerate(nodes)}
        n = len(nodes)
        # Adjacency as list of (dst, weight)
        adj: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
        # Pack ↔ token edges
        for t, masses in ranked:
            ti = index[t]
            for p, m in masses.items():
                if m <= 0:
                    continue
                pi = index[p]
                w = math.log1p(m)
                adj[ti].append((pi, w))
                adj[pi].append((ti, w))
        # Light pack–pack prior (small)
        for a in PACKS:
            for b in PACKS:
                if a == b:
                    continue
                adj[index[a]].append((index[b], 0.05))
        # Token–token: share if co-listed in same pack strongly (optional sparse)
        # Skip dense token-token for speed; PPR via packs is enough.
        # Normalize out-weights
        for i in range(n):
            s = sum(w for _, w in adj[i]) or 1.0
            adj[i] = [(j, w / s) for j, w in adj[i]]
        _ = rng  # reserved for future stochastic walks
        return nodes, adj, pack_nodes

    def _ppr(self, personalization: Dict[int, float]) -> np.ndarray:
        n = len(self.nodes)
        p = np.zeros(n, dtype=np.float64)
        tele = np.zeros(n, dtype=np.float64)
        s = sum(personalization.values()) or 1.0
        for i, v in personalization.items():
            tele[i] = v / s
        p[:] = tele
        d = self.damping
        for _ in range(self.steps):
            new = (1 - d) * tele
            for i, edges in enumerate(self.adj):
                if p[i] == 0:
                    continue
                for j, w in edges:
                    new[j] += d * p[i] * w
            p = new
        return p

    def route(self, query: str) -> Decision:
        # Personalize on query tokens that exist as nodes + weak evidence packs
        index = {n: i for i, n in enumerate(self.nodes)}
        pers: Dict[int, float] = {}
        for t in _tokenize(query):
            if t in index:
                pers[index[t]] = pers.get(index[t], 0.0) + 1.0
        ev = evidence_vector(self.mem, query)
        # Strong evidence teleports so pack BM25/keywords dominate token priors
        ev_max = max(ev.values()) if ev else 1.0
        for p, sc in ev.items():
            pers[self.pack_nodes[p]] = pers.get(self.pack_nodes[p], 0.0) + 5.0 * max(0.0, sc) / (ev_max + 1e-9)
        if not pers:
            pers = {self.pack_nodes[p]: 1.0 for p in PACKS}
        ranks = self._ppr(pers)
        scores = {p: float(ranks[self.pack_nodes[p]]) for p in PACKS}
        pack = max(scores, key=scores.get)
        total = sum(scores.values()) or 1.0
        conf = scores[pack] / total
        intent_sc = keyword_intent_scores(query)
        intent = max(intent_sc, key=intent_sc.get) if max(intent_sc.values()) > 0 else None
        return Decision(pack, scores, conf, "graph_ppr", intent, intent_sc)


# ---------- Y/Z helpers ----------

def _matched_var_zrand(query: str, seed_salt: int = 0, target_var: float = 0.04) -> float:
    """Deterministic scalar in (0,1) with approx matched variance to bias_sigma population."""
    rng = np.random.default_rng(_stable_seed("zrand", query, salt=seed_salt))
    # logistic of normal → variance tuned
    x = rng.normal(0.0, math.sqrt(target_var) * 4.0)
    return float(1.0 / (1.0 + math.exp(-x)))


def _ybit_adjust_probs(raw: Dict[str, float], z_bias: float, lam: float = 0.85) -> Dict[str, float]:
    """Map evidence → per-candidate YBit adjusted probs, then renormalize."""
    names = list(raw.keys())
    vec = np.array([max(0.0, float(raw[n])) for n in names], dtype=np.float64)
    if vec.sum() <= 0:
        base = np.ones(len(names)) / len(names)
    else:
        base = vec / vec.sum()
    out = {}
    # Shared ZBit stand-in: use bias as sigma_b for each YBit
    class _StubZ:
        def __init__(self, sigma):
            self._sigma = sigma
            self._value = 0.5

        def bias_sigma(self, k=1.0):
            return self._sigma

        def value(self):
            return self._value

        def phase(self):
            return 0.0

    stub = _StubZ(z_bias)
    for i, name in enumerate(names):
        p = float(base[i])
        # amplitude pair from p
        alpha = complex(math.sqrt(max(0.0, 1.0 - p)), 0.0)
        beta = complex(math.sqrt(p), 0.0)
        y = YBit((alpha, beta), stub, lam=lam)  # type: ignore[arg-type]
        out[name] = y.adjusted_prob_1()
    # renormalize
    s = sum(out.values()) or 1.0
    return {k: v / s for k, v in out.items()}


class YZRouter:
    name = "yz"

    def __init__(self, mem, lam: float = 0.85, z_mix: float = 0.35, seed_salt: int = 0, use_zrand: bool = False):
        self.mem = mem
        self.lam = lam
        self.z_mix = z_mix
        self.seed_salt = seed_salt
        self.use_zrand = use_zrand
        if use_zrand:
            self.name = "yz_zrand"

    def route(self, query: str) -> Decision:
        ev = evidence_vector(self.mem, query)
        if self.use_zrand:
            z_bias = _matched_var_zrand(query, self.seed_salt)
            reason = "yz_zrand"
        else:
            z_bias = _z_bias_from_query(query, self.seed_salt)
            reason = "yz"
        # Blend: raise/lower evidence trust by z_bias
        scale = (1.0 - self.z_mix) + self.z_mix * (0.5 + z_bias)
        scaled = {p: ev[p] * scale for p in PACKS}
        probs = _ybit_adjust_probs(scaled, z_bias=float(z_bias), lam=self.lam)
        pack = max(probs, key=probs.get)
        # Intent via same machinery on keyword scores
        intent_raw = keyword_intent_scores(query)
        if sum(intent_raw.values()) > 0:
            intent_probs = _ybit_adjust_probs(intent_raw, z_bias=float(z_bias), lam=self.lam)
            intent = max(intent_probs, key=intent_probs.get)
        else:
            intent_probs = {i: 1.0 / len(INTENTS) for i in INTENTS}
            intent = None
        return Decision(pack, probs, float(probs[pack]), reason, intent, intent_probs)


def _z_bias_from_query(query: str, seed_salt: int = 0):
    z = ZBit(seed=_stable_seed(query, salt=seed_salt), side="auto")
    return float(z.bias_sigma(1.0))


class SoftmaxRouter:
    name = "softmax"

    def __init__(self, mem, temperature: float = 1.0):
        self.mem = mem
        self.temperature = temperature

    def route(self, query: str) -> Decision:
        ev = evidence_vector(self.mem, query)
        names = list(PACKS)
        vec = np.array([ev[p] for p in names], dtype=np.float64)
        probs_a = _softmax(vec, self.temperature)
        probs = {p: float(probs_a[i]) for i, p in enumerate(names)}
        pack = max(probs, key=probs.get)
        intent_raw = keyword_intent_scores(query)
        ivec = np.array([intent_raw[i] for i in INTENTS], dtype=np.float64)
        if ivec.sum() > 0:
            ip = _softmax(ivec, self.temperature)
            intent_probs = {n: float(ip[i]) for i, n in enumerate(INTENTS)}
            intent = max(intent_probs, key=intent_probs.get)
        else:
            intent_probs = {i: 1.0 / len(INTENTS) for i in INTENTS}
            intent = None
        return Decision(pack, probs, float(probs[pack]), "softmax", intent, intent_probs)


class YZGRouter:
    name = "yzg"

    def __init__(self, mem, lam: float = 0.85, z_mix: float = 0.35, g_mix: float = 0.25, seed_salt: int = 0, g_n: int = 512):
        self.mem = mem
        self.lam = lam
        self.z_mix = z_mix
        self.g_mix = g_mix
        self.seed_salt = seed_salt
        # Smaller n for speed; still classical GGraph with 2-D output
        self.g = GGraph(n=g_n, out_degree=3, gamma=0.87, seed=424242 + seed_salt)

    def route(self, query: str) -> Decision:
        ev = evidence_vector(self.mem, query)
        z_bias = _z_bias_from_query(query, self.seed_salt)
        scale = (1.0 - self.z_mix) + self.z_mix * (0.5 + z_bias)
        scaled = {p: ev[p] * scale for p in PACKS}
        y_probs = _ybit_adjust_probs(scaled, z_bias=z_bias, lam=self.lam)
        # GGraph: influence(query|pack) → even mass as affinity (2-number bottleneck)
        g_raw = {}
        for p in PACKS:
            even, odd = self.g.influence(f"{query}::{p}")
            # Use even-odd signed signal mapped to [0,1]
            g_raw[p] = 0.5 + 0.5 * (even - odd)
        g_sum = sum(g_raw.values()) or 1.0
        g_probs = {p: g_raw[p] / g_sum for p in PACKS}
        # Mix
        probs = {
            p: (1.0 - self.g_mix) * y_probs[p] + self.g_mix * g_probs[p]
            for p in PACKS
        }
        s = sum(probs.values()) or 1.0
        probs = {p: v / s for p, v in probs.items()}
        pack = max(probs, key=probs.get)
        intent_raw = keyword_intent_scores(query)
        if sum(intent_raw.values()) > 0:
            intent_probs = _ybit_adjust_probs(intent_raw, z_bias=z_bias, lam=self.lam)
            intent = max(intent_probs, key=intent_probs.get)
        else:
            intent_probs = {i: 1.0 / len(INTENTS) for i in INTENTS}
            intent = None
        return Decision(pack, probs, float(probs[pack]), "yzg", intent, intent_probs)


def build_router(name: str, mem, seed: int = 0, **params):
    if name == "existing":
        return ExistingRouter(mem)
    if name == "graph_ppr":
        return GraphPPRRouter(mem, damping=params.get("damping", 0.85), seed=seed)
    if name == "yz":
        return YZRouter(mem, lam=params.get("lam", 0.85), z_mix=params.get("z_mix", 0.35), seed_salt=seed)
    if name == "yz_zrand":
        return YZRouter(mem, lam=params.get("lam", 0.85), z_mix=params.get("z_mix", 0.35), seed_salt=seed, use_zrand=True)
    if name == "softmax":
        return SoftmaxRouter(mem, temperature=params.get("temperature", 1.0))
    if name == "yzg":
        return YZGRouter(
            mem,
            lam=params.get("lam", 0.85),
            z_mix=params.get("z_mix", 0.35),
            g_mix=params.get("g_mix", 0.25),
            seed_salt=seed,
            g_n=params.get("g_n", 512),
        )
    raise ValueError(name)
