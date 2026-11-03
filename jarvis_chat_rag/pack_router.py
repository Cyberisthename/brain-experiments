"""Cheap knowledge-pack router for JARVIS RAG.

Classify → retrieve only inside that pack (avoids WordNet/book bleed into
novel facts). Three packs match the user's architecture:

  dictionary  → wordnet
  books       → literature + history + cyber_ai
  uploads     → novel + upload + fact_card + pack_fact

Routing = keyword rules + per-pack BM25 mass. No training.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set

from dictionary_lookup import extract_define_query

PACK_SOURCES: Dict[str, Set[str]] = {
    "dictionary": {"wordnet"},
    "books": {"literature", "history", "cyber_ai"},
    "uploads": {"novel", "upload", "fact_card", "pack_fact"},
}

PACK_KEYWORDS: Dict[str, Sequence[str]] = {
    "dictionary": (
        "mean", "means", "meaning", "define", "definition", "synonym", "word for",
    ),
    "books": (
        "passage", "chapter", "according to the book", "in the novel",
    ),
    "uploads": (
        "zorblax", "atlantis", "poseidonis", "gleeb", "quill", "aerogel",
        "xylith", "amberfall", "kestrel", "nexora", "helixwind", "marlowe",
        "glaucite", "northmere", "starlight wren", "capital of",
        "who wrote", "who invented", "who discovered", "who founded",
        "what year", "what color", "how many",
    ),
}


@dataclass
class RouteDecision:
    pack: str
    sources: Set[str]
    scores: Dict[str, float]
    reason: str


def _kw_scores(query: str) -> Dict[str, float]:
    q = query.lower()
    out = {p: 0.0 for p in PACK_SOURCES}
    for pack, kws in PACK_KEYWORDS.items():
        for kw in kws:
            if kw in q:
                out[pack] += 1.0 + 0.2 * len(kw.split())
    if extract_define_query(query):
        out["dictionary"] += 5.0
    return out


def _pack_bm25_scores(mem, query: str, per_pack_k: int = 5) -> Dict[str, float]:
    import numpy as np
    from model.memory_rag import tokenize

    scores = mem.bm25.get_scores(tokenize(query))
    idx = np.argsort(scores)[::-1]
    out = {p: 0.0 for p in PACK_SOURCES}
    counts = {p: 0 for p in PACK_SOURCES}
    for i in idx:
        sc = float(scores[int(i)])
        if sc <= 0:
            break
        src = (mem.chunks[int(i)].get("source") or "").strip()
        for pack, sources in PACK_SOURCES.items():
            if src in sources and counts[pack] < per_pack_k:
                out[pack] += sc
                counts[pack] += 1
        if all(counts[p] >= per_pack_k for p in PACK_SOURCES):
            break
    return out


def route_query(mem, query: str, alpha_kw: float = 1.5) -> RouteDecision:
    kw = _kw_scores(query)
    bm = _pack_bm25_scores(mem, query)
    combined = {p: alpha_kw * kw.get(p, 0.0) + bm.get(p, 0.0) for p in PACK_SOURCES}

    if extract_define_query(query):
        return RouteDecision("dictionary", set(PACK_SOURCES["dictionary"]), combined, "define-query → dictionary")

    # Fact-style questions: prefer uploads whenever uploads have any real BM25 mass.
    # Fact cards live there; long book passages often steal the global top-k.
    q = query.lower()
    factish = any(
        p in q
        for p in (
            "who wrote", "who invented", "who discovered", "who founded",
            "capital of", "what year", "what color", "how many", "what does the",
        )
    )
    if factish and bm.get("uploads", 0.0) > 0.0:
        return RouteDecision(
            "uploads",
            set(PACK_SOURCES["uploads"]),
            combined,
            f"factish → uploads (bm={bm['uploads']:.2f})",
        )

    pack = max(combined, key=combined.get)
    return RouteDecision(pack, set(PACK_SOURCES[pack]), combined, f"argmax ({pack}={combined[pack]:.2f})")


class RoutedRetriever:
    def __init__(self, base_retriever, enable_route: bool = True, k: Optional[int] = None, max_chars: Optional[int] = None):
        self.base = base_retriever
        self.mem = base_retriever.mem
        self.enable_route = enable_route
        self.k = k if k is not None else base_retriever.k
        self.max_chars = max_chars if max_chars is not None else base_retriever.max_chars
        self.last_route: Optional[RouteDecision] = None

    def route(self, query: str) -> RouteDecision:
        self.last_route = route_query(self.mem, query)
        return self.last_route

    def search(self, query: str, k: Optional[int] = None) -> List[dict]:
        k = k or self.k
        if not self.enable_route:
            return self.mem.search(query, k=k)
        dec = self.route(query)
        import numpy as np
        from model.memory_rag import tokenize

        scores = self.mem.bm25.get_scores(tokenize(query))
        idx = np.argsort(scores)[::-1]
        out = []
        for i in idx:
            if scores[i] <= 0:
                continue
            c = dict(self.mem.chunks[int(i)])
            if c.get("source") not in dec.sources:
                continue
            c["score"] = float(scores[int(i)])
            c["routed_pack"] = dec.pack
            out.append(c)
            if len(out) >= k:
                break
        if not out:
            out = self.mem.search(query, k=k)
            for h in out:
                h["routed_pack"] = dec.pack + ":fallback_global"
        return out

    def context(self, query: str, k: Optional[int] = None, max_chars: Optional[int] = None) -> str:
        hits = self.search(query, k=k or self.k)
        parts, n = [], 0
        cap = max_chars or self.max_chars
        for h in hits:
            block = f"[{h['source']} | {h['score']:.2f}]\n{h['text']}"
            if n + len(block) > cap and parts:
                break
            parts.append(block)
            n += len(block)
        return "\n\n---\n\n".join(parts)

    def augment(self, prompt: str, query: Optional[str] = None) -> str:
        q = query if query is not None else prompt
        ctx = self.context(q)
        if not ctx:
            return prompt
        return f"Context:\n{ctx}\n\n{prompt}"

    def info(self) -> dict:
        return {
            "routed": self.enable_route,
            "last_route": None
            if not self.last_route
            else {"pack": self.last_route.pack, "reason": self.last_route.reason, "scores": self.last_route.scores},
            "base": self.base.info() if hasattr(self.base, "info") else {},
        }
