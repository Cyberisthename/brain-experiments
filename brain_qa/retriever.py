#!/usr/bin/env python3
"""Pluggable retriever + extractive answerer."""
from __future__ import annotations
import json, re
from pathlib import Path
from index_corpus import BM25, tokenize, build, OUT

NUM = re.compile(r"[±]?\d+(?:\.\d+)?(?:e[-+]?\d+)?%?")

class Retriever:
    def __init__(self, index_path=None):
        path = Path(index_path) if index_path else OUT / "index.json"
        if not path.exists():
            build()
        data = json.loads(path.read_text())
        self.docs = data["docs"]
        self.bm25 = BM25(self.docs)
        self.meta = {k: data[k] for k in ("n_docs", "n_chunks", "method") if k in data}

    def search(self, query, k=5):
        return self.bm25.search(query, k=k)

def extract_numbers(text):
    return NUM.findall(text)

def extractive_answer(query, hits, max_chars=1200):
    """Return quoted evidence + pulled numbers; no generative hallucination."""
    if not hits:
        return {"answer": "No matching chunks found in the flybrain corpus.", "citations": [], "numbers": []}
    lines = [f"Q: {query}", "", "Top evidence (extractive):"]
    citations, nums = [], []
    used = 0
    for doc, score in hits:
        cite = f"{doc['path']}#chunk{doc['i']}"
        citations.append({"path": doc["path"], "chunk": doc["i"], "score": round(float(score), 4)})
        snippet = doc["text"].strip()
        if len(snippet) > 400:
            snippet = snippet[:400] + "…"
        block = f"\n[{cite}] (score={score:.3f})\n\"\"\"\n{snippet}\n\"\"\"\n"
        if used + len(block) > max_chars and used > 0:
            break
        lines.append(block)
        used += len(block)
        nums.extend(extract_numbers(doc["text"]))
    # unique numbers preserving order
    seen, uniq = set(), []
    for n in nums:
        if n not in seen:
            seen.add(n); uniq.append(n)
    lines.append("Numbers mentioned in top chunks: " + (", ".join(uniq[:40]) if uniq else "(none)"))
    return {"answer": "\n".join(lines), "citations": citations, "numbers": uniq[:40]}

# --- Optional generator interface (plug your AI here) --------------------
class AnswerGenerator:
    """Default: extractive only. Subclass or set BRAIN_QA_GENERATOR to replace."""
    def generate(self, query, hits):
        return extractive_answer(query, hits)

def load_generator():
    import os, importlib
    spec = os.environ.get("BRAIN_QA_GENERATOR")
    if not spec:
        return AnswerGenerator()
    # "module:ClassName" or "module:factory"
    mod_name, _, attr = spec.partition(":")
    mod = importlib.import_module(mod_name)
    obj = getattr(mod, attr) if attr else mod
    return obj() if callable(obj) else obj

def ask(query, k=5, generator=None):
    r = Retriever()
    hits = r.search(query, k=k)
    gen = generator or load_generator()
    out = gen.generate(query, hits)
    out["meta"] = r.meta
    return out
