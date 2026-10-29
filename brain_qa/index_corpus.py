#!/usr/bin/env python3
"""Build a local BM25 index over flybrain RESULTS/PREREG/ANALYSIS/SOURCES markdown."""
from __future__ import annotations
import json, re, math
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent
FLY = Path("/workspace/flybrain")
CORPUS = ROOT / "corpus"
OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)
CORPUS.mkdir(exist_ok=True)

PATTERNS = ("RESULTS*.md", "PREREG*.md", "PREREGISTRATION*.md", "ANALYSIS*.md", "SOURCES.md")

def find_docs():
    seen, paths = set(), []
    for pat in PATTERNS:
        for p in FLY.rglob(pat):
            if ".venv" in p.parts or "node_modules" in p.parts:
                continue
            rp = str(p.resolve())
            if rp not in seen:
                seen.add(rp); paths.append(p)
    return sorted(paths)

def tokenize(text: str):
    return re.findall(r"[a-z0-9]+(?:[./_-][a-z0-9]+)*|[±]?\d+(?:\.\d+)?(?:e[-+]?\d+)?", text.lower())

def chunk_markdown(text: str, path: str, max_chars=700, overlap=80):
    # split on headings / blank lines
    parts = re.split(r"(?=\n#{1,3}\s+)|\n{2,}", text)
    buf, chunks = "", []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if len(buf) + len(part) + 1 <= max_chars:
            buf = (buf + "\n\n" + part).strip()
        else:
            if buf:
                chunks.append(buf)
            if len(part) <= max_chars:
                buf = part
            else:
                # hard wrap long tables
                for i in range(0, len(part), max_chars - overlap):
                    chunks.append(part[i:i + max_chars])
                buf = ""
    if buf:
        chunks.append(buf)
    out = []
    for i, c in enumerate(chunks):
        out.append({"id": f"{path}::{i}", "path": path, "i": i, "text": c})
    return out

class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = docs
        self.toks = [tokenize(d["text"]) for d in docs]
        self.N = len(docs)
        self.df = Counter()
        for t in self.toks:
            for w in set(t):
                self.df[w] += 1
        self.avgdl = sum(len(t) for t in self.toks) / max(1, self.N)
        self.tf = [Counter(t) for t in self.toks]

    def idf(self, w):
        n = self.df.get(w, 0)
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def score(self, query: str):
        q = tokenize(query)
        scores = []
        for i, tf in enumerate(self.tf):
            dl = len(self.toks[i])
            s = 0.0
            for w in q:
                if w not in tf:
                    continue
                f = tf[w]
                s += self.idf(w) * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            scores.append(s)
        return scores

    def search(self, query, k=5):
        scores = self.score(query)
        idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(self.docs[i], scores[i]) for i in idx if scores[i] > 0]

def build():
    paths = find_docs()
    all_chunks = []
    manifest = []
    for p in paths:
        rel = str(p.relative_to(FLY))
        text = p.read_text(errors="ignore")
        # copy slim pointer into corpus/
        (CORPUS / (rel.replace("/", "__"))).write_text(text)
        ch = chunk_markdown(text, rel)
        all_chunks.extend(ch)
        manifest.append({"path": rel, "n_chunks": len(ch), "chars": len(text)})
    index = {
        "docs": all_chunks,
        "manifest": manifest,
        "n_docs": len(paths),
        "n_chunks": len(all_chunks),
        "method": "bm25",
    }
    (OUT / "index.json").write_text(json.dumps(index))
    print(f"indexed {len(paths)} files → {len(all_chunks)} chunks")
    return index

if __name__ == "__main__":
    build()
