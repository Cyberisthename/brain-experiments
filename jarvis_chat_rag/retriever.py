"""BM25 pack retriever + local upload store for JARVIS chat RAG.

Reuses PackMemory / build_index from jarvis_kaggle (read-only import).
Institutional Books are never indexed. Uploaded facts live under chat_rag/knowledge/.
"""
from __future__ import annotations

import json
import pickle
import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional

ROOT = Path(__file__).resolve().parent
KAGGLE = Path("/workspace/jarvis_kaggle")
sys.path.insert(0, str(KAGGLE))

from model.memory_rag import PackMemory, build_index, tokenize  # noqa: E402

PACK_INDEX = KAGGLE / "adapters" / "memory" / "bm25_packs.pkl"
KNOWLEDGE = Path("/workspace/jarvis_model/knowledge")
UPLOAD_JSONL = ROOT / "knowledge" / "uploaded_facts.jsonl"
NOTES_JSONL = ROOT / "knowledge" / "jarvis_notes.jsonl"
COMBINED_INDEX = ROOT / "knowledge" / "bm25_packs_plus_uploads.pkl"


def ensure_pack_index() -> Path:
    if PACK_INDEX.exists():
        return PACK_INDEX
    paths = [
        KNOWLEDGE / "literature" / "dataset" / "train_passages.jsonl",
        KNOWLEDGE / "history" / "dataset" / "train_passages.jsonl",
        KNOWLEDGE / "cyber_ai" / "dataset" / "train_passages.jsonl",
    ]
    PACK_INDEX.parent.mkdir(parents=True, exist_ok=True)
    return build_index(paths, PACK_INDEX, source_labels=["literature", "history", "cyber_ai"])


def load_uploads(path: Path = UPLOAD_JSONL) -> List[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def append_upload(text: str, source: str = "upload", meta: Optional[dict] = None) -> dict:
    """Append a fact document to the local upload store and invalidate combined index."""
    UPLOAD_JSONL.parent.mkdir(parents=True, exist_ok=True)
    row = {"text": text.strip(), "source": source}
    if meta:
        row.update(meta)
    with UPLOAD_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if COMBINED_INDEX.exists():
        COMBINED_INDEX.unlink()
    return row


def seed_novel_facts(force: bool = False) -> int:
    """Seed chat_rag uploads from the knowledge_injection novel_facts.jsonl (idempotent)."""
    src = KAGGLE / "knowledge_injection" / "data" / "novel_facts.jsonl"
    if not src.exists():
        return 0
    existing = load_uploads()
    have = {r.get("id") for r in existing if r.get("id")}
    if have and not force:
        # already seeded at least one novel id
        novel_have = [i for i in have if str(i).startswith("novel_")]
        if novel_have:
            return 0
    n = 0
    for line in src.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        if o.get("id") in have:
            continue
        text = o.get("inject_doc") or o.get("fact_statement") or ""
        if len(text) < 10:
            continue
        append_upload(
            text,
            source="novel",
            meta={
                "id": o.get("id"),
                "question": o.get("question"),
                "answer": o.get("answer"),
                "fact_statement": o.get("fact_statement"),
            },
        )
        n += 1
    return n



def seed_fact_cards(force: bool = False) -> int:
    """Short authorship/capital fact cards + pack_facts statements for reliable QA retrieval."""
    have = {r.get("id") for r in load_uploads()}
    n = 0
    manual = [
        {"id": "card_dracula", "text": "Bram Stoker wrote Dracula. Dracula was written by Bram Stoker.",
         "answer": "Bram Stoker", "question": "Who wrote Dracula?"},
        {"id": "card_frankenstein",
         "text": "Mary Shelley wrote Frankenstein. Frankenstein was written by Mary Shelley (Mary Wollstonecraft Shelley).",
         "answer": "Mary Shelley", "question": "Who wrote Frankenstein?"},
        {"id": "card_france", "text": "The capital of France is Paris. Paris is the capital of France.",
         "answer": "Paris", "question": "What is the capital of France?"},
        {"id": "card_wwi",
         "text": "World War I began in 1914. The First World War started in the year 1914.",
         "answer": "1914", "question": "What year did World War I begin?"},
        {"id": "card_hamlet",
         "text": "William Shakespeare wrote Hamlet. Hamlet was written by Shakespeare.",
         "answer": "Shakespeare", "question": "Who wrote Hamlet?"},
    ]
    for m in manual:
        if m["id"] in have and not force:
            continue
        if m["id"] in have:
            continue
        append_upload(m["text"], source="fact_card", meta={k: m[k] for k in m if k != "text"})
        n += 1
    pack = KAGGLE / "knowledge_injection" / "data" / "pack_facts.jsonl"
    if pack.exists():
        for i, line in enumerate(pack.read_text(encoding="utf-8").splitlines()):
            if i >= 40:
                break
            if not line.strip():
                continue
            o = json.loads(line)
            if o.get("id") in have:
                continue
            text = (o.get("fact_statement") or "").strip()
            if len(text) < 10:
                continue
            append_upload(
                text + " " + text,
                source="pack_fact",
                meta={"id": o.get("id"), "answer": o.get("answer"), "question": o.get("question")},
            )
            n += 1
    return n


def build_combined_index(force: bool = False) -> Path:
    """Pack index + uploaded facts into one BM25 pickle under chat_rag/knowledge/."""
    if COMBINED_INDEX.exists() and not force:
        return COMBINED_INDEX
    ensure_pack_index()
    seed_novel_facts()
    seed_fact_cards()
    with PACK_INDEX.open("rb") as f:
        blob = pickle.load(f)
    chunks = list(blob["chunks"])
    for i, row in enumerate(load_uploads()):
        text = (row.get("text") or "").strip()
        if len(text) < 10:
            continue
        chunks.append(
            {
                "id": row.get("id") or f"upload:{i}",
                "source": row.get("source") or "upload",
                "text": text[:4000],
            }
        )
    # jarvis_notes.jsonl — research fact cards (separate from novel eval uploads)
    for i, row in enumerate(load_uploads(NOTES_JSONL)):
        text = (row.get("text") or "").strip()
        if len(text) < 10:
            continue
        chunks.append(
            {
                "id": row.get("id") or f"jarvis_notes:{i}",
                "source": row.get("source") or "jarvis_notes",
                "text": text[:4000],
            }
        )
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as e:
        raise SystemExit("rank_bm25 required: pip install rank_bm25") from e
    corpus = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(corpus) if corpus else BM25Okapi([["empty"]])
    COMBINED_INDEX.parent.mkdir(parents=True, exist_ok=True)
    with COMBINED_INDEX.open("wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)
    meta = {
        "n_chunks": len(chunks),
        "n_uploads": len(load_uploads()),
        "n_jarvis_notes": len(load_uploads(NOTES_JSONL)),
        "sources": sorted({c["source"] for c in chunks}),
        "path": str(COMBINED_INDEX),
        "note": "literature+history+cyber_ai packs + chat_rag uploads; NO institutional_books",
    }
    COMBINED_INDEX.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return COMBINED_INDEX


class ChatRetriever:
    def __init__(self, index_path: Optional[Path] = None, k: int = 3, max_chars: int = 1200):
        path = Path(index_path) if index_path else build_combined_index()
        self.mem = PackMemory(path)
        self.k = k
        self.max_chars = max_chars
        self.index_path = path

    def search(self, query: str, k: Optional[int] = None) -> List[dict]:
        return self.mem.search(query, k=k or self.k)

    def context(self, query: str, k: Optional[int] = None, max_chars: Optional[int] = None) -> str:
        return self.mem.context(query, k=k or self.k, max_chars=max_chars or self.max_chars)

    def augment(self, prompt: str, query: Optional[str] = None) -> str:
        q = query if query is not None else prompt
        ctx = self.context(q)
        if not ctx:
            return prompt
        return f"Context:\n{ctx}\n\n{prompt}"

    def info(self) -> dict:
        return {
            "index": str(self.index_path),
            "n_chunks": len(self.mem.chunks),
            "k": self.k,
            "max_chars": self.max_chars,
            "uploads": len(load_uploads()),
        }
