#!/usr/bin/env python3
"""Build ~100 labeled pack+intent questions for Cognitive Router benchmark."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "questions.jsonl"

# Reuse gold from full_eval + expand.
BASE = [
    {"id": "france", "q": "What is the capital of France?", "answer": "Paris", "pack": "uploads", "kind": "pack"},
    {"id": "dracula", "q": "Who wrote Dracula?", "answer": "Bram Stoker", "pack": "uploads", "kind": "pack"},
    {"id": "frankenstein", "q": "Who wrote Frankenstein?", "answer": "Mary Shelley", "pack": "uploads", "kind": "pack"},
    {"id": "wwi", "q": "What year did World War I begin?", "answer": "1914", "pack": "uploads", "kind": "pack"},
    {"id": "hamlet", "q": "Who wrote Hamlet?", "answer": "Shakespeare", "pack": "uploads", "kind": "pack"},
    {"id": "gibbon", "q": "Who wrote The History of the Decline and Fall of the Roman Empire?", "answer": "Edward Gibbon", "pack": "uploads", "kind": "pack"},
    {"id": "herodotus", "q": "Who wrote The History of Herodotus — Volume 1?", "answer": "Herodotus", "pack": "uploads", "kind": "pack"},
    {"id": "thucydides", "q": "Who wrote The History of the Peloponnesian War?", "answer": "Thucydides", "pack": "uploads", "kind": "pack"},
    {"id": "paine", "q": "Who wrote Common Sense?", "answer": "Thomas Paine", "pack": "uploads", "kind": "pack"},
    {"id": "jefferson", "q": "Who wrote The Declaration of Independence of the United States of America?", "answer": "Thomas Jefferson", "pack": "uploads", "kind": "pack"},
    {"id": "zorblax", "q": "What is the capital of Zorblax?", "answer": "Quendara", "pack": "uploads", "kind": "novel"},
    {"id": "voss", "q": "Who discovered aerogel photosynthesis?", "answer": "Dr. Mirella Voss", "pack": "uploads", "kind": "novel"},
    {"id": "wren", "q": "In what year did the MV Starlight Wren launch?", "answer": "2147", "pack": "uploads", "kind": "novel"},
    {"id": "kestrel", "q": "What does the Kestrel Protocol use?", "answer": "three-phase lattice keys", "pack": "uploads", "kind": "novel"},
    {"id": "amberfall", "q": "Who founded the city of Amberfall?", "answer": "Queen Lysara", "pack": "uploads", "kind": "novel"},
    {"id": "xylith", "q": "What is the atomic number of Xylith?", "answer": "149", "pack": "uploads", "kind": "novel"},
    {"id": "okorie", "q": "Who wrote Bright Circadian?", "answer": "Hana Okorie", "pack": "uploads", "kind": "novel"},
    {"id": "nexora", "q": "How many moons does Nexora-4 have?", "answer": "seven", "pack": "uploads", "kind": "novel"},
    {"id": "helixwind", "q": "Who is the CEO of Helixwind?", "answer": "Priya Dunn", "pack": "uploads", "kind": "novel"},
    {"id": "marlowe", "q": "Who invented the MARLOWE algorithm?", "answer": "Eli Navarro", "pack": "uploads", "kind": "novel"},
    {"id": "glaucite", "q": "What color is glaucite?", "answer": "blue-green", "pack": "uploads", "kind": "novel"},
    {"id": "northmere", "q": "What is the motto of the University of Northmere?", "answer": "Lux in limine", "pack": "uploads", "kind": "novel"},
    {"id": "atlantis", "q": "What is the capital of Atlantis?", "answer": "Poseidonis", "pack": "uploads", "kind": "madeup"},
    {"id": "gleeb", "q": "Who invented the Gleeb capacitor?", "answer": "Nara Quill", "pack": "uploads", "kind": "madeup"},
    {"id": "def_neuron", "q": "What does neuron mean?", "answer": "nerve", "pack": "dictionary", "kind": "dict"},
    {"id": "def_synapse", "q": "What does synapse mean?", "answer": "junction", "pack": "dictionary", "kind": "dict"},
    {"id": "def_myelin", "q": "Define myelin", "answer": "fatty", "pack": "dictionary", "kind": "dict"},
    {"id": "def_photosynthesis", "q": "What does photosynthesis mean?", "answer": "synthesis", "pack": "dictionary", "kind": "dict"},
    {"id": "def_algorithm", "q": "What does algorithm mean?", "answer": "rule", "pack": "dictionary", "kind": "dict"},
    {"id": "def_entropy", "q": "What does entropy mean?", "answer": "uncertainty", "pack": "dictionary", "kind": "dict"},
]

# Extra dictionary
DICT_EXTRA = [
    ("def_axon", "What does axon mean?", "nerve"),
    ("def_dendrite", "Define dendrite", "branch"),
    ("def_cortex", "What does cortex mean?", "layer"),
    ("def_hippocampus", "What does hippocampus mean?", "brain"),
    ("def_quantum", "What does quantum mean?", "discrete"),
    ("def_latency", "Define latency", "delay"),
    ("def_router", "What does router mean?", "device"),
    ("def_graph", "What does graph mean?", "diagram"),
    ("def_precision", "Define precision", "accuracy"),
    ("def_recall", "What does recall mean?", "memory"),
]

# Book / literature style (gold pack = books) — answers that live in literature/history packs
BOOKS = [
    {"id": "book_romeo", "q": "According to the book, who is Romeo's love interest in Romeo and Juliet?", "answer": "Juliet", "pack": "books", "kind": "books"},
    {"id": "book_passage_odyssey", "q": "In the novel The Odyssey, who is the main hero?", "answer": "Odysseus", "pack": "books", "kind": "books"},
    {"id": "book_iliad", "q": "According to the book The Iliad, who is Achilles?", "answer": "warrior", "pack": "books", "kind": "books"},
    {"id": "book_pride", "q": "In the novel Pride and Prejudice, who wrote it — wait no, chapter mention of Elizabeth", "answer": "Elizabeth", "pack": "books", "kind": "books"},
    {"id": "book_chapter_moby", "q": "What passage mentions Captain Ahab in Moby Dick chapter?", "answer": "Ahab", "pack": "books", "kind": "books"},
    {"id": "book_history_rome", "q": "According to the book on Roman history, what was the Roman Empire?", "answer": "Rome", "pack": "books", "kind": "books"},
    {"id": "book_passage_ai", "q": "According to the book passage on cyber_ai, what is machine learning?", "answer": "learning", "pack": "books", "kind": "books"},
    {"id": "book_shakespeare_passage", "q": "In the novel / passage of Macbeth, who is the Scottish king?", "answer": "Macbeth", "pack": "books", "kind": "books"},
]

# Extra novel / upload paraphrases
NOVEL_EXTRA = [
    {"id": "zorblax2", "q": "Name the capital city of Zorblax", "answer": "Quendara", "pack": "uploads", "kind": "novel"},
    {"id": "voss2", "q": "Aerogel photosynthesis was discovered by whom?", "answer": "Dr. Mirella Voss", "pack": "uploads", "kind": "novel"},
    {"id": "atlantis2", "q": "Atlantis has which capital?", "answer": "Poseidonis", "pack": "uploads", "kind": "madeup"},
    {"id": "gleeb2", "q": "The Gleeb capacitor was invented by which person?", "answer": "Nara Quill", "pack": "uploads", "kind": "madeup"},
    {"id": "xylith2", "q": "Xylith atomic number is what?", "answer": "149", "pack": "uploads", "kind": "novel"},
    {"id": "amberfall2", "q": "Amberfall was founded by which queen?", "answer": "Queen Lysara", "pack": "uploads", "kind": "novel"},
    {"id": "nexora2", "q": "Nexora-4 moon count?", "answer": "seven", "pack": "uploads", "kind": "novel"},
    {"id": "marlowe2", "q": "MARLOWE algorithm inventor name?", "answer": "Eli Navarro", "pack": "uploads", "kind": "novel"},
    {"id": "helixwind2", "q": "Helixwind CEO is who?", "answer": "Priya Dunn", "pack": "uploads", "kind": "novel"},
    {"id": "glaucite2", "q": "Glaucite color description?", "answer": "blue-green", "pack": "uploads", "kind": "novel"},
    {"id": "northmere2", "q": "University of Northmere motto text?", "answer": "Lux in limine", "pack": "uploads", "kind": "novel"},
    {"id": "wren2", "q": "MV Starlight Wren launch year?", "answer": "2147", "pack": "uploads", "kind": "novel"},
    {"id": "kestrel2", "q": "Kestrel Protocol key type?", "answer": "three-phase lattice keys", "pack": "uploads", "kind": "novel"},
    {"id": "okorie2", "q": "Bright Circadian author?", "answer": "Hana Okorie", "pack": "uploads", "kind": "novel"},
]

# Intent-labeled (debug / explain / rewrite) — pack still needed for retrieval routing;
# intent is a separate label for the Cognitive Router's intent head.
INTENTS = [
    # debug
    {"id": "int_dbg1", "q": "My Python function throws IndexError on line 12, help me debug this code", "answer": "IndexError", "pack": "uploads", "kind": "intent", "intent": "debug"},
    {"id": "int_dbg2", "q": "Debug why this loop never terminates", "answer": "loop", "pack": "uploads", "kind": "intent", "intent": "debug"},
    {"id": "int_dbg3", "q": "There's a bug in my router scoring — fix the crash", "answer": "bug", "pack": "uploads", "kind": "intent", "intent": "debug"},
    {"id": "int_dbg4", "q": "Why am I getting TypeError: NoneType in this stack trace?", "answer": "TypeError", "pack": "uploads", "kind": "intent", "intent": "debug"},
    {"id": "int_dbg5", "q": "Help debug the failing unit test assertion", "answer": "test", "pack": "uploads", "kind": "intent", "intent": "debug"},
    {"id": "int_dbg6", "q": "This BM25 search returns empty — find the bug", "answer": "BM25", "pack": "uploads", "kind": "intent", "intent": "debug"},
    {"id": "int_dbg7", "q": "Null pointer exception when loading the model checkpoint", "answer": "checkpoint", "pack": "uploads", "kind": "intent", "intent": "debug"},
    {"id": "int_dbg8", "q": "Race condition in my async worker — debug please", "answer": "race", "pack": "uploads", "kind": "intent", "intent": "debug"},
    # explain
    {"id": "int_exp1", "q": "Explain how BM25 ranking works", "answer": "BM25", "pack": "books", "kind": "intent", "intent": "explain"},
    {"id": "int_exp2", "q": "Can you explain what a neural synapse is?", "answer": "synapse", "pack": "dictionary", "kind": "intent", "intent": "explain"},
    {"id": "int_exp3", "q": "Explain the difference between precision and recall", "answer": "precision", "pack": "dictionary", "kind": "intent", "intent": "explain"},
    {"id": "int_exp4", "q": "What does this code do? Please explain the algorithm step by step", "answer": "algorithm", "pack": "dictionary", "kind": "intent", "intent": "explain"},
    {"id": "int_exp5", "q": "Explain PageRank in simple terms", "answer": "PageRank", "pack": "books", "kind": "intent", "intent": "explain"},
    {"id": "int_exp6", "q": "Help me understand how pack routing chooses a knowledge source", "answer": "routing", "pack": "uploads", "kind": "intent", "intent": "explain"},
    {"id": "int_exp7", "q": "Explain entropy in information theory", "answer": "entropy", "pack": "dictionary", "kind": "intent", "intent": "explain"},
    {"id": "int_exp8", "q": "Walk me through how context-aware decoding works", "answer": "context", "pack": "uploads", "kind": "intent", "intent": "explain"},
    # rewrite
    {"id": "int_rw1", "q": "Rewrite this function to be more readable", "answer": "function", "pack": "uploads", "kind": "intent", "intent": "rewrite"},
    {"id": "int_rw2", "q": "Refactor and rewrite the router module with clearer names", "answer": "router", "pack": "uploads", "kind": "intent", "intent": "rewrite"},
    {"id": "int_rw3", "q": "Please rewrite this paragraph in simpler English", "answer": "paragraph", "pack": "uploads", "kind": "intent", "intent": "rewrite"},
    {"id": "int_rw4", "q": "Convert / rewrite this code from Python 2 to Python 3", "answer": "Python", "pack": "uploads", "kind": "intent", "intent": "rewrite"},
    {"id": "int_rw5", "q": "Rewrite the SQL query to use joins instead of subqueries", "answer": "SQL", "pack": "uploads", "kind": "intent", "intent": "rewrite"},
    {"id": "int_rw6", "q": "Make a cleaner rewrite of this class without changing behavior", "answer": "class", "pack": "uploads", "kind": "intent", "intent": "rewrite"},
    {"id": "int_rw7", "q": "Rewrite my commit message to be concise", "answer": "commit", "pack": "uploads", "kind": "intent", "intent": "rewrite"},
    {"id": "int_rw8", "q": "Restructure and rewrite the README for clarity", "answer": "README", "pack": "uploads", "kind": "intent", "intent": "rewrite"},
]

# More pack_fact style from uploads if available
def load_upload_questions():
    path = Path("/workspace/jarvis/chat_rag/knowledge/uploaded_facts.jsonl")
    rows = []
    if not path.exists():
        return rows
    seen = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        q = o.get("question")
        a = o.get("answer")
        oid = o.get("id") or ""
        if not q or not a or oid in seen:
            continue
        src = (o.get("source") or "").strip()
        if src == "wordnet":
            pack = "dictionary"
            kind = "dict"
        elif src in {"literature", "history", "cyber_ai"}:
            pack = "books"
            kind = "books"
        else:
            pack = "uploads"
            kind = "novel" if src == "novel" else "pack"
        # skip if already in BASE by question text
        rows.append({"id": f"up_{oid}", "q": q, "answer": a, "pack": pack, "kind": kind, "source": src})
        seen.add(oid)
    return rows


def main():
    items = []
    for r in BASE:
        items.append({**r, "intent": None})
    for i, (iid, q, a) in enumerate(DICT_EXTRA):
        items.append({"id": iid, "q": q, "answer": a, "pack": "dictionary", "kind": "dict", "intent": None})
    for r in BOOKS:
        items.append({**r, "intent": None})
    for r in NOVEL_EXTRA:
        items.append({**r, "intent": None})
    for r in INTENTS:
        items.append(r)

    # Add upload-derived that aren't duplicates
    have_q = {re.sub(r"\s+", " ", x["q"].lower().strip()) for x in items}
    for r in load_upload_questions():
        key = re.sub(r"\s+", " ", r["q"].lower().strip())
        if key in have_q:
            continue
        items.append({**r, "intent": None})
        have_q.add(key)
        if len(items) >= 120:
            break

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for r in items:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    packs = Counter(r["pack"] for r in items)
    intents = Counter(r.get("intent") for r in items)
    print(f"wrote {len(items)} → {OUT}")
    print("packs", dict(packs))
    print("intents", dict(intents))


if __name__ == "__main__":
    main()
