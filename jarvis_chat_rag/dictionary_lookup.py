"""Open WordNet (Princeton / NLTK) dictionary lookup for JARVIS chat.

Training-free: exact lemma lookup + short definition cards that can also be
uploaded into the BM25 knowledge store for 'what does X mean' RAG queries.
License: WordNet 3.0 license (Princeton); attribution required.
https://wordnet.princeton.edu/license-and-commercial-use
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional

HERE = Path(__file__).resolve().parent
DICT_JSONL = HERE / "knowledge" / "dictionary" / "wordnet_defs.jsonl"
DICT_META = HERE / "knowledge" / "dictionary" / "meta.json"

POS_NAME = {"n": "noun", "v": "verb", "a": "adjective", "s": "adjective satellite", "r": "adverb"}


def _ensure_wordnet():
    import sys
    from pathlib import Path as _P
    # user-site install (venv pip is flaky on this box)
    us = "/home/box/.local/lib/python3.13/site-packages"
    if us not in sys.path:
        sys.path.insert(0, us)
    from nltk.corpus import wordnet as wn  # noqa: WPS440
    try:
        wn.synsets("test")
    except LookupError:
        import nltk
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
    return wn


def lookup(word: str, max_senses: int = 5) -> List[dict]:
    """Return up to max_senses WordNet sense dicts for a lemma."""
    wn = _ensure_wordnet()
    lemma = (word or "").strip().lower().replace(" ", "_")
    if not lemma:
        return []
    out = []
    for ss in wn.synsets(lemma)[:max_senses]:
        synonyms = sorted({l.name().replace("_", " ") for l in ss.lemmas()})
        out.append(
            {
                "lemma": lemma.replace("_", " "),
                "pos": POS_NAME.get(ss.pos(), ss.pos()),
                "definition": ss.definition(),
                "examples": list(ss.examples())[:2],
                "synonyms": [s for s in synonyms if s != lemma.replace("_", " ")][:8],
                "synset": ss.name(),
            }
        )
    return out


def format_lookup(word: str, senses: Optional[List[dict]] = None) -> str:
    senses = senses if senses is not None else lookup(word)
    if not senses:
        return f"[DICT] No WordNet entry for {word!r}."
    parts = [f"[DICT WordNet] {word}"]
    for i, s in enumerate(senses, 1):
        line = f"{i}. ({s['pos']}) {s['definition']}"
        if s.get("examples"):
            line += f"  e.g. {s['examples'][0]!r}"
        if s.get("synonyms"):
            line += f"  syn: {', '.join(s['synonyms'][:5])}"
        parts.append(line)
    parts.append("[SOURCE] Princeton WordNet 3.0 via NLTK")
    return "\n".join(parts)


_MEAN_RE = re.compile(
    r"^\s*(?:what\s+does\s+(.+?)\s+mean\??|define\s+(.+?)\??|meaning\s+of\s+(.+?)\??|"
    r"what\s+is\s+(?:the\s+)?(?:definition|meaning)\s+of\s+(.+?)\??)\s*$",
    re.I,
)


def extract_define_query(text: str) -> Optional[str]:
    """If the user asked a definition-style question, return the target word/phrase."""
    m = _MEAN_RE.match((text or "").strip().rstrip("."))
    if not m:
        return None
    for g in m.groups():
        if g:
            return g.strip(" \"'").strip()
    return None


def answer_define(question: str) -> Optional[str]:
    """Direct dictionary path (bypasses the LM). Returns None if not a define-query."""
    w = extract_define_query(question)
    if not w:
        return None
    return format_lookup(w)


def build_definition_pack(
    lemmas: Optional[List[str]] = None,
    out_path: Path = DICT_JSONL,
    limit: int = 5000,
) -> Path:
    """Write short definition cards for BM25 RAG upload.

    Default: top frequent English lemmas from a small seed list + WordNet
    closures around STEM/school words (cheap, no huge dump).
    """
    wn = _ensure_wordnet()
    seed = lemmas or [
        "photosynthesis", "mitochondria", "neuron", "synapse", "algorithm",
        "democracy", "constitution", "gravity", "velocity", "molecule",
        "atom", "cell", "evolution", "hypothesis", "variable", "integer",
        "derivative", "integral", "circuit", "voltage", "current", "resistance",
        "capital", "empire", "republic", "treaty", "orbit", "planet", "galaxy",
        "protein", "enzyme", "chromosome", "genome", "virus", "bacteria",
        "osmosis", "diffusion", "catalyst", "isotope", "compound", "element",
        "metaphor", "simile", "irony", "allegory", "sonnet", "tragedy",
        "renaissance", "industrial", "revolution", "colony", "federation",
        "encryption", "firewall", "protocol", "latency", "bandwidth",
        "attention", "transformer", "embedding", "gradient", "loss",
        "corpus", "callosum", "cortex", "hippocampus", "myelin", "axon",
        "dendrite", "oscillator", "reservoir", "entropy", "perplexity",
    ]
    seen = set()
    rows = []
    queue = list(seed)
    while queue and len(rows) < limit:
        w = queue.pop(0)
        key = w.lower().replace(" ", "_")
        if key in seen:
            continue
        seen.add(key)
        synsets = wn.synsets(key)
        if not synsets:
            continue
        for ss in synsets[:3]:
            text = (
                f"The word '{key.replace('_', ' ')}' ({POS_NAME.get(ss.pos(), ss.pos())}) "
                f"means: {ss.definition()}."
            )
            if ss.examples():
                text += f" Example: {ss.examples()[0]}"
            rows.append(
                {
                    "id": f"wn:{ss.name()}",
                    "source": "wordnet",
                    "lemma": key.replace("_", " "),
                    "text": text,
                    "definition": ss.definition(),
                    "pos": POS_NAME.get(ss.pos(), ss.pos()),
                }
            )
            # light expansion via related lemmas
            for lem in ss.lemmas()[:4]:
                name = lem.name()
                if name.lower() not in seen and len(queue) < limit:
                    queue.append(name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {
        "n_cards": len(rows),
        "n_lemmas": len(seen),
        "path": str(out_path),
        "license": "Princeton WordNet 3.0 (see https://wordnet.princeton.edu/license-and-commercial-use)",
        "note": "Definition cards for BM25 RAG; also use dictionary_lookup.answer_define for direct path",
    }
    DICT_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out_path


def seed_dictionary_into_uploads(force: bool = False, limit: int = 400) -> int:
    """Append WordNet definition cards into uploaded_facts.jsonl (idempotent by id).

    Caps at `limit` cards so BM25 is not drowned in dictionary text.
    Full pack stays at knowledge/dictionary/wordnet_defs.jsonl for direct lookup.
    """
    from retriever import append_upload, load_uploads, build_combined_index

    if not DICT_JSONL.exists():
        build_definition_pack()
    have = {r.get("id") for r in load_uploads()}
    n = 0
    for line in DICT_JSONL.read_text(encoding="utf-8").splitlines():
        if n >= limit:
            break
        if not line.strip():
            continue
        o = json.loads(line)
        if o.get("id") in have and not force:
            # already present; still counts toward cap so we don't walk forever
            n += 1
            continue
        if o.get("id") in have:
            continue
        append_upload(
            o["text"],
            source="wordnet",
            meta={"id": o["id"], "lemma": o.get("lemma"), "answer": o.get("definition")},
        )
        n += 1
    if n:
        build_combined_index(force=True)
    return n


if __name__ == "__main__":
    import sys
    w = sys.argv[1] if len(sys.argv) > 1 else "photosynthesis"
    print(format_lookup(w))
    p = build_definition_pack()
    print("wrote", p, "cards", sum(1 for _ in p.open()))
