"""Cheap extractive answer-span scorer over retrieved passages.

Used as hybrid fallback when the LM rarely copies facts out of context.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path("/workspace/jarvis_kaggle")))
from model.memory_rag import tokenize  # noqa: E402

STOP = {
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is", "are",
    "was", "were", "be", "by", "what", "who", "when", "where", "which", "how",
    "did", "do", "does", "that", "this", "with", "from", "as", "it", "at",
}


def _query_terms(q: str) -> List[str]:
    return [t for t in tokenize(q) if t not in STOP and len(t) > 1]


def _sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if len(p.strip()) >= 8]


def _candidate_spans(sentence: str, question: str) -> List[str]:
    spans: List[str] = []
    patterns = [
        r"(?:is|are|was|were)\s+([A-Z][\w'’\-\.\s]{1,60}?)(?:[.!,;\n]|$)",
        r"([A-Z][\w'’\-]+(?:\s+[A-Z][\w'’\-]+){0,3})\s+(?:wrote|authored|discovered|invented|founded)",
        r"(?:written by|authored by|discovered by|invented by)\s+([A-Z][\w'’\-\.\s]{1,60}?)(?:[.!,;\n]|$)",
        r"(?:wrote|authored|discovered|invented|founded|began|started)\s+([A-Z][\w'’\-\.\s]{1,60}?)(?:[.!,;\n]|$)",
        r"([A-Z][\w'’\-]+(?:\s+[A-Z][\w'’\-]+){0,4})",
        r"\b((?:1[0-9]{3}|20[0-2][0-9]))\b",
        r"\b(\d+(?:\.\d+)?)\b",
    ]
    for pat in patterns:
        for m in re.finditer(pat, sentence):
            s = m.group(1).strip(" .,;:\"'")
            if 1 <= len(s) <= 80:
                spans.append(s)
    if 8 <= len(sentence) <= 120:
        spans.append(sentence.rstrip("."))
    seen = set()
    out = []
    for s in spans:
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def score_span(span: str, sentence: str, q_terms: List[str], bm25_hit_score: float) -> float:
    st = tokenize(span)
    sent_t = set(tokenize(sentence))
    if not st:
        return 0.0
    qset = set(q_terms)
    overlap_q = sum(1 for t in st if t in qset) / max(1, len(st))
    cov = sum(1 for t in q_terms if t in sent_t) / max(1, len(q_terms))
    brevity = 1.0 / (1.0 + max(0, len(st) - 4) * 0.15)
    penalty = 0.15 if overlap_q > 0.6 else 1.0
    return float(bm25_hit_score) * (0.35 + 0.65 * cov) * brevity * penalty


def best_extractive(
    question: str,
    hits: List[dict],
    min_score: float = 1.5,
) -> Optional[dict]:
    """Return best {span, sentence, source, score, passage_score} or None."""
    q_terms = _query_terms(question)
    if not hits or not q_terms:
        return None
    best = None
    for h in hits:
        text = h.get("text") or ""
        pscore = float(h.get("score") or 0.0)
        for sent in _sentences(text):
            sent_t = set(tokenize(sent))
            if sum(1 for t in q_terms if t in sent_t) < 1:
                continue
            for span in _candidate_spans(sent, question):
                sc = score_span(span, sent, q_terms, pscore)
                if best is None or sc > best["score"]:
                    best = {
                        "span": span,
                        "sentence": sent,
                        "source": h.get("source"),
                        "chunk_id": h.get("id"),
                        "score": sc,
                        "passage_score": pscore,
                    }
    if best is None or best["score"] < min_score:
        return None
    return best


def format_hybrid(extractive: Optional[dict], generation: str, mode_label: str = "hybrid") -> str:
    parts = []
    if extractive:
        parts.append(
            f"[EXTRACTIVE from {extractive.get('source')}|score={extractive['score']:.2f}]\n"
            f"{extractive['span']}"
        )
        parts.append(f"[SUPPORTING SENTENCE]\n{extractive['sentence']}")
    else:
        parts.append("[EXTRACTIVE]\n(none above threshold)")
    parts.append(f"[MODEL CONTINUATION]\n{generation}")
    parts.append(f"[MODE] {mode_label}")
    return "\n\n".join(parts)
