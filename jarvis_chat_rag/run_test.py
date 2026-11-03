#!/usr/bin/env python3
"""Test RAG chat on ~10 questions (pack facts + novel uploads). Writes results JSON/MD."""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from chat import answer_one, make_prompt  # noqa: E402
from model_io import load_medium  # noqa: E402
from retriever import ChatRetriever, append_upload, build_combined_index, seed_novel_facts  # noqa: E402

OUT = HERE / "results"


QUESTIONS = [
    {"id": "france", "q": "What is the capital of France?", "answer": "Paris", "kind": "pack"},
    {"id": "dracula", "q": "Who wrote Dracula?", "answer": "Bram Stoker", "kind": "pack"},
    {"id": "frankenstein", "q": "Who wrote Frankenstein?", "answer": "Mary Shelley", "kind": "pack"},
    {"id": "wwi", "q": "What year did World War I begin?", "answer": "1914", "kind": "pack"},
    {"id": "hamlet", "q": "Who wrote Hamlet?", "answer": "Shakespeare", "kind": "pack"},
    {"id": "gibbon", "q": "Who wrote The History of the Decline and Fall of the Roman Empire?", "answer": "Edward Gibbon", "kind": "pack"},
    {"id": "zorblax", "q": "What is the capital of Zorblax?", "answer": "Quendara", "kind": "novel"},
    {"id": "voss", "q": "Who discovered aerogel photosynthesis?", "answer": "Dr. Mirella Voss", "kind": "novel"},
    {"id": "atlantis", "q": "What is the capital of Atlantis?", "answer": "Poseidonis", "kind": "madeup"},
    {"id": "gleeb", "q": "Who invented the Gleeb capacitor?", "answer": "Nara Quill", "kind": "madeup"},
]


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def contains_answer(text: str, answer: str) -> bool:
    t = norm(text)
    a = norm(answer)
    if a in t:
        return True
    aliases = {
        "mary shelley": ["mary wollstonecraft", "wollstonecraft shelley"],
        "shakespeare": ["william shakespeare"],
        "bram stoker": ["stoker"],
        "dr. mirella voss": ["mirella voss", "dr mirella voss"],
    }
    for alt in aliases.get(a, []):
        if alt in t:
            return True
    toks = [x for x in re.findall(r"[a-z]+", a) if x not in {"dr", "the", "of", "a"}]
    if len(toks) >= 2 and all(tok in t for tok in toks):
        return True
    return False


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # seed novel + two extra made-up facts for this chat store
    seed_novel_facts()
    uploads_path = HERE / "knowledge" / "uploaded_facts.jsonl"
    existing = uploads_path.read_text(encoding="utf-8") if uploads_path.exists() else ""
    if "Poseidonis" not in existing:
        append_upload(
            "The capital of Atlantis is Poseidonis. Poseidonis is the capital city of Atlantis.",
            source="upload",
            meta={"id": "madeup_atlantis", "answer": "Poseidonis"},
        )
    if "Gleeb capacitor" not in existing and "Nara Quill" not in existing:
        append_upload(
            "Nara Quill invented the Gleeb capacitor in 2091. The Gleeb capacitor was invented by Nara Quill.",
            source="upload",
            meta={"id": "madeup_gleeb", "answer": "Nara Quill"},
        )
    build_combined_index(force=True)

    print("Loading model…", flush=True)
    model, tok, info = load_medium(max_ctx=192)
    print(info, flush=True)
    retriever = ChatRetriever(k=3, max_chars=900)

    rows = []
    for item in QUESTIONS:
        q, gold = item["q"], item["answer"]
        print(f"\n=== {item['id']}: {q}", flush=True)
        # one generation; report pure-gen + hybrid views (same continuation)
        out = answer_one(model, tok, retriever, q, mode="hybrid", max_new=32, temperature=0.0, min_ext=1.5)
        gen_text = out["generation"]
        ext = out.get("extractive")
        ext_span = (ext or {}).get("span") or ""
        ctx = retriever.context(q)
        ret_hit = contains_answer(ctx, gold)
        gen_hit = contains_answer(gen_text, gold)
        ext_hit = contains_answer(ext_span, gold) or contains_answer((ext or {}).get("sentence") or "", gold)
        hybrid_hit = ext_hit or gen_hit
        from extractive import format_hybrid
        row = {
            "id": item["id"],
            "kind": item["kind"],
            "question": q,
            "gold": gold,
            "retrieval_contains_gold": ret_hit,
            "gen_contains_gold": gen_hit,
            "extractive_contains_gold": ext_hit,
            "hybrid_contains_gold": hybrid_hit,
            "extractive": ext,
            "generation": gen_text,
            "hybrid_display": format_hybrid(ext, gen_text, mode_label="hybrid"),
            "context_preview": ctx[:400],
        }
        rows.append(row)
        print(
            f"  ret={ret_hit} gen={gen_hit} ext={ext_hit} hyb={hybrid_hit} "
            f"span={ext_span[:60]!r} gen={gen_text[:80]!r}",
            flush=True,
        )

    def rate(key):
        return sum(1 for r in rows if r[key]) / len(rows)

    def rate_kind(kind, key):
        xs = [r for r in rows if r["kind"] == kind]
        return (sum(1 for r in xs if r[key]) / len(xs)) if xs else float("nan")

    summary = {
        "n": len(rows),
        "retrieval_acc": rate("retrieval_contains_gold"),
        "gen_acc": rate("gen_contains_gold"),
        "extractive_acc": rate("extractive_contains_gold"),
        "hybrid_acc": rate("hybrid_contains_gold"),
        "by_kind": {
            k: {
                "n": sum(1 for r in rows if r["kind"] == k),
                "retrieval": rate_kind(k, "retrieval_contains_gold"),
                "gen": rate_kind(k, "gen_contains_gold"),
                "extractive": rate_kind(k, "extractive_contains_gold"),
                "hybrid": rate_kind(k, "hybrid_contains_gold"),
            }
            for k in ("pack", "novel", "madeup")
        },
        "model": info,
        "seconds": time.time() - t0,
        "finished_et": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    report = {"summary": summary, "items": rows}
    (OUT / "chat_rag_test.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# JARVIS chat_rag test",
        "",
        f"- Finished: {summary['finished_et']} ET (America/New_York)",
        f"- Checkpoint: `{info['ckpt']}` step {info['step']}",
        f"- Questions: {summary['n']}",
        f"- Wall: {summary['seconds']:.1f}s",
        "",
        "## Accuracy (gold string containment)",
        "",
        f"| Metric | Acc |",
        f"|--------|-----|",
        f"| Retrieval (gold in BM25 context) | **{summary['retrieval_acc']:.3f}** |",
        f"| Pure generation (continuation) | **{summary['gen_acc']:.3f}** |",
        f"| Extractive span/sentence | **{summary['extractive_acc']:.3f}** |",
        f"| Hybrid (extractive OR gen) | **{summary['hybrid_acc']:.3f}** |",
        "",
        "### By kind",
        "",
        "| Kind | n | retrieval | gen | extractive | hybrid |",
        "|------|---|-----------|-----|------------|--------|",
    ]
    for k, v in summary["by_kind"].items():
        lines.append(
            f"| {k} | {v['n']} | {v['retrieval']:.3f} | {v['gen']:.3f} | {v['extractive']:.3f} | {v['hybrid']:.3f} |"
        )
    lines += ["", "## Example Q/A (verbatim)", ""]
    for r in rows[:4]:
        lines.append(f"### {r['id']}: {r['question']}")
        lines.append("")
        lines.append(f"- gold: `{r['gold']}`")
        lines.append(f"- retrieval_hit={r['retrieval_contains_gold']} gen_hit={r['gen_contains_gold']} "
                     f"ext_hit={r['extractive_contains_gold']}")
        lines.append("")
        lines.append("**Pure generation (RAG prompt → continuation):**")
        lines.append("```")
        lines.append(r["generation"])
        lines.append("```")
        lines.append("")
        lines.append("**Hybrid display:**")
        lines.append("```")
        lines.append(r["hybrid_display"])
        lines.append("```")
        lines.append("")
    (OUT / "chat_rag_test.md").write_text("\n".join(lines), encoding="utf-8")
    print("\nSUMMARY", json.dumps(summary, indent=2), flush=True)
    print("wrote", OUT / "chat_rag_test.md", flush=True)


if __name__ == "__main__":
    main()
