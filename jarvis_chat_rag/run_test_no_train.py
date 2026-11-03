#!/usr/bin/env python3
"""Expanded ~30Q eval of training-free RAG modes: gen / copybias / cad / hybrid (+ dict).

Honest small-n test. Writes results under results/no_train_*.
"""
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

from answer_modes import answer_one_mode  # noqa: E402
from model_io import load_medium  # noqa: E402
from retriever import (  # noqa: E402
    ChatRetriever,
    append_upload,
    build_combined_index,
    seed_novel_facts,
)
from dictionary_lookup import build_definition_pack, seed_dictionary_into_uploads, answer_define  # noqa: E402

OUT = HERE / "results"

# ~30 questions from uploaded packs / novel / made-up / a few define-queries
QUESTIONS = [
    # pack / fact cards
    {"id": "france", "q": "What is the capital of France?", "answer": "Paris", "kind": "pack"},
    {"id": "dracula", "q": "Who wrote Dracula?", "answer": "Bram Stoker", "kind": "pack"},
    {"id": "frankenstein", "q": "Who wrote Frankenstein?", "answer": "Mary Shelley", "kind": "pack"},
    {"id": "wwi", "q": "What year did World War I begin?", "answer": "1914", "kind": "pack"},
    {"id": "hamlet", "q": "Who wrote Hamlet?", "answer": "Shakespeare", "kind": "pack"},
    {"id": "gibbon", "q": "Who wrote The History of the Decline and Fall of the Roman Empire?", "answer": "Edward Gibbon", "kind": "pack"},
    {"id": "herodotus", "q": "Who wrote The History of Herodotus — Volume 1?", "answer": "Herodotus", "kind": "pack"},
    {"id": "thucydides", "q": "Who wrote The History of the Peloponnesian War?", "answer": "Thucydides", "kind": "pack"},
    {"id": "paine", "q": "Who wrote Common Sense?", "answer": "Thomas Paine", "kind": "pack"},
    {"id": "jefferson", "q": "Who wrote The Declaration of Independence of the United States of America?", "answer": "Thomas Jefferson", "kind": "pack"},
    # novel facts
    {"id": "zorblax", "q": "What is the capital of Zorblax?", "answer": "Quendara", "kind": "novel"},
    {"id": "voss", "q": "Who discovered aerogel photosynthesis?", "answer": "Dr. Mirella Voss", "kind": "novel"},
    {"id": "wren", "q": "In what year did the MV Starlight Wren launch?", "answer": "2147", "kind": "novel"},
    {"id": "kestrel", "q": "What does the Kestrel Protocol use?", "answer": "three-phase lattice keys", "kind": "novel"},
    {"id": "amberfall", "q": "Who founded the city of Amberfall?", "answer": "Queen Lysara", "kind": "novel"},
    {"id": "xylith", "q": "What is the atomic number of Xylith?", "answer": "149", "kind": "novel"},
    {"id": "okorie", "q": "Who wrote Bright Circadian?", "answer": "Hana Okorie", "kind": "novel"},
    {"id": "nexora", "q": "How many moons does Nexora-4 have?", "answer": "seven", "kind": "novel"},
    {"id": "helixwind", "q": "Who is the CEO of Helixwind?", "answer": "Priya Dunn", "kind": "novel"},
    {"id": "marlowe", "q": "Who invented the MARLOWE algorithm?", "answer": "Eli Navarro", "kind": "novel"},
    {"id": "glaucite", "q": "What color is glaucite?", "answer": "blue-green", "kind": "novel"},
    {"id": "northmere", "q": "What is the motto of the University of Northmere?", "answer": "Lux in limine", "kind": "novel"},
    # made-up uploads
    {"id": "atlantis", "q": "What is the capital of Atlantis?", "answer": "Poseidonis", "kind": "madeup"},
    {"id": "gleeb", "q": "Who invented the Gleeb capacitor?", "answer": "Nara Quill", "kind": "madeup"},
    # dictionary / define path (gold = key phrase from WordNet def)
    {"id": "def_neuron", "q": "What does neuron mean?", "answer": "nerve", "kind": "dict"},
    {"id": "def_synapse", "q": "What does synapse mean?", "answer": "junction", "kind": "dict"},
    {"id": "def_myelin", "q": "Define myelin", "answer": "fatty", "kind": "dict"},
    {"id": "def_photosynthesis", "q": "What does photosynthesis mean?", "answer": "synthesis", "kind": "dict"},
    {"id": "def_algorithm", "q": "What does algorithm mean?", "answer": "rule", "kind": "dict"},
    {"id": "def_entropy", "q": "What does entropy mean?", "answer": "uncertainty", "kind": "dict"},
]

MODES = ["gen", "copybias", "cad", "hybrid"]  # hybrid uses extractive OR gen; dict Qs also scored via answer_define


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
        "thomas jefferson": ["jefferson"],
        "thomas paine": ["paine"],
        "edward gibbon": ["gibbon"],
        "blue-green": ["blue green", "bluish green", "blue-green"],
        "three-phase lattice keys": ["lattice keys", "three-phase"],
        "lux in limine": ["lux in limine"],
    }
    for alt in aliases.get(a, []):
        if alt in t:
            return True
    toks = [x for x in re.findall(r"[a-z0-9]+", a) if x not in {"dr", "the", "of", "a", "an"}]
    if len(toks) >= 2 and all(tok in t for tok in toks):
        return True
    return False


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

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

    build_definition_pack()
    # seed a capped set of dict cards into BM25 (full 5k is fine but slower index)
    n_dict = seed_dictionary_into_uploads(force=False, limit=400)
    print(f"seeded {n_dict} new wordnet cards into uploads", flush=True)
    build_combined_index(force=True)

    print("Loading model…", flush=True)
    model, tok, info = load_medium(max_ctx=192)
    print(info, flush=True)
    retriever = ChatRetriever(k=3, max_chars=900)

    rows = []
    for item in QUESTIONS:
        q, gold = item["q"], item["answer"]
        print(f"\n=== {item['id']}: {q}", flush=True)
        ctx = retriever.context(q)
        ret_hit = contains_answer(ctx, gold)
        mode_out = {}
        for mode in MODES:
            # dict questions: also record direct WordNet path
            out = answer_one_mode(
                model, tok, retriever, q, mode=mode,
                max_new=28, temperature=0.0, min_ext=1.5,
                prefer_dict=(mode == "hybrid" and item["kind"] == "dict"),
                extractive_only=(mode == "hybrid"),
            )
            gen_text = out.get("generation") or ""
            ext = out.get("extractive")
            ext_span = (ext or {}).get("span") or ""
            ext_sent = (ext or {}).get("sentence") or ""
            display = out.get("display") or ""
            gen_hit = contains_answer(gen_text, gold)
            ext_hit = contains_answer(ext_span, gold) or contains_answer(ext_sent, gold)
            disp_hit = contains_answer(display, gold)
            if mode == "hybrid":
                hit = ext_hit or gen_hit or disp_hit
            else:
                hit = gen_hit
            mode_out[mode] = {
                "hit": hit,
                "generation": gen_text[:240],
                "display": display[:400],
                "extractive_span": ext_span[:120],
            }
            print(f"  {mode}: hit={hit} gen={gen_text[:70]!r}", flush=True)

        dict_direct = answer_define(q) if item["kind"] == "dict" else None
        dict_hit = contains_answer(dict_direct or "", gold) if dict_direct else False
        if item["kind"] == "dict":
            print(f"  dict_direct: hit={dict_hit}", flush=True)

        rows.append({
            "id": item["id"],
            "kind": item["kind"],
            "question": q,
            "gold": gold,
            "retrieval_contains_gold": ret_hit,
            "modes": mode_out,
            "dict_direct": dict_direct,
            "dict_direct_hit": dict_hit,
            "context_preview": ctx[:300],
        })

    def mode_acc(mode):
        return sum(1 for r in rows if r["modes"][mode]["hit"]) / len(rows)

    def mode_acc_kind(kind, mode):
        xs = [r for r in rows if r["kind"] == kind]
        return (sum(1 for r in xs if r["modes"][mode]["hit"]) / len(xs)) if xs else float("nan")

    dict_rows = [r for r in rows if r["kind"] == "dict"]
    summary = {
        "n": len(rows),
        "retrieval_acc": sum(1 for r in rows if r["retrieval_contains_gold"]) / len(rows),
        "by_mode": {m: mode_acc(m) for m in MODES},
        "dict_direct_acc": (sum(1 for r in dict_rows if r["dict_direct_hit"]) / len(dict_rows)) if dict_rows else None,
        "by_kind": {
            k: {
                "n": sum(1 for r in rows if r["kind"] == k),
                "retrieval": sum(1 for r in rows if r["kind"] == k and r["retrieval_contains_gold"]) / max(1, sum(1 for r in rows if r["kind"] == k)),
                **{m: mode_acc_kind(k, m) for m in MODES},
            }
            for k in ("pack", "novel", "madeup", "dict")
        },
        "model": info,
        "seconds": time.time() - t0,
        "finished_et": time.strftime("%Y-%m-%d %H:%M:%S"),
        "notes": "Small-n (~30). Gold = substring containment. CAD alpha=1.0, copy_bias=3.0.",
    }
    report = {"summary": summary, "items": rows}
    (OUT / "no_train_test.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# JARVIS training-free RAG modes test",
        "",
        f"- Finished: {summary['finished_et']} ET (America/New_York)",
        f"- Checkpoint: `{info['ckpt']}` step {info['step']}",
        f"- Questions: {summary['n']} (small-n; treat as a smoke/direction check)",
        f"- Wall: {summary['seconds']:.1f}s",
        "",
        "## Accuracy (gold substring containment)",
        "",
        f"| Metric | Acc |",
        f"|--------|-----|",
        f"| Retrieval (gold in BM25 context) | **{summary['retrieval_acc']:.3f}** |",
    ]
    for m in MODES:
        lines.append(f"| Mode `{m}` | **{summary['by_mode'][m]:.3f}** |")
    if summary["dict_direct_acc"] is not None:
        lines.append(f"| WordNet direct define path (dict Qs only) | **{summary['dict_direct_acc']:.3f}** |")
    lines += ["", "### By kind", "", "| Kind | n | retrieval | gen | copybias | cad | hybrid |", "|------|---|-----------|-----|----------|-----|--------|"]
    for k, v in summary["by_kind"].items():
        lines.append(
            f"| {k} | {v['n']} | {v['retrieval']:.3f} | {v['gen']:.3f} | {v['copybias']:.3f} | {v['cad']:.3f} | {v['hybrid']:.3f} |"
        )
    lines += ["", "## Sample answers (verbatim, truncated)", ""]
    # pick funny / representative samples
    prefer_ids = ["atlantis", "zorblax", "dracula", "def_neuron", "france", "gleeb"]
    shown = set()
    for pid in prefer_ids + [r["id"] for r in rows]:
        if pid in shown:
            continue
        r = next((x for x in rows if x["id"] == pid), None)
        if not r:
            continue
        shown.add(pid)
        if len(shown) > 6:
            break
        lines.append(f"### {r['id']}: {r['question']}")
        lines.append("")
        lines.append(f"- gold: `{r['gold']}`  retrieval_hit={r['retrieval_contains_gold']}")
        for m in MODES:
            mo = r["modes"][m]
            lines.append(f"- **{m}** hit={mo['hit']}: `{mo['generation'][:160].replace(chr(10), ' / ')}`")
        if r.get("dict_direct"):
            lines.append(f"- **dict_direct** hit={r['dict_direct_hit']}:")
            lines.append("```")
            lines.append(r["dict_direct"][:500])
            lines.append("```")
        lines.append("")
    (OUT / "no_train_test.md").write_text("\n".join(lines), encoding="utf-8")
    print("\nSUMMARY", json.dumps(summary, indent=2), flush=True)
    print("wrote", OUT / "no_train_test.md", flush=True)


if __name__ == "__main__":
    main()
