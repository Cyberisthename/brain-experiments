#!/usr/bin/env python3
"""Full ~30Q training-free eval: CAD, kNN-LM, soup, pack router, soft-exclusion,
dictionary/define — each vs gen + hybrid baselines. Real scores only.
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
from dictionary_lookup import (  # noqa: E402
    answer_define,
    build_definition_pack,
    seed_dictionary_into_uploads,
)
from extractive import best_extractive, format_hybrid  # noqa: E402
from model_io import load_medium  # noqa: E402
from model_soup import discover_medium_ckpts, save_soup  # noqa: E402
from pack_router import RoutedRetriever  # noqa: E402
from retriever import (  # noqa: E402
    ChatRetriever,
    append_upload,
    build_combined_index,
    seed_novel_facts,
)

OUT = HERE / "results"

QUESTIONS = [
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
    {"id": "atlantis", "q": "What is the capital of Atlantis?", "answer": "Poseidonis", "kind": "madeup"},
    {"id": "gleeb", "q": "Who invented the Gleeb capacitor?", "answer": "Nara Quill", "kind": "madeup"},
    {"id": "def_neuron", "q": "What does neuron mean?", "answer": "nerve", "kind": "dict"},
    {"id": "def_synapse", "q": "What does synapse mean?", "answer": "junction", "kind": "dict"},
    {"id": "def_myelin", "q": "Define myelin", "answer": "fatty", "kind": "dict"},
    {"id": "def_photosynthesis", "q": "What does photosynthesis mean?", "answer": "synthesis", "kind": "dict"},
    {"id": "def_algorithm", "q": "What does algorithm mean?", "answer": "rule", "kind": "dict"},
    {"id": "def_entropy", "q": "What does entropy mean?", "answer": "uncertainty", "kind": "dict"},
]

# LM decode modes (all use global BM25 context unless noted)
LM_MODES = ["gen", "cad", "soft_exclusion", "knn"]


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
        "blue-green": ["blue green", "bluish green"],
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


def acc(rows, key_fn):
    return sum(1 for r in rows if key_fn(r)) / max(1, len(rows))


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
    n_dict = seed_dictionary_into_uploads(force=False, limit=400)
    print(f"seeded {n_dict} new wordnet cards", flush=True)
    build_combined_index(force=True)

    # --- model soup (honest: only compatible medium ckpts on disk) ---
    ckpts = discover_medium_ckpts()
    soup_meta = save_soup(OUT / "medium_soup.pt", paths=ckpts or None)
    print("SOUP", json.dumps(soup_meta), flush=True)

    print("Loading model…", flush=True)
    model, tok, info = load_medium(max_ctx=192)
    print(info, flush=True)

    # If soup has N>1, also load souped weights for a separate gen pass.
    # N=1 → soup_gen ≡ gen (skip second load).
    soup_model = None
    if not soup_meta.get("identical_to_single", True):
        from model_soup import load_soup_medium
        soup_model, _, soup_info = load_soup_medium(OUT / "medium_soup.pt", max_ctx=192, rebuild=False)
        print("loaded multi-ckpt soup", soup_info.get("soup"), flush=True)

    base = ChatRetriever(k=3, max_chars=900)
    routed = RoutedRetriever(base, enable_route=True, k=3, max_chars=900)

    rows = []
    for item in QUESTIONS:
        q, gold = item["q"], item["answer"]
        print(f"\n=== {item['id']}: {q}", flush=True)

        global_ctx = base.context(q)
        routed_ctx = routed.context(q)
        route = routed.last_route
        ret_global = contains_answer(global_ctx, gold)
        ret_routed = contains_answer(routed_ctx, gold)

        mode_out = {}
        for mode in LM_MODES:
            out = answer_one_mode(
                model, tok, base, q, mode=mode,
                max_new=24, temperature=0.0, min_ext=1.5,
                prefer_dict=False, extractive_only=False,
            )
            gen_text = out.get("generation") or ""
            hit = contains_answer(gen_text, gold)
            mode_out[mode] = {"hit": hit, "generation": gen_text[:240]}
            print(f"  {mode}: hit={hit} gen={gen_text[:70]!r}", flush=True)

        # hybrid extractive on GLOBAL retrieval
        hits_g = base.search(q, k=3)
        ext_g = best_extractive(q, hits_g, min_score=1.5)
        ext_g_text = ((ext_g or {}).get("span") or "") + " " + ((ext_g or {}).get("sentence") or "")
        hybrid_hit = contains_answer(ext_g_text, gold) or contains_answer(
            format_hybrid(ext_g, mode_out["gen"]["generation"], mode_label="hybrid"), gold
        )
        mode_out["hybrid"] = {
            "hit": hybrid_hit,
            "generation": "",
            "extractive_span": ((ext_g or {}).get("span") or "")[:120],
        }
        print(f"  hybrid: hit={hybrid_hit} span={mode_out['hybrid']['extractive_span']!r}", flush=True)

        # hybrid extractive on ROUTED retrieval (pack router)
        hits_r = routed.search(q, k=3)
        ext_r = best_extractive(q, hits_r, min_score=1.5)
        ext_r_text = ((ext_r or {}).get("span") or "") + " " + ((ext_r or {}).get("sentence") or "")
        routed_hybrid_hit = contains_answer(ext_r_text, gold)
        mode_out["routed_hybrid"] = {
            "hit": routed_hybrid_hit,
            "extractive_span": ((ext_r or {}).get("span") or "")[:120],
            "pack": route.pack if route else None,
            "reason": route.reason if route else None,
        }
        print(
            f"  routed_hybrid: hit={routed_hybrid_hit} pack={mode_out['routed_hybrid']['pack']} "
            f"span={mode_out['routed_hybrid']['extractive_span']!r}",
            flush=True,
        )

        # dictionary /define direct path (all Qs; meaningful on dict kind)
        dict_direct = answer_define(q)
        dict_hit = contains_answer(dict_direct or "", gold) if dict_direct else False
        mode_out["dict_define"] = {
            "hit": dict_hit if item["kind"] == "dict" else False,
            "answered": bool(dict_direct),
            "text": (dict_direct or "")[:300],
            "scored": item["kind"] == "dict",
        }
        if item["kind"] == "dict":
            print(f"  dict_define: hit={dict_hit}", flush=True)

        # soup gen
        if soup_model is not None:
            out_s = answer_one_mode(
                soup_model, tok, base, q, mode="gen",
                max_new=24, temperature=0.0, prefer_dict=False,
            )
            s_text = out_s.get("generation") or ""
            mode_out["soup_gen"] = {"hit": contains_answer(s_text, gold), "generation": s_text[:240]}
            print(f"  soup_gen: hit={mode_out['soup_gen']['hit']}", flush=True)
        else:
            # N=1: identical to gen
            mode_out["soup_gen"] = {
                "hit": mode_out["gen"]["hit"],
                "generation": mode_out["gen"]["generation"],
                "note": "N=1 soup ≡ gen (only one medium-compatible ckpt)",
            }

        rows.append({
            "id": item["id"],
            "kind": item["kind"],
            "question": q,
            "gold": gold,
            "retrieval_global": ret_global,
            "retrieval_routed": ret_routed,
            "route_pack": route.pack if route else None,
            "route_reason": route.reason if route else None,
            "modes": mode_out,
            "context_preview_global": global_ctx[:240],
            "context_preview_routed": routed_ctx[:240],
        })

        # incremental save
        (OUT / "full_eval_partial.json").write_text(
            json.dumps({"n_done": len(rows), "items": rows}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def mode_acc(mode, kind=None):
        xs = [r for r in rows if kind is None or r["kind"] == kind]
        return sum(1 for r in xs if r["modes"][mode]["hit"]) / max(1, len(xs))

    dict_rows = [r for r in rows if r["kind"] == "dict"]
    summary = {
        "n": len(rows),
        "retrieval_global_acc": acc(rows, lambda r: r["retrieval_global"]),
        "retrieval_routed_acc": acc(rows, lambda r: r["retrieval_routed"]),
        "by_mode": {
            "gen": mode_acc("gen"),
            "hybrid": mode_acc("hybrid"),
            "cad": mode_acc("cad"),
            "soft_exclusion": mode_acc("soft_exclusion"),
            "knn": mode_acc("knn"),
            "soup_gen": mode_acc("soup_gen"),
            "routed_hybrid": mode_acc("routed_hybrid"),
            "dict_define_on_dict_qs": (
                sum(1 for r in dict_rows if r["modes"]["dict_define"]["hit"]) / max(1, len(dict_rows))
            ),
        },
        "by_kind": {
            k: {
                "n": sum(1 for r in rows if r["kind"] == k),
                "retrieval_global": acc([r for r in rows if r["kind"] == k], lambda r: r["retrieval_global"]),
                "retrieval_routed": acc([r for r in rows if r["kind"] == k], lambda r: r["retrieval_routed"]),
                **{m: mode_acc(m, k) for m in ("gen", "hybrid", "cad", "soft_exclusion", "knn", "soup_gen", "routed_hybrid")},
            }
            for k in ("pack", "novel", "madeup", "dict")
        },
        "soup": soup_meta,
        "model": info,
        "seconds": time.time() - t0,
        "finished_et": time.strftime("%Y-%m-%d %H:%M:%S"),
        "notes": (
            "Small-n (~30). Gold = substring containment. "
            "CAD α=1.0; soft_exclusion strength=0.7; kNN-LM λ=0.5 k=16 passage-scoped; "
            "soup = uniform avg of compatible medium ckpts (N=1 ⇒ ≡ gen)."
        ),
    }
    report = {"summary": summary, "items": rows}
    (OUT / "full_eval.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# JARVIS full training-free eval (~30Q)",
        "",
        f"- Finished: {summary['finished_et']} ET (America/New_York)",
        f"- Checkpoint: `{info['ckpt']}` step {info['step']}",
        f"- Soup: {soup_meta.get('note')} (n={soup_meta.get('n')})",
        f"- Questions: {summary['n']} (small-n direction check)",
        f"- Wall: {summary['seconds']:.1f}s",
        "",
        "## Accuracy (gold substring containment)",
        "",
        "| Method | Acc |",
        "|--------|-----|",
        f"| Retrieval global BM25 | **{summary['retrieval_global_acc']:.3f}** |",
        f"| Retrieval **pack-routed** BM25 | **{summary['retrieval_routed_acc']:.3f}** |",
        f"| **gen** (baseline) | **{summary['by_mode']['gen']:.3f}** |",
        f"| **hybrid** extractive (baseline) | **{summary['by_mode']['hybrid']:.3f}** |",
        f"| CAD | **{summary['by_mode']['cad']:.3f}** |",
        f"| soft-exclusion (neg-constraint) | **{summary['by_mode']['soft_exclusion']:.3f}** |",
        f"| kNN-LM (passage-scoped) | **{summary['by_mode']['knn']:.3f}** |",
        f"| model soup gen | **{summary['by_mode']['soup_gen']:.3f}** |",
        f"| pack-routed hybrid | **{summary['by_mode']['routed_hybrid']:.3f}** |",
        f"| dictionary /define (dict Qs only, n={len(dict_rows)}) | **{summary['by_mode']['dict_define_on_dict_qs']:.3f}** |",
        "",
        "### By kind",
        "",
        "| Kind | n | ret_g | ret_r | gen | hybrid | cad | soft | knn | soup | routed_hyb |",
        "|------|---|-------|-------|-----|--------|-----|------|-----|------|------------|",
    ]
    for k, v in summary["by_kind"].items():
        lines.append(
            f"| {k} | {v['n']} | {v['retrieval_global']:.3f} | {v['retrieval_routed']:.3f} | "
            f"{v['gen']:.3f} | {v['hybrid']:.3f} | {v['cad']:.3f} | {v['soft_exclusion']:.3f} | "
            f"{v['knn']:.3f} | {v['soup_gen']:.3f} | {v['routed_hybrid']:.3f} |"
        )
    lines += ["", "## Sample answers", ""]
    prefer = ["atlantis", "zorblax", "voss", "def_neuron", "france", "gleeb"]
    shown = set()
    for pid in prefer + [r["id"] for r in rows]:
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
        lines.append(
            f"- gold: `{r['gold']}`  ret_global={r['retrieval_global']}  "
            f"ret_routed={r['retrieval_routed']}  pack={r['route_pack']}"
        )
        for m in ("gen", "cad", "soft_exclusion", "knn", "hybrid", "routed_hybrid", "soup_gen"):
            mo = r["modes"][m]
            snippet = (mo.get("generation") or mo.get("extractive_span") or "")[:160].replace("\n", " / ")
            lines.append(f"- **{m}** hit={mo['hit']}: `{snippet}`")
        if r["kind"] == "dict":
            d = r["modes"]["dict_define"]
            lines.append(f"- **dict_define** hit={d['hit']}:")
            lines.append("```")
            lines.append(d.get("text") or "")
            lines.append("```")
        lines.append("")
    (OUT / "full_eval.md").write_text("\n".join(lines), encoding="utf-8")
    print("\nSUMMARY", json.dumps(summary, indent=2), flush=True)
    print("wrote", OUT / "full_eval.md", flush=True)


if __name__ == "__main__":
    main()
