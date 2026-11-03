#!/usr/bin/env python3
"""Compare pack-routed BM25 vs global BM25, and soft-exclusion vs plain/top-k/copy-bias.

Retrieval comparison is cheap (no LM). Decode comparison uses a short subset.
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

from decode_extras import (  # noqa: E402
    generate_copy_bias,
    generate_plain,
    generate_soft_exclusion,
    make_copy_prompt,
)
from model_io import continuation_only, generate, load_medium  # noqa: E402
from pack_router import RoutedRetriever  # noqa: E402
from retriever import ChatRetriever, build_combined_index  # noqa: E402

OUT = HERE / "results"

QUESTIONS = [
    {"id": "france", "q": "What is the capital of France?", "answer": "Paris", "kind": "pack"},
    {"id": "dracula", "q": "Who wrote Dracula?", "answer": "Bram Stoker", "kind": "pack"},
    {"id": "wwi", "q": "What year did World War I begin?", "answer": "1914", "kind": "pack"},
    {"id": "jefferson", "q": "Who wrote The Declaration of Independence of the United States of America?", "answer": "Thomas Jefferson", "kind": "pack"},
    {"id": "zorblax", "q": "What is the capital of Zorblax?", "answer": "Quendara", "kind": "novel"},
    {"id": "voss", "q": "Who discovered aerogel photosynthesis?", "answer": "Dr. Mirella Voss", "kind": "novel"},
    {"id": "wren", "q": "In what year did the MV Starlight Wren launch?", "answer": "2147", "kind": "novel"},
    {"id": "atlantis", "q": "What is the capital of Atlantis?", "answer": "Poseidonis", "kind": "madeup"},
    {"id": "gleeb", "q": "Who invented the Gleeb capacitor?", "answer": "Nara Quill", "kind": "madeup"},
    {"id": "def_neuron", "q": "What does neuron mean?", "answer": "nerve", "kind": "dict"},
    {"id": "def_synapse", "q": "What does synapse mean?", "answer": "junction", "kind": "dict"},
    {"id": "def_photosynthesis", "q": "What does photosynthesis mean?", "answer": "synthesis", "kind": "dict"},
    {"id": "kestrel", "q": "What does the Kestrel Protocol use?", "answer": "three-phase lattice keys", "kind": "novel"},
    {"id": "hamlet", "q": "Who wrote Hamlet?", "answer": "Shakespeare", "kind": "pack"},
    {"id": "okorie", "q": "Who wrote Bright Circadian?", "answer": "Hana Okorie", "kind": "novel"},
]


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def contains_answer(text: str, answer: str) -> bool:
    t, a = norm(text), norm(answer)
    if a in t:
        return True
    aliases = {
        "mary shelley": ["mary wollstonecraft"],
        "shakespeare": ["william shakespeare"],
        "bram stoker": ["stoker"],
        "dr. mirella voss": ["mirella voss"],
        "thomas jefferson": ["jefferson"],
        "three-phase lattice keys": ["lattice keys", "three-phase"],
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
    build_combined_index(force=False)
    base = ChatRetriever(k=3, max_chars=900)
    routed = RoutedRetriever(base, enable_route=True, k=3, max_chars=900)

    # --- retrieval-only compare (all Qs) ---
    ret_rows = []
    for item in QUESTIONS:
        q, gold = item["q"], item["answer"]
        global_ctx = base.context(q)
        routed_ctx = routed.context(q)
        dec = routed.last_route
        ret_rows.append({
            "id": item["id"],
            "kind": item["kind"],
            "question": q,
            "gold": gold,
            "global_hit": contains_answer(global_ctx, gold),
            "routed_hit": contains_answer(routed_ctx, gold),
            "pack": dec.pack if dec else None,
            "reason": dec.reason if dec else None,
            "scores": dec.scores if dec else None,
            "global_preview": global_ctx[:220],
            "routed_preview": routed_ctx[:220],
        })
        print(f"ret {item['id']}: pack={dec.pack if dec else '?'} global={ret_rows[-1]['global_hit']} routed={ret_rows[-1]['routed_hit']}", flush=True)

    n = len(ret_rows)
    ret_summary = {
        "n": n,
        "global_retrieval_acc": sum(r["global_hit"] for r in ret_rows) / n,
        "routed_retrieval_acc": sum(r["routed_hit"] for r in ret_rows) / n,
        "by_kind": {},
    }
    for k in ("pack", "novel", "madeup", "dict"):
        xs = [r for r in ret_rows if r["kind"] == k]
        if not xs:
            continue
        ret_summary["by_kind"][k] = {
            "n": len(xs),
            "global": sum(r["global_hit"] for r in xs) / len(xs),
            "routed": sum(r["routed_hit"] for r in xs) / len(xs),
        }

    # --- decode compare on a subset (uses routed context) ---
    decode_ids = {"zorblax", "atlantis", "france", "dracula", "voss", "def_neuron", "wren", "gleeb"}
    print("Loading model for decode compare…", flush=True)
    model, tok, info = load_medium(max_ctx=192)
    decode_rows = []
    for item in QUESTIONS:
        if item["id"] not in decode_ids:
            continue
        q, gold = item["q"], item["answer"]
        print(f"\n=== decode {item['id']}: {q}", flush=True)
        prompt = f"Question: {q}\nAnswer:"
        # use ROUTED context for all decode arms so the comparison is about decoding
        ctx = routed.context(q)
        aug = f"Context:\n{ctx}\n\n{prompt}" if ctx else prompt
        modes = {}

        # greedy baseline (temp=0)
        full = generate(model, tok, aug, max_new=28, temperature=0.0)
        modes["greedy"] = continuation_only(full, aug).strip()

        # plain sampling (no top-k)
        full = generate_plain(model, tok, aug, max_new=28, temperature=0.8, top_k=None)
        modes["plain_sample"] = continuation_only(full, aug).strip()

        # hard top-k
        full = generate_plain(model, tok, aug, max_new=28, temperature=0.8, top_k=40)
        modes["hard_topk"] = continuation_only(full, aug).strip()

        # copy bias
        copy_prompt = make_copy_prompt(q)
        aug_c = f"Context:\n{ctx}\n\n{copy_prompt}" if ctx else copy_prompt
        full = generate_copy_bias(model, tok, aug_c, ctx, max_new=28, temperature=0.0, bias=3.0)
        modes["copybias"] = continuation_only(full, aug_c).strip()

        # negative-constraint / soft exclusion (user idea)
        full = generate_soft_exclusion(
            model, tok, aug, ctx, max_new=28, temperature=0.8, top_k=40, strength=0.7
        )
        modes["soft_exclusion"] = continuation_only(full, aug).strip()

        row = {
            "id": item["id"],
            "kind": item["kind"],
            "question": q,
            "gold": gold,
            "pack": routed.last_route.pack if routed.last_route else None,
            "modes": {
                m: {"text": modes[m][:240], "hit": contains_answer(modes[m], gold)}
                for m in modes
            },
        }
        decode_rows.append(row)
        for m, v in row["modes"].items():
            print(f"  {m}: hit={v['hit']} {v['text'][:70]!r}", flush=True)

    def dec_acc(name):
        return sum(1 for r in decode_rows if r["modes"][name]["hit"]) / max(1, len(decode_rows))

    summary = {
        "retrieval": ret_summary,
        "decode": {
            "n": len(decode_rows),
            "acc": {m: dec_acc(m) for m in ("greedy", "plain_sample", "hard_topk", "copybias", "soft_exclusion")},
            "note": "Decode arms all use ROUTED context. soft_exclusion = negative-constraint decoding (user idea).",
        },
        "model": info,
        "seconds": time.time() - t0,
        "finished_et": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    report = {"summary": summary, "retrieval_items": ret_rows, "decode_items": decode_rows}
    (OUT / "router_soft_test.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Pack router + soft-exclusion (negative-constraint) test",
        "",
        f"- Finished: {summary['finished_et']} ET (America/New_York)",
        f"- Wall: {summary['seconds']:.1f}s",
        "",
        "## Retrieval: global BM25 vs pack-routed BM25",
        "",
        f"| Metric | Acc (n={ret_summary['n']}) |",
        "|--------|-----|",
        f"| Global BM25 | **{ret_summary['global_retrieval_acc']:.3f}** |",
        f"| Pack-routed BM25 | **{ret_summary['routed_retrieval_acc']:.3f}** |",
        "",
        "| Kind | n | global | routed |",
        "|------|---|--------|--------|",
    ]
    for k, v in ret_summary["by_kind"].items():
        lines.append(f"| {k} | {v['n']} | {v['global']:.3f} | {v['routed']:.3f} |")
    lines += [
        "",
        "## Decode (routed context): plain / top-k / copy-bias / soft-exclusion",
        "",
        f"n={summary['decode']['n']} (subset). soft_exclusion = **negative-constraint decoding (user's idea)**.",
        "",
        "| Mode | Acc |",
        "|------|-----|",
    ]
    for m, a in summary["decode"]["acc"].items():
        label = m + (" (negative-constraint / user idea)" if m == "soft_exclusion" else "")
        lines.append(f"| {label} | **{a:.3f}** |")
    lines += ["", "## Samples", ""]
    for r in decode_rows[:4]:
        lines.append(f"### {r['id']} (pack={r['pack']}): {r['question']}")
        lines.append(f"- gold: `{r['gold']}`")
        for m, v in r["modes"].items():
            lines.append(f"- **{m}** hit={v['hit']}: `{v['text'][:140].replace(chr(10),' / ')}`")
        lines.append("")
    (OUT / "router_soft_test.md").write_text("\n".join(lines), encoding="utf-8")
    print("\nSUMMARY", json.dumps(summary, indent=2), flush=True)
    print("wrote", OUT / "router_soft_test.md", flush=True)


if __name__ == "__main__":
    main()
