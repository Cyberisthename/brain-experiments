#!/usr/bin/env python3
"""Re-check the DEFAULT chat pipeline scores (~30Q).

Default chat = pack-routed retrieval + CAD hybrid + auto /define.

Also re-measures the published baselines on GLOBAL retrieval so CAD≈0.433
and gen≈0.300 remain comparable to results/full_eval.md.

Usage:
  cd /workspace/jarvis/chat_rag
  OMP_NUM_THREADS=2 nice -n 19 python3 run_default_check.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from answer_modes import answer_one_mode  # noqa: E402
from dictionary_lookup import answer_define, build_definition_pack, seed_dictionary_into_uploads  # noqa: E402
from extractive import best_extractive  # noqa: E402
from model_io import load_medium  # noqa: E402
from pack_router import RoutedRetriever  # noqa: E402
from retriever import ChatRetriever, append_upload, build_combined_index, seed_novel_facts  # noqa: E402
from run_full_eval import QUESTIONS, contains_answer  # noqa: E402

OUT = HERE / "results"


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
    seed_dictionary_into_uploads(force=False, limit=400)
    build_combined_index(force=False)

    print("Loading medium_v3…", flush=True)
    model, tok, info = load_medium(max_ctx=192)
    print(info, flush=True)

    base = ChatRetriever(k=3, max_chars=900)
    routed = RoutedRetriever(base, enable_route=True, k=3, max_chars=900)

    rows = []
    for item in QUESTIONS:
        q, gold = item["q"], item["answer"]
        print(f"\n=== {item['id']}: {q}", flush=True)

        global_ctx = base.context(q)
        routed_ctx = routed.context(q)
        route = routed.last_route
        ret_g = contains_answer(global_ctx, gold)
        ret_r = contains_answer(routed_ctx, gold)

        # Published baselines: gen + CAD on GLOBAL BM25 (matches full_eval.md)
        out_gen = answer_one_mode(
            model, tok, base, q, mode="gen",
            max_new=24, temperature=0.0, prefer_dict=False,
        )
        gen_text = out_gen.get("generation") or ""
        gen_hit = contains_answer(gen_text, gold)

        out_cad = answer_one_mode(
            model, tok, base, q, mode="cad",
            max_new=24, temperature=0.0, prefer_dict=False,
        )
        cad_text = out_cad.get("generation") or ""
        cad_hit = contains_answer(cad_text, gold)

        # routed hybrid extractive
        hits_r = routed.search(q, k=3)
        ext_r = best_extractive(q, hits_r, min_score=1.5)
        ext_text = ((ext_r or {}).get("span") or "") + " " + ((ext_r or {}).get("sentence") or "")
        routed_hyb_hit = contains_answer(ext_text, gold)

        # dictionary direct
        dict_direct = answer_define(q)
        dict_hit = contains_answer(dict_direct or "", gold) if dict_direct else False

        # DEFAULT pipeline: routed + cad_hybrid + prefer_dict (what chat.py does now)
        out_def = answer_one_mode(
            model, tok, routed, q, mode="default",
            max_new=24, temperature=0.0, prefer_dict=True,
        )
        def_display = out_def.get("display") or ""
        def_hit = contains_answer(def_display, gold)
        if out_def.get("mode") == "dict" and item["kind"] == "dict":
            def_hit = bool(dict_hit)

        row = {
            "id": item["id"],
            "kind": item["kind"],
            "question": q,
            "gold": gold,
            "retrieval_global": ret_g,
            "retrieval_routed": ret_r,
            "route_pack": route.pack if route else None,
            "gen_hit": gen_hit,
            "cad_hit": cad_hit,
            "routed_hybrid_hit": routed_hyb_hit,
            "dict_hit": dict_hit if item["kind"] == "dict" else None,
            "dict_answered": bool(dict_direct),
            "default_hit": def_hit,
            "default_mode": out_def.get("mode"),
            "gen_preview": gen_text[:160],
            "cad_preview": cad_text[:160],
            "default_preview": def_display[:240],
        }
        rows.append(row)
        print(
            f"  ret_r={ret_r} gen={gen_hit} cad={cad_hit} hyb={routed_hyb_hit} "
            f"def={def_hit} mode={out_def.get('mode')} pack={row['route_pack']}",
            flush=True,
        )
        (OUT / "default_check_partial.json").write_text(
            json.dumps({"n_done": len(rows), "items": rows}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    n = len(rows)
    dict_rows = [r for r in rows if r["kind"] == "dict"]
    summary = {
        "n": n,
        "retrieval_global_acc": sum(r["retrieval_global"] for r in rows) / n,
        "retrieval_routed_acc": sum(r["retrieval_routed"] for r in rows) / n,
        "gen_acc": sum(r["gen_hit"] for r in rows) / n,
        "cad_acc": sum(r["cad_hit"] for r in rows) / n,
        "routed_hybrid_acc": sum(r["routed_hybrid_hit"] for r in rows) / n,
        "dict_define_acc": (
            sum(1 for r in dict_rows if r["dict_hit"]) / max(1, len(dict_rows))
        ),
        "dict_define_n": f"{sum(1 for r in dict_rows if r['dict_hit'])}/{len(dict_rows)}",
        "default_pipeline_acc": sum(r["default_hit"] for r in rows) / n,
        "expected": {
            "retrieval_routed": 0.967,
            "cad": 0.433,
            "gen": 0.300,
            "routed_hybrid": 0.900,
            "dict_define": 1.000,
        },
        "model": info,
        "seconds": time.time() - t0,
        "finished_et": time.strftime("%Y-%m-%d %H:%M:%S"),
        "defaults": {
            "mode": "default → cad_hybrid",
            "retrieval": "routed",
            "prefer_dict": True,
        },
    }

    # ±1/30 tolerance for index-order edge cases; CAD/gen allow ±0.05 absolute
    tol_ret = 1.0 / 30 + 0.02
    checks = {
        "retrieval_routed≈0.967": abs(summary["retrieval_routed_acc"] - 0.967) <= tol_ret,
        "cad≈0.433": abs(summary["cad_acc"] - 0.433) <= 0.05 + 1e-9,
        "gen≈0.300": abs(summary["gen_acc"] - 0.300) <= 0.05 + 1e-9,
        "routed_hybrid≈0.900": abs(summary["routed_hybrid_acc"] - 0.900) <= tol_ret,
        "dict_define=6/6": summary["dict_define_acc"] >= 1.0 - 1e-9,
    }
    summary["checks"] = checks
    summary["all_pass"] = all(checks.values())

    report = {"summary": summary, "items": rows}
    (OUT / "default_check.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# JARVIS default pipeline re-check (~30Q)",
        "",
        f"- Finished: {summary['finished_et']} ET (America/New_York)",
        f"- Defaults: mode=default(cad_hybrid), retrieval=routed, prefer_dict=True",
        f"- Checkpoint: `{info.get('ckpt')}` step {info.get('step')}",
        f"- Wall: {summary['seconds']:.1f}s",
        "",
        "## Scores",
        "",
        "| Metric | Got | Expected |",
        "|--------|-----|----------|",
        f"| Retrieval pack-routed | **{summary['retrieval_routed_acc']:.3f}** | 0.967 |",
        f"| Retrieval global | {summary['retrieval_global_acc']:.3f} | 0.933 |",
        f"| gen (greedy, global ctx) | **{summary['gen_acc']:.3f}** | 0.300 |",
        f"| CAD (global ctx) | **{summary['cad_acc']:.3f}** | 0.433 |",
        f"| pack-routed hybrid | **{summary['routed_hybrid_acc']:.3f}** | 0.900 |",
        f"| dictionary /define (dict Qs) | **{summary['dict_define_acc']:.3f}** ({summary['dict_define_n']}) | 1.000 (6/6) |",
        f"| **default pipeline** (routed+CAD hybrid+auto-dict) | **{summary['default_pipeline_acc']:.3f}** | ≥ hybrid / CAD |",
        "",
        "## Checks",
        "",
    ]
    for k, v in checks.items():
        lines.append(f"- {'PASS' if v else 'FAIL'}: {k}")
    lines.append("")
    lines.append(f"**all_pass = {summary['all_pass']}**")
    lines.append("")
    lines.append("## How to run chat with these defaults")
    lines.append("")
    lines.append("```bash")
    lines.append("cd /workspace/jarvis/chat_rag")
    lines.append("OMP_NUM_THREADS=2 nice -n 19 python3 chat.py --once \"What does neuron mean?\"")
    lines.append("# old modes: --mode greedy --retrieval global")
    lines.append("```")
    (OUT / "default_check.md").write_text("\n".join(lines), encoding="utf-8")
    print("\nSUMMARY", json.dumps(summary, indent=2), flush=True)
    print("wrote", OUT / "default_check.md", flush=True)
    return 0 if summary["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
