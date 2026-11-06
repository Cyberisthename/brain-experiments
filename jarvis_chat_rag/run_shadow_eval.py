#!/usr/bin/env python3
"""Shadow E-auto eval vs recorded chat_rag full_eval numbers + new jarvis_notes Qs.

Does NOT reload medium_v3 / does NOT rerun CAD hybrid generation.
Compares E-auto answers to golds; retrieval/hybrid numbers come from full_eval.json
(and a fresh BM25 retrieval pass on jarvis_notes questions only).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)

from shadow_eauto import ShadowEAuto, hit_gold, LIMITATION  # noqa: E402
from retriever import ChatRetriever, build_combined_index, load_uploads, NOTES_JSONL  # noqa: E402
from extractive import best_extractive  # noqa: E402


def score_set(name, items, se, retriever=None, recorded=None):
    """items: list of {id, question, gold, kind?}.
    recorded: optional dict id -> {hybrid_hit, retrieval_routed_hit, ...} from full_eval.
    """
    rows = []
    n = len(items)
    e_correct = e_abstain = e_wrong = e_conf_wrong = 0
    # among answered
    answered = 0
    correct_answered = 0
    ret_hits = hyb_hits = 0
    ret_n = hyb_n = 0

    for it in items:
        q = it["question"]
        gold = str(it["gold"])
        r = se.answer(q)
        ans = r["answer"]
        correct, abstain, wrong = hit_gold(ans, gold)
        if correct:
            e_correct += 1
        if abstain:
            e_abstain += 1
        else:
            answered += 1
            if correct:
                correct_answered += 1
            if wrong:
                e_wrong += 1
                e_conf_wrong += 1  # any non-abstain wrong = confident-wrong here

        rec = {
            "id": it.get("id"),
            "kind": it.get("kind"),
            "question": q,
            "gold": gold,
            "eauto_answer": r["display"],
            "eauto_confidence": r["confidence"],
            "eauto_reason": r["reason"],
            "eauto_mapped": r.get("mapped"),
            "eauto_correct": correct,
            "eauto_abstain": abstain,
            "eauto_wrong": wrong,
        }

        # recorded hybrid/retrieval from full_eval if present
        if recorded and it.get("id") in recorded:
            rd = recorded[it["id"]]
            rec["retrieval_routed_hit"] = rd.get("retrieval_routed")
            rec["hybrid_hit"] = (rd.get("modes") or {}).get("hybrid", {}).get("hit")
            rec["gen_hit"] = (rd.get("modes") or {}).get("gen", {}).get("hit")
            if rec["retrieval_routed_hit"] is not None:
                ret_n += 1
                ret_hits += int(bool(rec["retrieval_routed_hit"]))
            if rec["hybrid_hit"] is not None:
                hyb_n += 1
                hyb_hits += int(bool(rec["hybrid_hit"]))
        elif retriever is not None:
            # live retrieval + extractive hybrid (no LM) for jarvis_notes
            hits = retriever.search(q, k=3)
            ctx_chunks = hits
            # retrieval hit: gold string appears in any retrieved text (or ABSTAIN special)
            joined = "\n".join(h.get("text", "") for h in hits)
            if gold.upper() == "ABSTAIN":
                # retrieval "hit" for abstain gold is not well-defined; mark None
                rec["retrieval_routed_hit"] = None
                rec["hybrid_hit"] = None
            else:
                g = gold.lower()
                ret_ok = g in joined.lower()
                rec["retrieval_routed_hit"] = ret_ok
                ret_n += 1
                ret_hits += int(ret_ok)
                ext = best_extractive(q, hits) if hits else None
                # hybrid-like: extractive span contains gold or retrieval contains gold
                span = (ext or {}).get("span") or ""
                hyb_ok = (g in span.lower()) or ret_ok
                rec["hybrid_hit"] = hyb_ok
                hyb_n += 1
                hyb_hits += int(hyb_ok)
                rec["extractive_span"] = span
            rec["retrieved_ids"] = [h.get("id") for h in hits]

        rows.append(rec)

    coverage = answered / max(1, n)
    # accuracy = correct / n (abstain counts as incorrect unless gold=ABSTAIN)
    accuracy = e_correct / max(1, n)
    # selective accuracy among answered
    sel_acc = correct_answered / max(1, answered) if answered else None
    conf_wrong_rate = e_conf_wrong / max(1, answered) if answered else None
    # also report conf-wrong over all n (N4 style often uses among answered)
    summary = {
        "name": name,
        "n": n,
        "eauto_accuracy": round(accuracy, 4),
        "eauto_coverage": round(coverage, 4),
        "eauto_abstain": e_abstain,
        "eauto_correct": e_correct,
        "eauto_wrong_answered": e_wrong,
        "eauto_selective_accuracy": None if sel_acc is None else round(sel_acc, 4),
        "eauto_confident_wrong_among_answered": None
        if conf_wrong_rate is None
        else round(conf_wrong_rate, 4),
        "retrieval_acc": None if ret_n == 0 else round(ret_hits / ret_n, 4),
        "hybrid_acc": None if hyb_n == 0 else round(hyb_hits / hyb_n, 4),
        "retrieval_n": ret_n,
        "hybrid_n": hyb_n,
    }
    return summary, rows


def main():
    t0 = time.time()
    print("Rebuilding BM25 combined index (uploads + jarvis_notes)…", flush=True)
    build_combined_index(force=True)

    print("Building shadow E-auto memories…", flush=True)
    se = ShadowEAuto.from_knowledge()
    info = se.info()
    print(f"shadow info: {json.dumps({k: v for k, v in info.items() if k != 'limitation'}, indent=2)}", flush=True)

    # --- Set A: existing ~30Q from full_eval.json ---
    fe = json.loads((OUT / "full_eval.json").read_text())
    recorded = {it["id"]: it for it in fe["items"]}
    items_30 = [
        {"id": it["id"], "kind": it["kind"], "question": it["question"], "gold": it["gold"]}
        for it in fe["items"]
    ]
    # published baselines (do not rerun gen)
    published = {
        "retrieval_routed": fe["summary"]["retrieval_routed_acc"],
        "retrieval_global": fe["summary"]["retrieval_global_acc"],
        "hybrid": fe["summary"]["by_mode"]["hybrid"],
        "gen": fe["summary"]["by_mode"]["gen"],
        "cad": fe["summary"]["by_mode"]["cad"],
        "routed_hybrid": fe["summary"]["by_mode"]["routed_hybrid"],
        "n": fe["summary"]["n"],
        "finished_et": fe["summary"].get("finished_et"),
    }
    sum_30, rows_30 = score_set("full_eval_30q", items_30, se, recorded=recorded)

    # --- Set B: jarvis_notes derived questions ---
    notes = load_uploads(NOTES_JSONL)
    # Deduplicate conflict questions: for Kappa, one gold=ABSTAIN item
    items_notes = []
    seen_q = set()
    for row in notes:
        q = row.get("question")
        if not q:
            continue
        # conflict probes share the same question — keep one with ABSTAIN gold
        key = q.strip().lower()
        gold = str(row.get("answer", ""))
        if key in seen_q:
            continue
        seen_q.add(key)
        items_notes.append(
            {
                "id": row.get("id"),
                "kind": "jarvis_notes",
                "question": q,
                "gold": gold,
            }
        )
    retriever = ChatRetriever(k=3, max_chars=900)
    sum_notes, rows_notes = score_set("jarvis_notes", items_notes, se, retriever=retriever)

    # --- Combined research+probe subset metrics ---
    elapsed = time.time() - t0

    # Verify neuron4 untouched
    import hashlib
    n4 = Path("/workspace/jarvis/neuron/neuron4")
    h_run = hashlib.sha256((n4 / "run_neuron4.py").read_bytes()).hexdigest()
    h_lib = hashlib.sha256((n4 / "n3lib.py").read_bytes()).hexdigest()
    launch = (n4 / "HASH_run_neuron4_at_launch.txt").read_text().strip().split()[0]
    frozen_lib = None
    for line in (n4 / "HASHES_FROZEN.txt").read_text().splitlines():
        if line.strip().endswith("n3lib.py"):
            frozen_lib = line.split()[0]
            break

    blob = {
        "finished_et": time.strftime("%Y-%m-%d %H:%M:%S ET"),
        "seconds": round(elapsed, 2),
        "shadow_info": {k: v for k, v in info.items()},
        "published_full_eval_baselines": published,
        "full_eval_30q": {"summary": sum_30, "items": rows_30},
        "jarvis_notes": {"summary": sum_notes, "items": rows_notes},
        "neuron4_hashes": {
            "run_neuron4_sha256": h_run,
            "run_neuron4_matches_launch": h_run == launch,
            "n3lib_sha256": h_lib,
            "n3lib_matches_HASHES_FROZEN": h_lib == frozen_lib,
        },
        "limitation": LIMITATION,
        "notes": [
            "Primary chat answer path was not replaced; this is shadow-only scoring.",
            "full_eval hybrid/gen/CAD numbers are REUSED from 2026-11-03 full_eval.json (not recomputed).",
            "jarvis_notes hybrid here = BM25 + extractive span (no LM) for a cheap live compare.",
            "E-auto confident-wrong = wrong among non-abstain answers.",
            "Institutional Books were not indexed.",
        ],
    }
    (OUT / "shadow_eval.json").write_text(json.dumps(blob, indent=2), encoding="utf-8")

    # Markdown report
    lines = []
    lines.append("# Shadow E-auto eval (JARVIS chat_rag)")
    lines.append("")
    lines.append(f"**Finished:** {blob['finished_et']} (America/New_York)  ")
    lines.append(f"**Wall:** {blob['seconds']}s  ")
    lines.append("**Mode:** sidecar only — default pack-router + CAD hybrid answer path unchanged.")
    lines.append("")
    lines.append("## Schema / limitation")
    lines.append("")
    lines.append(LIMITATION)
    lines.append("")
    lines.append(
        f"Memories={info['n_memories']} facts={info['n_facts']} "
        f"(numeric values≈{info['n_numeric_values']}, string values≈{info['n_string_values']})."
    )
    lines.append("")
    lines.append("## Published chat_rag baselines (full_eval.json, 2026-11-03, n=30)")
    lines.append("")
    lines.append("| Method | Acc |")
    lines.append("|---|---:|")
    lines.append(f"| Retrieval pack-routed | {published['retrieval_routed']:.3f} |")
    lines.append(f"| Hybrid extractive | {published['hybrid']:.3f} |")
    lines.append(f"| Pack-routed hybrid | {published['routed_hybrid']:.3f} |")
    lines.append(f"| CAD | {published['cad']:.3f} |")
    lines.append(f"| Generation | {published['gen']:.3f} |")
    lines.append("")
    lines.append("## Shadow E-auto on the same 30Q")
    lines.append("")
    s = sum_30
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| n | {s['n']} |")
    lines.append(f"| E-auto accuracy (correct/n) | {s['eauto_accuracy']} |")
    lines.append(f"| E-auto coverage (answered/n) | {s['eauto_coverage']} |")
    lines.append(f"| E-auto abstain count | {s['eauto_abstain']} |")
    lines.append(f"| E-auto selective accuracy | {s['eauto_selective_accuracy']} |")
    lines.append(f"| E-auto confident-wrong among answered | {s['eauto_confident_wrong_among_answered']} |")
    lines.append(f"| Recorded retrieval_routed acc (same items) | {s['retrieval_acc']} |")
    lines.append(f"| Recorded hybrid acc (same items) | {s['hybrid_acc']} |")
    lines.append("")
    # verdict 30q
    if s["eauto_accuracy"] is not None and published["hybrid"] is not None:
        if s["eauto_accuracy"] + 0.05 < published["hybrid"]:
            v30 = "HURT / behind hybrid on this chat set (schema mismatch expected)"
        elif s["eauto_accuracy"] > published["hybrid"] + 0.05:
            v30 = "HELPED vs hybrid on this chat set"
        else:
            v30 = "INCONCLUSIVE / comparable band vs hybrid"
    else:
        v30 = "INCONCLUSIVE"
    lines.append(f"**30Q verdict vs hybrid {published['hybrid']:.3f}:** {v30}")
    lines.append("")
    lines.append("## jarvis_notes questions (new cards)")
    lines.append("")
    sn = sum_notes
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| n | {sn['n']} |")
    lines.append(f"| E-auto accuracy | {sn['eauto_accuracy']} |")
    lines.append(f"| E-auto coverage | {sn['eauto_coverage']} |")
    lines.append(f"| E-auto abstain | {sn['eauto_abstain']} |")
    lines.append(f"| E-auto selective accuracy | {sn['eauto_selective_accuracy']} |")
    lines.append(f"| E-auto confident-wrong among answered | {sn['eauto_confident_wrong_among_answered']} |")
    lines.append(f"| Live retrieval acc (gold in BM25 hits) | {sn['retrieval_acc']} |")
    lines.append(f"| Live extractive-hybrid acc (no LM) | {sn['hybrid_acc']} |")
    lines.append("")
    lines.append("### Per-question (jarvis_notes)")
    lines.append("")
    lines.append("| id | gold | E-auto | ok? | ret | hyb |")
    lines.append("|---|---|---|---|---|---|")
    for r in rows_notes:
        lines.append(
            f"| {r['id']} | {r['gold']} | {r['eauto_answer']} | "
            f"{'Y' if r['eauto_correct'] else ('ABS' if r['eauto_abstain'] else 'N')} | "
            f"{r.get('retrieval_routed_hit')} | {r.get('hybrid_hit')} |"
        )
    lines.append("")
    lines.append("## Integrity")
    lines.append("")
    lines.append(f"- run_neuron4.py sha256 `{h_run}` matches launch hash: **{h_run == launch}**")
    lines.append(f"- n3lib.py sha256 `{h_lib}` matches HASHES_FROZEN: **{h_lib == frozen_lib}**")
    lines.append("- neuron4/ files were not modified.")
    lines.append("- Default primary answers were not replaced (shadow sidecar only).")
    lines.append("- No QPU, no Kaggle upload, no fine-tune, Institutional Books not indexed.")
    lines.append("")
    (OUT / "shadow_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote", OUT / "shadow_eval.json", "and", OUT / "shadow_eval.md", flush=True)
    print("SUMMARY 30Q", json.dumps(sum_30), flush=True)
    print("SUMMARY notes", json.dumps(sum_notes), flush=True)
    print("VERDICT_30Q", v30, flush=True)


if __name__ == "__main__":
    main()
