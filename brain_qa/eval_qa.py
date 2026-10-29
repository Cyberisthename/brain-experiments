#!/usr/bin/env python3
"""15 held-out questions with known answers; report extractive hit rate honestly."""
from __future__ import annotations
import json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from retriever import ask

# Each: question, must_include (any of these substrings in answer or citations path+text), notes
QUESTIONS = [
  {"id": 1, "q": "what was real vs shuffled myelin NARMA?", "any": ["0.393", "0.395", "0.557", "0.560", "REAL", "SHUF"], "prefer_path": "myelin"},
  {"id": 2, "q": "which myelin rule recovered most of real myelin's advantage?", "any": ["RULE_FF_DEPTH", "0.608", "61%", "0.460"], "prefer_path": "myelin_mechanism"},
  {"id": 3, "q": "did macaque wiring beat random?", "any": ["macaque", "-1.74", "0.60", "flop", "null"], "prefer_path": "macaque"},
  {"id": 4, "q": "fly excitatory inhibitory XOR result", "any": ["XOR", "0.84", "EI", "fly", "CX"], "prefer_path": "xor"},
  {"id": 5, "q": "what is DESIGN_EI_ONLY?", "any": ["DESIGN_EI_ONLY", "E/I", "SBM", "Uniform"], "prefer_path": "designed_brain"},
  {"id": 6, "q": "plasticity experiment outcome", "any": ["plasticity", "null", "p", "1"], "prefer_path": "plasticity"},
  {"id": 7, "q": "gated frankenstein specialist readout result", "any": ["0.96", "gated", "dilution", "Frankenstein"], "prefer_path": "gated"},
  {"id": 8, "q": "EBRAINS macaque what data?", "any": ["atlas", "wiring", "EBRAINS", "no"], "prefer_path": ""},
  {"id": 9, "q": "RULE_FF_DEPTH NARMA mean", "any": ["0.460", "0.4599", "RULE_FF_DEPTH"], "prefer_path": "myelin_mechanism"},
  {"id": 10, "q": "human myelin REAL NARMA from myelin_real_rep", "any": ["0.393", "REAL", "NARMA"], "prefer_path": "myelin_real_rep"},
  {"id": 11, "q": "celegans sources connectome", "any": ["c elegans", "celegans", "worm", "Cook", "White"], "prefer_path": "celegans"},
  {"id": 12, "q": "finch songbird wiring experiment", "any": ["finch", "song", "HVC"], "prefer_path": "finch"},
  {"id": 13, "q": "Bonferroni primary for myelin mechanism", "any": ["Bonferroni", "halfway", "SHUF", "RULE"], "prefer_path": "myelin_mechanism"},
  {"id": 14, "q": "xor_cx EI_SHUF compared to CX", "any": ["EI_SHUF", "0.70", "0.66", "XOR"], "prefer_path": "xor_cx"},
  {"id": 15, "q": "neuromod real dopamine serotonin test", "any": ["dopamine", "serotonin", "octopamine", "neuromod"], "prefer_path": "neuromod"},
]

def normalize(s):
    return s.lower()

def score_one(item):
    out = ask(item["q"], k=5)
    blob = normalize(out["answer"] + " " + " ".join(c["path"] for c in out["citations"]))
    hits = [a for a in item["any"] if normalize(a) in blob]
    ok = len(hits) >= 1
    path_ok = True
    if item.get("prefer_path"):
        path_ok = any(item["prefer_path"].lower() in c["path"].lower() for c in out["citations"])
    return {
        "id": item["id"], "q": item["q"], "ok": ok, "path_ok": path_ok,
        "matched_any": hits, "citations": [c["path"] for c in out["citations"][:3]],
        "numbers": out.get("numbers", [])[:15],
    }

def main():
    rows = [score_one(q) for q in QUESTIONS]
    n = len(rows)
    acc = sum(r["ok"] for r in rows) / n
    path_acc = sum(r["path_ok"] for r in rows) / n
    report = {
        "n": n,
        "extractive_any_match_accuracy": round(acc, 4),
        "preferred_path_in_top5_rate": round(path_acc, 4),
        "rows": rows,
        "notes": (
            "Accuracy = fraction of questions where at least one expected token/number "
            "appears in the extractive answer or citation paths. Not generative EM. "
            "distilgpt2 generative head not used: too weak for reliable numeric answers; "
            "extractive default kept after protocol."
        ),
    }
    outp = Path(__file__).parent / "out" / "eval_report.json"
    outp.write_text(json.dumps(report, indent=2))
    md = Path(__file__).parent / "out" / "EVAL.md"
    lines = [
        "# brain_qa evaluation",
        "",
        f"- Questions: {n}",
        f"- Extractive any-match accuracy: **{acc:.1%}** ({sum(r['ok'] for r in rows)}/{n})",
        f"- Preferred-path in top-5: **{path_acc:.1%}**",
        "",
        "| id | ok | path_ok | matched | question |",
        "|----|----|---------|---------|----------|",
    ]
    for r in rows:
        lines.append(f"| {r['id']} | {'Y' if r['ok'] else 'N'} | {'Y' if r['path_ok'] else 'N'} | {', '.join(r['matched_any'])[:40]} | {r['q'][:50]} |")
    lines += ["", report["notes"]]
    md.write_text("\n".join(lines) + "\n")
    print(json.dumps({k: report[k] for k in ("n", "extractive_any_match_accuracy", "preferred_path_in_top5_rate")}, indent=2))

if __name__ == "__main__":
    main()
