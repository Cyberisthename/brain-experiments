#!/usr/bin/env python3
"""JARVIS medium_v3 CLI chat with BM25 RAG + hybrid extractive fallback.

Examples:
  nice -n 19 python3 chat.py --once "What is the capital of France?"
  nice -n 19 python3 chat.py --mode hybrid
  nice -n 19 python3 chat.py --upload "The capital of Atlantis is Poseidonis."
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from extractive import best_extractive, format_hybrid  # noqa: E402
from model_io import continuation_only, generate, load_medium  # noqa: E402
from retriever import ChatRetriever, append_upload, seed_novel_facts  # noqa: E402
from answer_modes import answer_one_mode  # noqa: E402
from dictionary_lookup import answer_define  # noqa: E402


def make_prompt(question: str) -> str:
    q = question.strip()
    if q.lower().startswith("question:"):
        return q if "answer:" in q.lower() else q + "\nAnswer:"
    return f"Question: {q}\nAnswer:"


def answer_one(model, tok, retriever, question: str, mode: str, max_new: int, temperature: float, min_ext: float):
    prompt = make_prompt(question)
    hits = retriever.search(question, k=retriever.k)
    aug = retriever.augment(prompt, query=question)
    full = generate(model, tok, aug, max_new=max_new, temperature=temperature)
    cont = continuation_only(full, aug).strip()
    ext = best_extractive(question, hits, min_score=min_ext)

    if mode == "gen":
        out = {
            "mode": "gen",
            "question": question,
            "context_preview": retriever.context(question)[:500],
            "prompt_preview": aug[:500],
            "generation": cont,
            "full": full,
            "extractive": ext,
            "display": cont,
        }
    else:
        display = format_hybrid(ext, cont, mode_label="hybrid")
        out = {
            "mode": "hybrid",
            "question": question,
            "context_preview": retriever.context(question)[:500],
            "prompt_preview": aug[:500],
            "generation": cont,
            "full": full,
            "extractive": ext,
            "display": display,
        }
    return out


def main():
    ap = argparse.ArgumentParser(description="JARVIS medium chat with BM25 RAG")
    ap.add_argument("--ckpt", default="/workspace/jarvis/checkpoints/medium_v3/best.pt")
    ap.add_argument("--data", default="/workspace/jarvis_kaggle/bundle_medium")
    ap.add_argument("--mode", choices=["gen", "hybrid", "cad", "copybias", "cad_hybrid", "dict", "soft_exclusion", "knn"], default="hybrid",
                    help="gen|hybrid|cad|copybias|cad_hybrid|dict (WordNet define path)")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--max-chars", type=int, default=900)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--max-ctx", type=int, default=256, help="cap model ctx for CPU RAM/speed")
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    ap.add_argument("--min-ext-score", type=float, default=1.5)
    ap.add_argument("--once", type=str, default=None, help="single question then exit")
    ap.add_argument("--upload", type=str, default=None, help="append a fact doc to local knowledge store")
    ap.add_argument("--seed-novel", action="store_true", help="seed novel_facts into uploads")
    ap.add_argument("--rebuild-index", action="store_true")
    args = ap.parse_args()

    if args.seed_novel:
        n = seed_novel_facts()
        print(f"seeded {n} novel facts into {HERE / 'knowledge' / 'uploaded_facts.jsonl'}", flush=True)

    if args.upload:
        row = append_upload(args.upload, source="upload")
        print(f"uploaded: {row['text'][:120]}", flush=True)

    from retriever import build_combined_index

    if args.upload or args.seed_novel or args.rebuild_index:
        build_combined_index(force=True)
        print("rebuilt combined BM25 index", flush=True)

    if args.upload and not args.once:
        # upload-only invocation
        if not sys.stdin.isatty() and not args.once:
            return
        # still allow interactive after upload unless --once unset and no tty intent
        pass

    print("Loading medium_v3 (CPU, 2 threads)…", flush=True)
    model, tok, info = load_medium(Path(args.ckpt), Path(args.data), max_ctx=args.max_ctx)
    print(f"loaded step={info['step']} params={info['n_params']:,} ctx={info['max_seq_len']}", flush=True)

    retriever = ChatRetriever(k=args.k, max_chars=args.max_chars)
    print(f"retriever: {retriever.info()}", flush=True)
    print(f"mode={args.mode}  (commands: /quit /mode gen|hybrid|cad|copybias|soft_exclusion|knn|dict /define <word> /upload <text>)", flush=True)

    def handle(q: str):
        q = q.strip()
        if not q:
            return
        if q.lower() in {"/quit", "/exit", "quit", "exit"}:
            raise SystemExit(0)
        if q.lower().startswith("/mode "):
            nonlocal_mode = q.split(None, 1)[1].strip().lower()
            if nonlocal_mode in {"gen", "hybrid", "cad", "copybias", "cad_hybrid", "dict", "soft_exclusion", "knn", "softexcl"}:
                args.mode = nonlocal_mode
                print(f"mode → {args.mode}", flush=True)
            else:
                print("use /mode gen|hybrid|cad|copybias|soft_exclusion|knn|dict", flush=True)
            return
        if q.lower().startswith("/upload "):
            text = q.split(None, 1)[1]
            append_upload(text, source="upload")
            build_combined_index(force=True)
            # reload retriever
            nonlocal_r = ChatRetriever(k=args.k, max_chars=args.max_chars)
            # rebind
            handle.retriever = nonlocal_r  # type: ignore
            print("uploaded + index rebuilt", flush=True)
            return

        if q.lower().startswith("/define "):
            print("\n" + (answer_define("what does " + q.split(None, 1)[1] + " mean?") or "[DICT] miss") + "\n", flush=True)
            return
        r = getattr(handle, "retriever", retriever)
        if args.mode in {"cad", "copybias", "cad_hybrid", "dict", "soft_exclusion", "knn", "softexcl"}:
            out = answer_one_mode(
                model, tok, r, q, mode=args.mode,
                max_new=args.max_new, temperature=args.temperature, min_ext=args.min_ext_score,
            )
        else:
            out = answer_one(
                model, tok, r, q, args.mode, args.max_new, args.temperature, args.min_ext_score
            )
        print("\n" + out["display"] + "\n", flush=True)
        if out.get("extractive") and args.mode == "gen":
            print(f"(extractive hint: {out['extractive']['span']!r} score={out['extractive']['score']:.2f})", flush=True)

    handle.retriever = retriever  # type: ignore

    if args.once:
        handle(args.once)
        return

    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        handle(line)


if __name__ == "__main__":
    main()
