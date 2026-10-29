#!/usr/bin/env python3
"""CLI: python ask.py \"what was real vs shuffled myelin NARMA?\""""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from retriever import ask

def main():
    ap = argparse.ArgumentParser(description="Flybrain corpus Q&A (extractive BM25)")
    ap.add_argument("question", nargs="+", help="question text")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    q = " ".join(args.question)
    out = ask(q, k=args.k)
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print(out["answer"])
        print("\n— citations —")
        for c in out["citations"]:
            print(f"  {c['path']}#{c['chunk']}  score={c['score']}")

if __name__ == "__main__":
    main()
