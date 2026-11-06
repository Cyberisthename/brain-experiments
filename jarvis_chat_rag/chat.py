#!/usr/bin/env python3
"""JARVIS medium_v3 CLI chat — default: pack router + CAD hybrid + auto /define.

Default pipeline (no flags needed):
  1. Pack-routed BM25 retrieval (dictionary / books / uploads)
  2. Context-aware decoding (CAD) for generation, with extractive hybrid
  3. Automatic WordNet dictionary for "what does X mean" / "define X"

Old modes stay reachable:
  --mode greedy|gen          greedy LM continuation (no CAD)
  --mode hybrid              extractive + greedy gen (no CAD)
  --mode cad                 CAD generation only
  --retrieval global         classic global BM25 (no pack router)
  --reason / --mode reason   training-free evidence audit; no model load

Examples:
  nice -n 19 python3 chat.py --once "What is the capital of France?"
  nice -n 19 python3 chat.py
  nice -n 19 python3 chat.py --mode greedy --retrieval global
  nice -n 19 python3 chat.py --upload "The capital of Atlantis is Poseidonis."
  nice -n 19 python3 chat.py --shadow --once "What was NEURON-4's confident-wrong rate?"
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

from answer_modes import MODE_ALIASES, CANONICAL_MODES, answer_one_mode, normalize_mode  # noqa: E402
from dictionary_lookup import answer_define  # noqa: E402
from pack_router import RoutedRetriever  # noqa: E402
from retriever import ChatRetriever, append_upload, seed_novel_facts  # noqa: E402

MODE_CHOICES = sorted(CANONICAL_MODES | set(MODE_ALIASES.keys()) | {"default"})


def build_retriever(k: int, max_chars: int, retrieval: str):
    base = ChatRetriever(k=k, max_chars=max_chars)
    if retrieval == "global":
        return base
    return RoutedRetriever(base, enable_route=True, k=k, max_chars=max_chars)


def main():
    ap = argparse.ArgumentParser(
        description="JARVIS medium chat: default = pack router + CAD hybrid + auto WordNet define"
    )
    ap.add_argument("--ckpt", default="/workspace/jarvis/checkpoints/medium_v3/best.pt")
    ap.add_argument("--data", default="/workspace/jarvis_kaggle/bundle_medium")
    ap.add_argument(
        "--mode",
        choices=MODE_CHOICES,
        default="default",
        help=(
            "default=cad_hybrid (CAD+extractive+auto-dict). "
            "Opt-in reason (no LM). Old: greedy|gen, hybrid, cad, copybias, dict, soft_exclusion, knn"
        ),
    )
    ap.add_argument(
        "--retrieval",
        choices=["routed", "global"],
        default="routed",
        help="routed=pack router (default); global=classic BM25 over all packs",
    )
    ap.add_argument("--reason", action="store_true", help="training-free evidence sorting/checking; no model load needed")
    ap.add_argument("--no-dict", action="store_true", help="disable automatic /define path")
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
    ap.add_argument(
        "--shadow",
        action="store_true",
        help="print frozen NEURON-4 E-auto sidecar line after the primary answer (does NOT replace it)",
    )
    args = ap.parse_args()

    try:
        args.mode = "reason" if args.reason else normalize_mode(args.mode)
    except ValueError as e:
        ap.error(str(e))

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

    # Model-free reasoning/dictionary startup; lazily load if REPL changes mode.
    model = tok = None

    def ensure_model():
        nonlocal model, tok
        if model is None:
            print("Loading medium_v3 (CPU, 2 threads)…", flush=True)
            from model_io import load_medium
            model, tok, info = load_medium(Path(args.ckpt), Path(args.data), max_ctx=args.max_ctx)
            print(f"loaded step={info['step']} params={info['n_params']:,} ctx={info['max_seq_len']}", flush=True)

    if args.mode not in {"reason", "dict"}:
        ensure_model()

    retriever = build_retriever(args.k, args.max_chars, args.retrieval)
    print(f"retriever: {retriever.info()}", flush=True)
    shadow = None
    if args.shadow:
        from shadow_eauto import ShadowEAuto
        print("Loading shadow E-auto (sidecar only; primary answer unchanged)…", flush=True)
        shadow = ShadowEAuto.from_knowledge()
        print(f"shadow E-auto: {shadow.info()}", flush=True)
    print(
        f"mode={args.mode} retrieval={args.retrieval} prefer_dict={not args.no_dict}\n"
        f"  (commands: /quit /mode <name> /retrieval routed|global /define <word> /upload <text> /shadow /reason)",
        flush=True,
    )

    state = {"retriever": retriever, "retrieval": args.retrieval, "shadow": shadow if args.shadow else None, "shadow_on": bool(args.shadow)}

    def handle(q: str):
        q = q.strip()
        if not q:
            return
        if q.lower() in {"/quit", "/exit", "quit", "exit"}:
            raise SystemExit(0)
        if q.lower() == "/reason":
            args.mode = "reason"
            print("mode → reason (training-free evidence audit)", flush=True)
            return
        if q.lower().startswith("/mode "):
            name = q.split(None, 1)[1].strip().lower()
            try:
                args.mode = normalize_mode(name)
                print(f"mode → {args.mode}", flush=True)
            except ValueError:
                print(
                    "use /mode default|cad_hybrid|greedy|gen|hybrid|cad|copybias|"
                    "soft_exclusion|knn|dict|reason",
                    flush=True,
                )
            return
        if q.lower().startswith("/retrieval "):
            name = q.split(None, 1)[1].strip().lower()
            if name not in {"routed", "global"}:
                print("use /retrieval routed|global", flush=True)
                return
            state["retrieval"] = name
            state["retriever"] = build_retriever(args.k, args.max_chars, name)
            print(f"retrieval → {name}  info={state['retriever'].info()}", flush=True)
            return
        if q.lower().startswith("/upload "):
            text = q.split(None, 1)[1]
            append_upload(text, source="upload")
            build_combined_index(force=True)
            state["retriever"] = build_retriever(args.k, args.max_chars, state["retrieval"])
            print("uploaded + index rebuilt", flush=True)
            return

        if q.lower().startswith("/define "):
            print(
                "\n"
                + (answer_define("what does " + q.split(None, 1)[1] + " mean?") or "[DICT] miss")
                + "\n",
                flush=True,
            )
            return
        if q.lower().strip() in {"/shadow", "/shadow on", "/shadow off"}:
            cmd = q.lower().strip()
            if cmd == "/shadow":
                state["shadow_on"] = not state.get("shadow_on")
            elif cmd.endswith("on"):
                state["shadow_on"] = True
            else:
                state["shadow_on"] = False
            if state["shadow_on"] and state.get("shadow") is None:
                from shadow_eauto import ShadowEAuto
                print("Loading shadow E-auto (sidecar only)…", flush=True)
                state["shadow"] = ShadowEAuto.from_knowledge()
                print(f"shadow E-auto: {state['shadow'].info()}", flush=True)
            print(f"shadow sidecar → {'ON' if state['shadow_on'] else 'OFF'} (primary answer unchanged)", flush=True)
            return

        if args.mode not in {"reason", "dict"}:
            ensure_model()
        out = answer_one_mode(
            model,
            tok,
            state["retriever"],
            q,
            mode=args.mode,
            max_new=args.max_new,
            temperature=args.temperature,
            min_ext=args.min_ext_score,
            prefer_dict=not args.no_dict,
        )
        print("\n" + out["display"] + "\n", flush=True)
        # Shadow sidecar: NEVER replaces primary display above
        if state.get("shadow_on") and state.get("shadow") is not None:
            try:
                print(state["shadow"].answer_sidecar(q), flush=True)
            except Exception as e:
                print(f"[shadow E-auto] error: {e}", flush=True)
        if out.get("extractive") and args.mode == "gen" and out["mode"] != "dict":
            ext = out["extractive"]
            if ext:
                print(
                    f"(extractive hint: {ext.get('span')!r} score={ext.get('score', 0):.2f})",
                    flush=True,
                )
        route = getattr(state["retriever"], "last_route", None)
        if route is not None:
            print(f"(route: pack={route.pack} reason={route.reason})", flush=True)

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
