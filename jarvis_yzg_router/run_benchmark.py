#!/usr/bin/env python3
"""Benchmark Cognitive Router configs. Prereg: PREREG.md. No LM training."""
from __future__ import annotations

import json
import os
import random
import re
import sys
import time
import resource
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "2")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path("/workspace/jarvis/chat_rag")))
sys.path.insert(0, str(Path("/workspace/jarvis_kaggle")))

from routers import PACKS, PACK_SOURCES, build_router, evidence_vector  # noqa: E402
from retriever import ChatRetriever  # noqa: E402
from extractive import best_extractive  # noqa: E402

DATA = ROOT / "data" / "questions.jsonl"
OUT = ROOT / "results"
SEEDS = [0, 1, 2, 3, 4]
CONFIGS = ["existing", "graph_ppr", "yz", "yzg", "yz_zrand", "softmax"]


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
    }
    for alt in aliases.get(a, []):
        if alt in t:
            return True
    toks = [x for x in re.findall(r"[a-z0-9]+", a) if x not in {"dr", "the", "of", "a", "an"}]
    if len(toks) >= 2 and all(tok in t for tok in toks):
        return True
    return False


def load_questions() -> List[dict]:
    rows = []
    for line in DATA.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def stratified_split(rows: List[dict], seed: int = 0, train_frac: float = 0.7):
    rng = random.Random(seed)
    by = defaultdict(list)
    for r in rows:
        by[r["pack"]].append(r)
    train, test = [], []
    for pack, items in by.items():
        items = list(items)
        rng.shuffle(items)
        n_tr = max(1, int(round(len(items) * train_frac)))
        if len(items) >= 2:
            n_tr = min(n_tr, len(items) - 1)
        train.extend(items[:n_tr])
        test.extend(items[n_tr:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def search_in_pack(mem, query: str, pack: str, k: int = 3) -> List[dict]:
    from model.memory_rag import tokenize

    sources = PACK_SOURCES[pack]
    scores = mem.bm25.get_scores(tokenize(query))
    idx = np.argsort(scores)[::-1]
    out = []
    for i in idx:
        if scores[i] <= 0:
            break
        c = dict(mem.chunks[int(i)])
        if c.get("source") not in sources:
            continue
        c["score"] = float(scores[int(i)])
        out.append(c)
        if len(out) >= k:
            break
    if not out:
        # global fallback
        for i in idx[:k]:
            if scores[i] <= 0:
                break
            c = dict(mem.chunks[int(i)])
            c["score"] = float(scores[int(i)])
            out.append(c)
    return out


def rss_mb() -> float:
    # ru_maxrss on Linux is KB
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def ece_auroc(confs: List[float], correct: List[int], n_bins: int = 10) -> Tuple[float, float]:
    confs = np.asarray(confs, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    if len(confs) == 0:
        return float("nan"), float("nan")
    # ECE
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (confs >= bins[i]) & (confs < bins[i + 1] if i < n_bins - 1 else confs <= bins[i + 1])
        if m.sum() == 0:
            continue
        ece += (m.sum() / len(confs)) * abs(correct[m].mean() - confs[m].mean())
    # AUROC
    try:
        from sklearn.metrics import roc_auc_score
        if len(set(correct.tolist())) < 2:
            auroc = float("nan")
        else:
            auroc = float(roc_auc_score(correct, confs))
    except Exception:
        # Mann-Whitney fallback
        pos = confs[correct == 1]
        neg = confs[correct == 0]
        if len(pos) == 0 or len(neg) == 0:
            auroc = float("nan")
        else:
            # P(score_pos > score_neg)
            wins = 0.0
            for p in pos:
                wins += np.mean(p > neg) + 0.5 * np.mean(p == neg)
            auroc = float(wins / len(pos))
    return float(ece), float(auroc)


def mcnemar_p(correct_a: List[int], correct_b: List[int]) -> float:
    """Exact binomial McNemar p (two-sided) for paired binary outcomes."""
    a = np.asarray(correct_a, dtype=int)
    b = np.asarray(correct_b, dtype=int)
    b01 = int(((a == 0) & (b == 1)).sum())  # B right A wrong
    b10 = int(((a == 1) & (b == 0)).sum())
    n = b01 + b10
    if n == 0:
        return 1.0
    # two-sided exact
    from math import comb
    k = min(b01, b10)
    p = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** (n - 1)) if n < 40 else None
    if p is None:
        # mid-p normal approx
        stat = (abs(b01 - b10) - 1) ** 2 / max(1, n)
        # chi2 df=1 survival rough
        p = float(np.exp(-0.5 * stat))  # crude
    return float(min(1.0, p))


def bootstrap_delta(ca: List[int], cb: List[int], seed: int = 0, n_boot: int = 1000) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    a = np.asarray(ca, dtype=float)
    b = np.asarray(cb, dtype=float)
    n = len(a)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas.append(b[idx].mean() - a[idx].mean())
    deltas = np.sort(deltas)
    lo, hi = deltas[int(0.025 * n_boot)], deltas[int(0.975 * n_boot)]
    return float(b.mean() - a.mean()), float(lo), float(hi)


def evaluate_split(router, mem, rows: List[dict], k: int = 3) -> dict:
    pack_correct = []
    intent_correct = []
    intent_total = 0
    recall_at_k = []
    precision_at_k = []
    hybrid_hits = []
    confs_pack = []
    confs_ret = []
    correct_ret = []
    latencies = []
    details = []

    for r in rows:
        q, gold, pack_gold = r["q"], r["answer"], r["pack"]
        t0 = time.perf_counter()
        dec = router.route(q)
        dt = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt)
        pack_hit = int(dec.pack == pack_gold)
        pack_correct.append(pack_hit)
        confs_pack.append(float(dec.confidence))

        intent_gold = r.get("intent")
        if intent_gold:
            intent_total += 1
            intent_correct.append(int(dec.intent == intent_gold))

        hits = search_in_pack(mem, q, dec.pack, k=k)
        texts = [h.get("text") or "" for h in hits]
        hit_flags = [contains_answer(t, gold) for t in texts]
        rec = float(any(hit_flags))
        prec = float(sum(hit_flags) / max(1, len(hit_flags)))
        recall_at_k.append(rec)
        precision_at_k.append(prec)
        confs_ret.append(float(dec.confidence))
        correct_ret.append(int(rec))

        ext = best_extractive(q, hits, min_score=1.5)
        ext_text = ((ext or {}).get("span") or "") + " " + ((ext or {}).get("sentence") or "")
        hybrid_hits.append(int(contains_answer(ext_text, gold) or contains_answer(" ".join(texts), gold)))

        details.append({
            "id": r["id"],
            "pack_gold": pack_gold,
            "pack_pred": dec.pack,
            "pack_ok": pack_hit,
            "confidence": dec.confidence,
            "recall@3": rec,
            "hybrid": hybrid_hits[-1],
            "intent_gold": intent_gold,
            "intent_pred": dec.intent,
            "latency_ms": dt,
            "reason": dec.reason,
        })

    ece_p, auroc_p = ece_auroc(confs_pack, pack_correct)
    ece_r, auroc_r = ece_auroc(confs_ret, correct_ret)

    # abstain sweep
    abstain = []
    for tau in [0.0, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8]:
        answered = [i for i, c in enumerate(confs_pack) if c >= tau]
        cov = len(answered) / max(1, len(rows))
        acc = float(np.mean([pack_correct[i] for i in answered])) if answered else float("nan")
        abstain.append({"tau": tau, "coverage": cov, "pack_acc_answered": acc})

    return {
        "n": len(rows),
        "pack_acc": float(np.mean(pack_correct)) if pack_correct else float("nan"),
        "intent_acc": float(np.mean(intent_correct)) if intent_correct else float("nan"),
        "intent_n": intent_total,
        "recall@3": float(np.mean(recall_at_k)) if recall_at_k else float("nan"),
        "precision@3": float(np.mean(precision_at_k)) if precision_at_k else float("nan"),
        "hybrid_acc": float(np.mean(hybrid_hits)) if hybrid_hits else float("nan"),
        "latency_ms_mean": float(np.mean(latencies)),
        "latency_ms_p95": float(np.percentile(latencies, 95)),
        "ece_pack": ece_p,
        "auroc_pack": auroc_p,
        "ece_retrieval": ece_r,
        "auroc_retrieval": auroc_r,
        "abstain": abstain,
        "pack_correct": pack_correct,
        "recall_flags": [int(x) for x in recall_at_k],
        "confs": confs_pack,
        "details": details,
    }


def tune_on_train(mem, train: List[dict], seed: int) -> dict:
    """Light grid on train only. Return per-config params."""
    grids = {
        "graph_ppr": [{"damping": d} for d in (0.80, 0.85, 0.90)],
        "yz": [{"lam": lam, "z_mix": zm} for lam in (0.7, 0.85, 0.95) for zm in (0.2, 0.35, 0.5)],
        "yzg": [{"lam": 0.85, "z_mix": 0.35, "g_mix": gm} for gm in (0.1, 0.25, 0.4)],
        "yz_zrand": [{"lam": 0.85, "z_mix": 0.35}],
        "softmax": [{"temperature": t} for t in (0.5, 1.0, 2.0, 5.0)],
        "existing": [{}],
    }
    best = {}
    for name, grid in grids.items():
        scores = []
        for params in grid:
            r = build_router(name, mem, seed=seed, **params)
            # quick pack acc on train (no retrieval for speed on tune)
            ok = 0
            for row in train:
                dec = r.route(row["q"])
                ok += int(dec.pack == row["pack"])
            acc = ok / max(1, len(train))
            scores.append((acc, params))
        scores.sort(key=lambda x: -x[0])
        best[name] = {"params": scores[0][1], "train_pack_acc": scores[0][0]}
        print(f"  tune {name}: {scores[0][1]} train_pack={scores[0][0]:.3f}", flush=True)
    return best


def pick_tau(train_metrics_abstain: list) -> float:
    # maximize pack_acc * coverage^0.5 on train
    best_tau, best_s = 0.0, -1.0
    for row in train_metrics_abstain:
        acc = row["pack_acc_answered"]
        if acc != acc:  # nan
            continue
        s = acc * (row["coverage"] ** 0.5)
        if s > best_s:
            best_s, best_tau = s, row["tau"]
    return best_tau


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_questions()
    print(f"loaded {len(rows)} questions", flush=True)

    print("Loading BM25 index…", flush=True)
    base = ChatRetriever(k=3, max_chars=900)
    mem = base.mem
    print(f"chunks={len(mem.chunks)} rss={rss_mb():.1f}MB", flush=True)

    # Fixed split seed for held-out (prereg: tune on train only)
    train, test = stratified_split(rows, seed=0, train_frac=0.7)
    print(f"split train={len(train)} test={len(test)}", flush=True)
    (OUT / "split.json").write_text(json.dumps({
        "train_ids": [r["id"] for r in train],
        "test_ids": [r["id"] for r in test],
    }, indent=2))

    print("Tuning on train…", flush=True)
    tuned = tune_on_train(mem, train, seed=0)
    (OUT / "tuned_params.json").write_text(json.dumps(tuned, indent=2))

    # Multi-seed eval on held-out (routers with seed_salt)
    all_seed_results = {c: [] for c in CONFIGS}
    rss_before = rss_mb()

    for seed in SEEDS:
        print(f"\n=== seed {seed} ===", flush=True)
        for name in CONFIGS:
            params = dict(tuned[name]["params"])
            router = build_router(name, mem, seed=seed, **params)
            # also get train abstain tau
            train_m = evaluate_split(router, mem, train, k=3)
            tau = pick_tau(train_m["abstain"])
            test_m = evaluate_split(router, mem, test, k=3)
            test_m["tau_from_train"] = tau
            # apply tau summary on test
            answered = [i for i, c in enumerate(test_m["confs"]) if c >= tau]
            test_m["abstain_test"] = {
                "tau": tau,
                "coverage": len(answered) / max(1, len(test)),
                "pack_acc_answered": float(np.mean([test_m["pack_correct"][i] for i in answered])) if answered else float("nan"),
            }
            test_m["params"] = params
            test_m["rss_mb"] = rss_mb()
            all_seed_results[name].append(test_m)
            print(
                f"  {name}: pack={test_m['pack_acc']:.3f} R@3={test_m['recall@3']:.3f} "
                f"hyb={test_m['hybrid_acc']:.3f} lat={test_m['latency_ms_mean']:.2f}ms "
                f"auroc_p={test_m['auroc_pack']:.3f}",
                flush=True,
            )
            # persist per-seed details once for seed 0
            if seed == 0:
                (OUT / f"details_{name}_seed0.json").write_text(
                    json.dumps({k: v for k, v in test_m.items() if k != "details"} | {"details": test_m["details"]}, indent=2)
                )

    # Aggregate
    summary = {}
    for name in CONFIGS:
        packs = [m["pack_acc"] for m in all_seed_results[name]]
        recalls = [m["recall@3"] for m in all_seed_results[name]]
        hybs = [m["hybrid_acc"] for m in all_seed_results[name]]
        lats = [m["latency_ms_mean"] for m in all_seed_results[name]]
        intents = [m["intent_acc"] for m in all_seed_results[name] if m["intent_n"] > 0]
        aurocs = [m["auroc_pack"] for m in all_seed_results[name]]
        eces = [m["ece_pack"] for m in all_seed_results[name]]
        summary[name] = {
            "pack_acc_mean": float(np.mean(packs)),
            "pack_acc_std": float(np.std(packs)),
            "recall@3_mean": float(np.mean(recalls)),
            "recall@3_std": float(np.std(recalls)),
            "hybrid_acc_mean": float(np.mean(hybs)),
            "hybrid_acc_std": float(np.std(hybs)),
            "intent_acc_mean": float(np.mean(intents)) if intents else float("nan"),
            "latency_ms_mean": float(np.mean(lats)),
            "auroc_pack_mean": float(np.nanmean(aurocs)),
            "ece_pack_mean": float(np.nanmean(eces)),
            "rss_mb": float(all_seed_results[name][0]["rss_mb"]),
            "params": tuned[name]["params"],
            "train_pack_acc": tuned[name]["train_pack_acc"],
            "abstain_test_seed0": all_seed_results[name][0]["abstain_test"],
        }

    # Paired tests seed0 vs existing
    base_pack = all_seed_results["existing"][0]["pack_correct"]
    base_rec = all_seed_results["existing"][0]["recall_flags"]
    contrasts = {}
    for name in CONFIGS:
        if name == "existing":
            continue
        d_pack, lo_p, hi_p = bootstrap_delta(base_pack, all_seed_results[name][0]["pack_correct"], seed=0)
        d_rec, lo_r, hi_r = bootstrap_delta(base_rec, all_seed_results[name][0]["recall_flags"], seed=0)
        contrasts[name] = {
            "pack_delta": d_pack,
            "pack_ci95": [lo_p, hi_p],
            "pack_mcnemar_p": mcnemar_p(base_pack, all_seed_results[name][0]["pack_correct"]),
            "recall_delta": d_rec,
            "recall_ci95": [lo_r, hi_r],
            "recall_mcnemar_p": mcnemar_p(base_rec, all_seed_results[name][0]["recall_flags"]),
        }

    # Primary win check (Bonferroni m=4 on C3,C4 vs C1 and Ctrl-Y)
    primary = {}
    for name in ("yz", "yzg"):
        win_pack = (
            summary[name]["pack_acc_mean"] >= summary["existing"]["pack_acc_mean"] + 0.03
            and summary[name]["pack_acc_mean"] >= summary["softmax"]["pack_acc_mean"]
            and contrasts[name]["pack_mcnemar_p"] * 4 < 0.05
        )
        win_rec = (
            summary[name]["recall@3_mean"] >= summary["existing"]["recall@3_mean"] + 0.03
            and contrasts[name]["recall_mcnemar_p"] * 4 < 0.05
        )
        primary[name] = {
            "pack_win": bool(win_pack),
            "recall_win": bool(win_rec),
            "beats_existing": bool(win_pack or win_rec),
        }

    verdict = "DO_NOT_INTEGRATE"
    if primary["yz"]["beats_existing"] or primary["yzg"]["beats_existing"]:
        verdict = "INTEGRATE_CANDIDATE"

    payload = {
        "finished_et": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_total": len(rows),
        "n_train": len(train),
        "n_test": len(test),
        "seeds": SEEDS,
        "summary": summary,
        "contrasts_vs_existing_seed0": contrasts,
        "primary": primary,
        "verdict": verdict,
        "rss_mb_peak": rss_mb(),
        "rss_mb_after_load": rss_before,
        "note": "Y/Z/G are classical. GGraph output is 2 numbers (even/odd mass).",
    }
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2))
    print("\nVERDICT", verdict, flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    main()
