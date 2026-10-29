"""SMOKE ONLY — not a preregistered test. FLY_CX vs ER on XOR_DELAY, 3 seeds."""
from __future__ import annotations

import json
import time
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine2.builders import build_arm
from engine2.tasks import prepare_network, run_xor_delay
from engine2.network import NetworkConfig
from engine2.params import SimParams

SEEDS = [0, 1, 2]
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)


def main():
    print("=" * 60)
    print("SMOKE ONLY — NOT A PREREGISTERED TEST")
    print("FLY_CX vs ER on XOR_DELAY d=5, 3 seeds, engine2 spiking")
    print("=" * 60)
    rows = []
    for arm in ("FLY_CX", "ER"):
        for seed in SEEDS:
            t0 = time.perf_counter()
            b = build_arm(arm, seed=seed)
            net, meta = prepare_network(
                b["W"], b["sign"], nt_labels=b["nt_labels"], delays_ms=b["delays_ms"],
                seed=seed, dt_ms=0.5, calibrate=True,
            )
            # input onto first 200 neurons (or L4-like)
            rng = np.random.default_rng(seed + 99)
            in_idx = rng.choice(net.N, size=min(200, net.N), replace=False)
            res = run_xor_delay(net, np.random.default_rng(seed + 6), in_idx, d=5)
            elapsed = time.perf_counter() - t0
            row = {
                "arm": arm,
                "seed": seed,
                "score": res.get("score"),
                "nrmse": res.get("nrmse"),
                "aborted": res.get("aborted", False),
                "cal_scale": meta["cal"].get("scale"),
                "cal_rate": meta["cal"].get("rate_hz"),
                "elapsed_s": elapsed,
                "smoke_only": True,
                "preregistered": False,
            }
            rows.append(row)
            print(row, flush=True)

    # summary
    def mean_arm(a):
        xs = [r["score"] for r in rows if r["arm"] == a and r["score"] is not None and np.isfinite(r["score"])]
        return float(np.mean(xs)) if xs else float("nan")

    summary = {
        "label": "SMOKE ONLY — not a preregistered test",
        "FLY_CX_mean_score": mean_arm("FLY_CX"),
        "ER_mean_score": mean_arm("ER"),
        "rows": rows,
    }
    (OUT / "smoke_xor.json").write_text(json.dumps(summary, indent=2))
    print("\nSMOKE summary:", json.dumps({k: summary[k] for k in summary if k != "rows"}, indent=2))
    print("Wrote", OUT / "smoke_xor.json")


if __name__ == "__main__":
    main()
