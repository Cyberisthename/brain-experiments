# Myelin REAL vs SHUF replication results

Prereg: `PREREGISTRATION_myelin_real_rep.md`.

## Primary (REAL vs SHUF NARMA, leak=0.2, one-sided)

- REAL mean NARMA = **0.3934** (n=24)
- SHUF mean NARMA = **0.5575** (n=24, 3 draws/seed averaged)
- Δ (REAL−SHUF) = **-0.1640**
- p_one_sided (REAL < SHUF) = **1.967e-09**
- **claim_ok = True**

## Means ± sd at leak=0.2

| task | NODELAY | REAL | SHUF | LAYER_SHUF | EI_SHUF |
|---|---:|---:|---:|---:|---:|
| MC | 6.4850±0.9474 | 18.9924±1.0933 | 18.0801±1.1855 | 18.7447±1.1042 | 18.1095±1.0619 |
| NARMA | 0.5916±0.0377 | 0.3934±0.0363 | 0.5575±0.0619 | 0.4408±0.0342 | 0.5560±0.0482 |
| PC_HARD | 0.8421±0.0469 | 0.8847±0.0398 | 0.8872±0.0367 | 0.8888±0.0350 | 0.8858±0.0374 |

## Secondary NARMA vs other controls (two-sided raw p)

- vs LAYER_SHUF: Δ=-0.0473, p_raw=3.561e-05
- vs EI_SHUF: Δ=-0.1625, p_raw=3.935e-09
- vs NODELAY: Δ=-0.1981, p_raw=3.064e-09

## Leak robustness (REAL vs SHUF NARMA, one-sided)

- leak=0.1: REAL=0.4978 SHUF=0.6028 Δ=-0.1050 p=3.483e-08
- leak=0.2: REAL=0.3934 SHUF=0.5575 Δ=-0.1640 p=1.967e-09
- leak=0.3: REAL=0.3103 SHUF=0.5014 Δ=-0.1910 p=1.532e-09

Figures: `myelin_real_rep_narma.png`, `myelin_real_rep_all_tasks.png` under `/workspace/flybrain/figs/`.

## Honest verdict

**Verdict: REPLICATED.** Real myelin placement beats global shuffle on NARMA with 24 new seeds under the preregistered one-sided test.


## Interpretation notes

- **LAYER_SHUF** (preserve per-layer myelin histogram) lands between REAL and global SHUF on NARMA (0.441 vs REAL 0.393 vs SHUF 0.557). REAL still beats LAYER_SHUF (raw p≈3.6e-5), so fine-scale placement *within* layers carries extra signal beyond layer averages.
- **EI_SHUF** ≈ global SHUF (0.556 vs 0.557). Preserving only E vs I myelin histograms does **not** recover the REAL advantage — class-level myelin mass is not enough.
- MC rises a lot with any delays vs NODELAY; REAL vs SHUF on MC is small (same pattern as the parent experiment). PC_HARD is near-tied across delay arms.
- Locked ridge α was mostly `1e-08` (see `out/alpha_choice.json`); parent used fixed `1e-6`. Replication is stronger under per-arm tuned α, not weaker.
- Proof pushed to GitHub (RESULTS + small json; no runs.jsonl/logs).

## Locked alphas
```json
{
  "NODELAY|0.1": 1e-06,
  "NODELAY|0.2": 1e-08,
  "NODELAY|0.3": 1e-08,
  "REAL_MYELIN|0.1": 1e-08,
  "REAL_MYELIN|0.2": 1e-08,
  "REAL_MYELIN|0.3": 1e-08,
  "SHUF_MYELIN|0.1": 1e-08,
  "SHUF_MYELIN|0.2": 1e-08,
  "SHUF_MYELIN|0.3": 1e-08,
  "LAYER_SHUF|0.1": 1e-08,
  "LAYER_SHUF|0.2": 1e-08,
  "LAYER_SHUF|0.3": 1e-08,
  "EI_SHUF|0.1": 1e-08,
  "EI_SHUF|0.2": 1e-08,
  "EI_SHUF|0.3": 1e-08
}
```
