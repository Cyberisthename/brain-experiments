# Preregistration addendum: EMPN v2 organ-fix (v2b)
Locked before the corrected full-arm v2b rerun. Diagnostics: `out/diag_v2b_cfs_tr.json`.

## Plain root causes (v2 final + seed 900/901)
1. **CFS overrode food-seeking.** Old `cfs()` was risk-only once goals were gated; bias was added 1:1 onto Q; unseen WM self-loops inflated V; food location alternates every 10 eps so stale goals locked the agent onto the previous food. EMPN success **0%** vs EMPN-noCFS **50%**.
2. **+TR replaced TD and exploded Wq.** `use_tr` skipped online TD and applied unnormalized episodic `α·e·RPE` → `‖Wq‖` ~10–14 (healthy ~0.7). FLY **46% → FLY+TR 2.5%**.

## Shared fixes (every arm)
- **CFS → `cfs_er`:** `w_r·E[rew] − w_risk·risk·loom` (w_r=1); fall back (bias=0) if `seen[p]<2` or no reward signal; bootstrap only through **seen** transitions; per-episode decay of rew/risk (non-stationary food); **assist-only** — never flip `argmax(q+bonus)` (RNG-aligned fallback).
- **TR:** keep online TD always; `+TR` adds small L2-normalized episodic bonus `η_tr=0.02`; does not overwrite TD Wq; no SCNC V in plasticity RPE.
- CTB gated until food stores ≥5; NTRAIN×2.

## Seed 900/901 gate (post-fix)
| Arm | 900 | 901 |
|---|---:|---:|
| FLY | 0.567 | 0.817 |
| FLY+TR | 0.500 | 0.567 |
| EMPN | 0.600 | 0.583 |
| EMPN-noCFS | 0.600 | 0.583 |
| EMPN+TR | 0.650 | (see diag json) |

EMPN matches noCFS (CFS no longer harms). FLY+TR within ~15–25 pp of FLY; Wq~1.0.

## Full rerun
10 bases × ±TR × seeds 0–7 → `out/runs_v2b.jsonl` → `RESULTS_empn_v2b.md`.
