# Diagnosis: why organs / +TR hurt (EMPN v2 final)

Date: 2026-10-27. Evidence: full `runs_v2.jsonl` (160 jobs) + seeds 900/901 instrumentation.

## Key v2 numbers (mean success, seeds 0–7)
| Arm | success | notes |
|---|---:|---|
| RANDMOD | 0.565 | best baseline |
| EMPN-noCFS | 0.500 | TBU OK without CFS |
| FLY | 0.463 | TD works |
| RANDMOD+TR | 0.123 | TR hurts |
| FLY+TR | 0.025 | TR destroys FLY |
| EMPN | **0.000** | CFS collapse |
| EMPN+SCNC±organs | ≲0.03 | floor on CFS failure |

## Plain root cause
**Two learner bugs, not bad biology:**

1. **CFS overrode food-seeking.** Objective was effectively risk-only (goals gated off); bias added 1:1 onto Q; unseen WM self-loops inflated V; food alternates every 10 eps so stale value locked onto the previous food. Agent stopped seeking food → Q never learned. Ablation: EMPN 0% vs noCFS 50%.

2. **+TR replaced TD and used an unnormalized trace.** Online TD was disabled under `use_tr`; episodic `α·e·RPE` blew `‖Wq‖` to ~10–14 (healthy TD ~0.7). FLY 46% → FLY+TR 2.5%.

Organs (SCNC/CTB/MGM/CVB) sat on the CFS floor (~0–3%).

## Fix (v2b) + seed 900/901 gate
- CFS → expected-reward−risk, seen-only bootstrap, rew decay, **assist-only** (never flip policy argmax).
- TR → keep TD; small L2-normalized episodic bonus.
- Post-fix diag: EMPN=noCFS (0.60/0.58); FLY+TR 0.50/0.57 vs FLY 0.57/0.82; EMPN+TR 0.65/0.67; Wq~1.0.
