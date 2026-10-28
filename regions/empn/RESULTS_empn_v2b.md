# EMPN v2b organ-fix results

Prereg: `PREREGISTRATION_empn_v2_organ_fix.md`. Diagnostics: `DIAGNOSIS_organ_harm.md`, `out/diag_v2b_cfs_tr.json`.

**160 jobs** (10 bases × ±TR × seeds 0–7). τ = **0.85** (locked from prior tune; restored after box wipe mid-run).

## Root cause (from broken v2)

Two learner bugs, not bad biology:

1. **CFS overrode food-seeking.** Old objective was effectively risk-only once goals gated off; bias added 1:1 onto Q; unseen WM self-loops inflated V; non-stationary food locked stale goals. EMPN success **0.000** vs EMPN-noCFS **0.500**.
2. **+TR replaced TD and used an unnormalized trace.** Online TD skipped under `use_tr`; episodic `α·e·RPE` blew `‖Wq‖` to ~10–14 (healthy ~0.7–1.2). FLY **0.463 → FLY+TR 0.025**.

## Fixes applied in v2b

- **CFS → `cfs_er`:** `w_r·E[rew] − w_risk·risk·loom`; fall back (bias=0) if `seen[p]<2` or no reward signal; bootstrap only through **seen** transitions; per-episode rew/risk decay; **assist-only** (never flip `argmax(q+bonus)`).
- **TR:** keep online TD always; `+TR` adds small L2-normalized episodic bonus `η_tr=0.02`; does not overwrite TD Wq; no SCNC V in plasticity RPE.
- CTB gated until food stores ≥5; NTRAIN×2.

## Success (mean ± sd over 8 seeds)

| Agent | success | sd | train food | threat loom | steps | wq | sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| RANDMOD | 0.700 | 0.174 | 0.356 | 0.061 | 18.3 | 1.168 | 49.6 |
| EMPN+SCNC+ALL3+TR | 0.637 | 0.170 | 0.281 | 0.078 | 20.2 | 1.121 | 235.4 |
| EMPN+SCNC+MGM+TR | 0.637 | 0.170 | 0.281 | 0.078 | 20.2 | 1.121 | 190.4 |
| EMPN+SCNC | 0.621 | 0.237 | 0.312 | 0.067 | 20.3 | 1.094 | 170.8 |
| EMPN+SCNC+CVB | 0.621 | 0.237 | 0.312 | 0.067 | 20.3 | 1.094 | 171.2 |
| EMPN+SCNC+CTB | 0.621 | 0.237 | 0.312 | 0.067 | 20.3 | 1.094 | 213.8 |
| EMPN+SCNC+MGM | 0.598 | 0.186 | 0.256 | 0.128 | 21.3 | 1.031 | 172.4 |
| EMPN+SCNC+ALL3 | 0.598 | 0.186 | 0.256 | 0.128 | 21.3 | 1.031 | 218.4 |
| FLY+TR | 0.592 | 0.163 | 0.288 | 0.092 | 22.6 | 1.116 | 51.8 |
| EMPN+SCNC+ALL3RAND | 0.585 | 0.256 | 0.300 | 0.088 | 18.8 | 1.019 | 172.7 |
| RANDMOD+TR | 0.583 | 0.275 | 0.263 | 0.059 | 15.5 | 1.210 | 53.9 |
| EMPN+SCNC+ALL3RAND+TR | 0.579 | 0.259 | 0.263 | 0.075 | 18.3 | 1.112 | 191.2 |
| FLY | 0.562 | 0.237 | 0.231 | 0.082 | 25.7 | 0.998 | 50.7 |
| EMPN | 0.556 | 0.231 | 0.312 | 0.111 | 22.4 | 1.021 | 103.8 |
| EMPN-noCFS | 0.556 | 0.231 | 0.312 | 0.111 | 22.4 | 1.021 | 88.1 |
| EMPN+TR | 0.508 | 0.149 | 0.219 | 0.095 | 22.0 | 1.075 | 260.8 |
| EMPN-noCFS+TR | 0.508 | 0.149 | 0.219 | 0.095 | 22.0 | 1.075 | 240.7 |
| EMPN+SCNC+CTB+TR | 0.506 | 0.265 | 0.244 | 0.091 | 23.4 | 1.051 | 247.7 |
| EMPN+SCNC+CVB+TR | 0.506 | 0.265 | 0.244 | 0.091 | 23.4 | 1.051 | 200.8 |
| EMPN+SCNC+TR | 0.506 | 0.265 | 0.244 | 0.091 | 23.4 | 1.051 | 197.9 |

## +TR vs no-TR

| Base | noTR | +TR | Δ | wq noTR → +TR |
|---|---:|---:|---:|---:|
| EMPN | 0.556 | 0.508 | -0.048 | 1.02 → 1.07 |
| EMPN+SCNC | 0.621 | 0.506 | -0.115 | 1.09 → 1.05 |
| EMPN+SCNC+ALL3 | 0.598 | 0.637 | +0.040 | 1.03 → 1.12 |
| EMPN+SCNC+ALL3RAND | 0.585 | 0.579 | -0.006 | 1.02 → 1.11 |
| EMPN+SCNC+CTB | 0.621 | 0.506 | -0.115 | 1.09 → 1.05 |
| EMPN+SCNC+CVB | 0.621 | 0.506 | -0.115 | 1.09 → 1.05 |
| EMPN+SCNC+MGM | 0.598 | 0.637 | +0.040 | 1.03 → 1.12 |
| EMPN-noCFS | 0.556 | 0.508 | -0.048 | 1.02 → 1.07 |
| FLY | 0.562 | 0.592 | +0.029 | 1.00 → 1.12 |
| RANDMOD | 0.700 | 0.583 | -0.117 | 1.17 → 1.21 |

## Prereg contrasts (Mann–Whitney two-sided, Bonferroni ×13)

| Comparison | kind | mean A | mean B | Δ | p | p_bonf | sig |
|---|---|---:|---:|---:|---:|---:|---|
| EMPN vs FLY | fix: EMPN vs FLY | 0.556 | 0.562 | -0.006 | 0.6737 | 1 | False |
| EMPN vs RANDMOD | fix: EMPN vs RANDMOD | 0.556 | 0.700 | -0.144 | 0.1149 | 1 | False |
| EMPN vs EMPN-noCFS | fix: CFS vs noCFS | 0.556 | 0.556 | +0.000 | 1 | 1 | False |
| FLY+TR vs FLY | fix: TR on FLY | 0.592 | 0.562 | +0.029 | 0.636 | 1 | False |
| EMPN+TR vs EMPN | fix: TR on EMPN | 0.508 | 0.556 | -0.048 | 1 | 1 | False |
| RANDMOD+TR vs RANDMOD | fix: TR on RANDMOD | 0.583 | 0.700 | -0.117 | 0.3435 | 1 | False |
| EMPN+SCNC vs EMPN | organ: SCNC | 0.621 | 0.556 | +0.065 | 0.5283 | 1 | False |
| EMPN+SCNC+ALL3 vs EMPN+SCNC | organ: ALL3 vs SCNC | 0.598 | 0.621 | -0.023 | 0.8747 | 1 | False |
| EMPN+SCNC+ALL3RAND vs EMPN+SCNC | organ: ALL3RAND vs SCNC | 0.585 | 0.621 | -0.035 | 0.7522 | 1 | False |
| EMPN+SCNC+CTB vs EMPN+SCNC | organ: CTB | 0.621 | 0.621 | +0.000 | 1 | 1 | False |
| EMPN+SCNC+MGM vs EMPN+SCNC | organ: MGM | 0.598 | 0.621 | -0.023 | 0.8747 | 1 | False |
| EMPN+SCNC+CVB vs EMPN+SCNC | organ: CVB | 0.621 | 0.621 | +0.000 | 1 | 1 | False |
| EMPN+SCNC+ALL3+TR vs EMPN+SCNC+ALL3RAND+TR | organ+TR: ALL3 vs RAND | 0.637 | 0.579 | +0.058 | 0.712 | 1 | False |

## v2 → v2b (same maze, fixed learner)

| Arm | v2 success | v2b success | note |
|---|---:|---:|---|
| RANDMOD | 0.565 | **0.700** | still best baseline |
| FLY | 0.463 | 0.562 | |
| FLY+TR | **0.025** | **0.592** | TR no longer destroys |
| EMPN | **0.000** | **0.556** | CFS no longer collapses |
| EMPN-noCFS | 0.500 | 0.556 | |
| EMPN+TR | ~0.10 | 0.508 | |
| EMPN+SCNC±organs | ≲0.03 | 0.50–0.64 | off the CFS floor |
| ‖Wq‖ | ~10–14 under +TR | **0.73–1.40** | healthy |

## Identity checks (bit-identical across seeds)

- `EMPN` ≡ `EMPN-noCFS` (and `EMPN+TR` ≡ `EMPN-noCFS+TR`): under assist-only + seen/reward fallback, CFS never changes the action → CFS is a **no-op** in this maze.
- `EMPN+SCNC+CTB` ≡ `EMPN+SCNC` and `EMPN+SCNC+CVB` ≡ `EMPN+SCNC`: CTB/CVB add **zero** behavioral effect (CTB gated; CVB unused or null).
- `EMPN+SCNC+ALL3` ≡ `EMPN+SCNC+MGM`: ALL3 reduces to MGM alone.

## Honest takeaway

1. **The two learner bugs are fixed.** EMPN recovers from 0% to ~56%; FLY+TR from 2.5% to ~59%; Wq stays ~1.0. That was the point of v2b.
2. **CFS no longer harms — but also no longer helps.** It is behaviorally identical to noCFS. Assist-only + fallback neutralized the override; it did not unlock planning value.
3. **+TR no longer explodes Wq.** Deltas are small (±0.03 to −0.12) and **no +TR contrast survives Bonferroni**. TR is safe, not useful.
4. **Organs still do not beat controls after Bonferroni.** SCNC (0.621) edges EMPN (0.556) and FLY (0.562) but p_bonf ≫ 0.05. CTB/CVB are inert. MGM/ALL3 ≈ SCNC. ALL3 vs ALL3RAND is a coin flip.
5. **RANDMOD remains the best arm (0.700).** Extra “brain organs” do not beat a same-size random module on this maze once the learner is fair.
6. Seed variance is large (sd ~0.15–0.27). Claims of organ/TR wins from single seeds would be flukes.

Raw: `out/runs_v2b.jsonl`, curves: `out/curves_v2b.jsonl`. Queue continued to thinking → aetherlink → roles → … → hybrid.
