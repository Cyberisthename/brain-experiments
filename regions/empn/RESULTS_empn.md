# EMPN maze results

Prereg: `PREREGISTRATION_empn.md` (with SCNC, CTB/MGM/CVB addenda written before results). 18 arms × 8 seeds = 144 jobs. Eval = 60 episodes after 200 training. Grid 7×7 (logged deviation from 9×9).

## Success, steps, risk (means over 8 seeds)

| Agent | success | no-odour success | steps to food | threat hit (loom) | shortcut when safe | shortcut under loom | CA3 clutter |
|---|---:|---:|---:|---:|---:|---:|---:|
| **RANDMOD** | **0.569** | 0.548 | 14.9 | 0.108 | 0.694 | 0.108 | — |
| EMPN-TBUstatic | 0.067 | 0.057 | 25.7 | 0.068 | 0.151 | 0.068 | 0 |
| EMPN-noTBU | 0.050 | 0.045 | 37.8 | 0.063 | 0.169 | 0.063 | — |
| EMPN | 0.035 | 0.030 | 17.1 | 0.087 | 0.281 | 0.087 | 0.59 |
| FLY | 0.035 | 0.039 | 22.9 | 0.037 | 0.080 | 0.037 | — |
| EMPN-FBT | 0.025 | 0.027 | 17.2 | 0.031 | 0.286 | 0.031 | 0.55 |
| EMPN-noVTM | 0.021 | 0.024 | 39.8 | 0.063 | 0.064 | 0.063 | 1.79 |
| EMPN-noCFS | 0.019 | 0.021 | 45.3 | **0.505** | 0.558 | 0.505 | 1.23 |
| EMPN+SCNC-Vmean | 0.012 | 0.012 | 14.0 | 0.049 | 0.084 | 0.049 | 1.73 |
| EMPN+SCNC+MGM | 0.010 | 0.009 | 38.1 | 0.030 | 0.133 | 0.030 | 1.75 |
| EMPN+SCNC / +ALL3 / EMPN-RCX | 0.008 | ~0.006 | 14–36 | 0.02–0.03 | 0.06–0.19 | 0.02–0.03 | ~1.7–1.8 |
| EMPN+SCNC+CVB | 0.006 | 0.006 | 27.3 | 0.029 | 0.049 | 0.029 | 1.73 |
| EMPN+SCNC+CTB | 0.002 | 0.000 | 43 | 0.043 | 0.083 | 0.043 | 1.80 |
| EMPN+SCNC-Vshuf / +RAND500 / +ALL3RAND | **0.000** | 0 | — | 0.04–0.17 | 0.04–0.39 | 0.04–0.17 | — |

Worst-seed success is 0 for every EMPN-family arm. RANDMOD's worst seed is 0.283.

## Which parts matter

| Part | What happens when you remove / change it | Matters? |
|---|---|---|
| TBU (hippocampus) | noTBU 0.050, TBUstatic 0.067 — both ≥ EMPN 0.035 | **No.** Plastic memory does not help; freezing it is slightly better. |
| VTM (valence gate) | noVTM 0.021 ≈ EMPN | Weak / none. |
| CFS (planner) | noCFS threat-hit 0.505 vs EMPN 0.087 | **Yes for risk.** Without CFS the agent walks into the loom zone half the time. Success stays near floor either way. |
| CA1→CX targets | E-PG 0.035, FBT 0.025, RCX 0.008 | No clear benefit for E-PG. |
| SCNC | EMPN+SCNC 0.008 < EMPN 0.035; Vmean 0.012, Vshuf/RAND500 0.000 | **Hurts** vs plain EMPN. Gating timing does not reliably help (S1 fails after correction). |
| CTB / MGM / CVB / ALL3 | all ≤ 0.010 success; none beats SCNC base after correction | **None helps.** |
| Random 400-unit add-on (RANDMOD) | **0.569** | The only thing that clearly works. Extra random recurrent capacity in the Q readout beats the whole EMPN stack. |

## Predictions

| # | Prediction | Held? |
|---|---|---|
| E1–E4 (prereg body: EMPN > FLY / RANDMOD / ablations) | **Failed.** RANDMOD >> EMPN. EMPN ≈ FLY. No ablation vs EMPN survives Bonferroni except RANDMOD *beating* EMPN (p_bonf=0.0055). |
| S1 (SCNC > Vmean, Vshuf, RAND500) | **Failed** after correction. Directionally SCNC ≥ Vshuf/RAND500 (0.008 vs 0), but p_bonf > 0.5. SCNC is worse than plain EMPN. |
| O1 (CTB ≥ base, lower threat) | **Failed.** CTB success 0.002 < base 0.008; threat hits *higher* (0.043 vs 0.022). |
| O2 (MGM lowers CA3 clutter, success unchanged) | **Failed on clutter** (1.751 ≈ 1.742, p=0.88). Success unchanged, yes. |
| O3 (CVB: more shortcuts when safe, fewer under loom) | **Failed.** Safe shortcut 0.049 < base 0.063; under loom 0.029 ≈ 0.022. |
| O4 (ALL3 > ALL3RAND) | Directionally yes (0.008 vs 0.000) but does **not** survive correction (p=0.38). |

## Honest takeaway
On this maze, the Episodic Meta-Plasticity Node and its add-ons (plastic hippocampus, SCNC valence gate, CTB rollouts, MGM sparsity, CVB curiosity/fear) do not beat a plain fly readout, and they lose badly to a same-size random recurrent module. The one clear EMPN-family finding is that **CFS planning cuts threat exposure** (0.09 vs 0.50 without it). Everything else is at the floor; differences among SCNC/organs are noise. This matches the earlier fragile smoke tests and the plastic-hippo PC result (plasticity ≈ frozen ≈ CX).

Full tables: `out/analysis.md`. Raw: `out/runs.jsonl`.
