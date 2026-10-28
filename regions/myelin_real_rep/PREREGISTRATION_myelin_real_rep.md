# Preregistration: H01 myelin REAL vs SHUF replication (Level-2 check)

**Locked:** 2026-10-28, BEFORE any `myelin_real_rep` eval scores.  
**Parent win:** `regions/myelin_real` HUMAN_LAYERED REAL_MYELIN NARMA 0.497 vs SHUF 0.627, p_bonf=0.002 (two-sided ×6).

## Question

Does real human myelin *placement* still beat shuffled placement on NARMA under (a) 24 new seeds, (b) multiple shuffle draws, (c) stronger structure-preserving shuffles, (d) per-arm ridge α, (e) leak robustness?

## Network / delays (frozen)

- Network: **HUMAN_LAYERED only** (the fair win). FRANKEN exploratory optional later, not primary.
- Myelin map: `hybrid_human/data/human_myelin.npz` (same as parent).
- Delay rule unchanged: `d = 1 + round((1 - clip(mf,0,1))*(dmax-1))`, dmax=5, NaN→median; delays on **human-pre** edges only.
- Fast kernels + FIXED delay CSR (reuse `myelin_real.DelayReservoirFast` / same leak-integrate recipe).

## Arms

| Arm | Myelin map |
|---|---|
| NODELAY | all d=1 |
| REAL_MYELIN | real mf → delays |
| SHUF_MYELIN | global shuffle of mf across human neurons |
| LAYER_SHUF | shuffle mf **within each cortical layer** (L2/3, L4, L5, L6) independently — preserves layer-wise myelin histogram |
| EI_SHUF | shuffle mf **within E (sign=+1) and within I (sign=−1)** separately |

For SHUF_MYELIN / LAYER_SHUF / EI_SHUF: **3 independent shuffle draws** per eval seed (`shuffle_id ∈ {0,1,2}`), RNG `seed + 909 + 10007*shuffle_id` (LAYER/EI use same base + arm salt). Aggregate for primary: **per-seed mean** over the 3 draws, then compare n=24 REAL vs n=24 SHUF means.

## Seeds / leaks / readout

- **Eval seeds:** 100–123 (24 NEW; never 0–7 from parent, never tune seeds).
- **Tune seeds:** 90–92 only — pick per-arm ridge α from `{1e-8,1e-6,1e-4,1e-2,0.1,1,10}` minimizing mean NARMA on tune seeds (shuffle arms use `shuffle_id=0` only during tune). Lock to `out/alpha_choice.json` before eval.
- **Leak rates:** `{0.1, 0.2, 0.3}`. Primary analysis at **leak=0.2**. Other leaks = robustness (report; do not expand primary Bonferroni).
- Tasks logged: MC ↑, NARMA ↓, PC_HARD ↑. Primary uses NARMA only.

## Primary test (one family)

- **Contrast:** REAL_MYELIN vs SHUF_MYELIN on **NARMA**, leak=0.2.  
- **Direction:** one-sided — REAL better means **lower** NARMA (`mannwhitneyu(..., alternative='less')` on the 24 per-seed values; SHUF = mean of 3 draws).  
- **Claim rule:** significant at α=0.05 one-sided **and** mean(REAL) < mean(SHUF). No Bonferroni on this single primary.  
- Secondary (exploratory, raw p + Bonferroni note): REAL vs LAYER_SHUF, REAL vs EI_SHUF, REAL vs NODELAY on NARMA; same for MC/PC_HARD; leak 0.1/0.3 mirrors. Two-sided for non-primary unless noted.

## Predictions (locked)

- **P1:** REAL NARMA < SHUF NARMA at leak=0.2 (replication of parent).  
- **P2:** LAYER_SHUF and EI_SHUF sit between REAL and global SHUF if layer/E-I myelin structure carries the effect.  
- **P3:** Effect direction stable across leaks 0.1/0.3 (magnitude free).

## Ops / honesty

- WORKERS≤2, nice. No GitHub push.  
- If coverage of myelin artifact drops below parent gate, STOP.  
- Report negatives; do not claim biology from REAL vs NODELAY alone.
