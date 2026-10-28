# Plastic hippocampus results

Prereg: `PREREGISTRATION_hippo_plastic.md`. Pattern-completion only. 8 test seeds. η=15 (calibrated on tuning seed 900 for ~24% weight change; no task score used).

## Pattern completion (higher is better)

| Condition | mean | sd | worst seed | worst 10% lesion |
|---|---:|---:|---:|---:|
| CX alone | **0.758** | 0.034 | 0.708 | 0.693 |
| Plastic hippo → E-PG (HP_EPG) | 0.737 | 0.019 | 0.713 | 0.713 |
| Plasticity OFF (HS_EPG) | 0.738 | 0.022 | 0.713 | 0.700 |
| Plastic → FB tangential | 0.724 | 0.021 | 0.690 | 0.685 |
| Plastic → random CX | 0.725 | 0.018 | 0.703 | 0.690 |
| Plastic random module | 0.743 | 0.019 | 0.705 | 0.705 |
| Plastic shuffled | 0.707 | 0.026 | 0.663 | 0.665 |

## Predictions

| # | Prediction | Held? |
|---|---|---|
| P1 | HP_EPG > HS_EPG (plasticity helps) | **Failed.** 0.737 ≈ 0.738, p=0.67. |
| P2 | HP_EPG > RANDP and SHUFP | **Failed** vs RANDP (0.743). Held directionally vs SHUFP (p=0.027) but does **not** survive Bonferroni (p_bonf=0.16). |
| P3 | HP_EPG > CX | **Failed.** CX still best. Same pattern as the static hippo. |
| P4 | E-PG vs FBT vs RCX exploratory | No clear winner. E-PG slightly above FBT/RCX; none significant. |

## Honest takeaway
The BCM/covariance CA3 rule and the CA1 contrastive rule, as implemented, do not improve pattern completion over a frozen hippocampus or over CX alone. Aiming CA1 at E-PG cells instead of random CX also does not help under this test. The plastic random module matches the plastic hippo. Whatever pattern completion CX already does, the added plastic module does not beat it.
