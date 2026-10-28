# Preregistration: plastic hippocampus (written 2026-10-27, before any test-seed run)

User spec update. Builds on regions/hippo (same EC/DG/CA3/CA1 sizes and wiring).

## Learning rules (storage phase, before each pattern-completion trial set)
- The 20 stored patterns are shown clean for 5 epochs × 10 steps. Weights are frozen during testing.
- **CA3 (CA3→CA3 recurrents AND DG→CA3 mossy fibres):** covariance Hebbian rule with a BCM sliding threshold.
  - Update: dw = η · post·(post−θ) · (pre − mean pre).
  - θ is a running mean of post², with decay 0.9.
  - Only existing synapses are updated, so the connectome-style sparsity is kept.
  - After each epoch, weights are clipped to ±3 and each row is renormalised to its original L1 norm, for stability. This is Hopfield-like attractor storage.
- **CA1:** a GeneRec/contrastive delta rule on CA3→CA1.
  - dw = η · (CA1_plus − CA1_minus) · CA3.
  - Plus phase = tanh(CA3 input + direct EC input). Minus phase = tanh(CA3 recall only).
  - This minimises the mismatch between the EC cue and the CA3 recall.
- **η = 15.** It was calibrated on tuning seed 900 only, with no task score used, so that the module's weights change by about 24% (relative L1).
  - At η = 0.05 they changed only 1%, because CA3 activity is low (about 0.03), so the rule would have been a no-op.
- **CA1→CX targets (30 per CA1 cell),** from FlyWire annotations (data/ann.tsv, cell_type):
  - EPG: E-PG/P-EG/EPGt, 71 cells.
  - FBT: fan-shaped-body tangential FB*, 591 cells.
  - RCX: random CX, 2875 cells.

## Conditions
| Name | What it is |
|---|---|
| CX | CX alone |
| HP_EPG | Plastic hippo, E-PG/P-EG targets (main) |
| HP_FBT | Plastic, FB tangential targets |
| HP_RCX | Plastic, random CX targets |
| HS_EPG | Plasticity OFF |
| RANDP_EPG | Same-size random module, same plasticity applied to random "CA3/CA1" sets |
| SHUFP_EPG | Degree-preserving shuffled hippo, plastic |

## Protocol
- Pattern-completion (PC) task identical to regions/hippo.
- Tuning: a 3×3 leak×ρ grid on seeds 900 and 901. Testing: seeds 0–7.
- Robustness: the worst of 3 random 10% synapse lesions.
- Statistics: two-sided Mann–Whitney tests, Bonferroni over 6 planned comparisons. All are HP_EPG vs: CX, HS_EPG, RANDP, SHUFP, HP_FBT, HP_RCX.

## Predictions
- P1: HP_EPG > HS_EPG (plasticity helps).
- P2: HP_EPG > RANDP and SHUFP (the structure matters).
- P3: HP_EPG > CX. (Note: the static hippo lost to CX before.)
- P4: E-PG vs FBT vs RCX targets: no strong prior, exploratory.

Smoke check (seed 0, val, η=0.05, before calibration): all conditions scored 0.57–0.62 on PC. This is reported for honesty.

## EMPN
The EMPN TBU will be replaced by this plastic module (HP_EPG); see the EMPN prereg addendum.
