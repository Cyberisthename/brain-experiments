# Preregistration: EMPN v2 — eligibility-trace / RPE learning fix
Written BEFORE any v2 result. Queued to run only after the SAG test finishes.

## Diagnosis plan (run first, logged, same code for all arms)
- Plot per-episode success and return during the 200 train episodes for FLY, RANDMOD, EMPN (seed 0).
- Check whether the Q update `Wq += α/(φ·φ)·δ·φ` ever moves: ‖ΔWq‖, and whether δ is near 0.
- Suspected shared bugs (to confirm, then fix for ALL arms including FLY and RANDMOD):
  1. Normaliser `φ·φ` can be tiny after z-scoring → huge unstable updates, or huge → dead updates.
  2. No eligibility on the policy: credit only hits the last action.
  3. Hippocampal store is gated by noisy end-of-episode valence; mid-path structure is lost.
- Fixes applied equally to every arm (fairness):
  - Q update uses RMSProp-style denom `φ·φ + ε` with ε=1.0, lr clipped, Wq clipped to ±5.
  - **Three-factor eligibility traces** (optional arm flag `+TR`):
    - Per-synapse (or per-feature-action) trace e decays with τ (tuned on seeds 900/901 only, grid {0.85,0.9,0.95}); bumped by pre×post (for TBU) or φ_a (for policy).
    - dW = η · e · (R − R̄), R̄ = exponential moving average of return (RPE baseline).
    - Applied to: CA3 recurrents, mossy fibres, CA1→CX, and the policy readout.
    - When SCNC is present, the broadcast RPE uses SCNC V (signed: +food, −threat) instead of raw reward.
  - Clipping/row-normalisation on TBU weights as in v1.

## Arms (each × {no-trace, +TR})
FLY, RANDMOD, EMPN, EMPN+SCNC, EMPN+SCNC+CTB, +MGM, +CVB, +ALL3, +ALL3RAND, EMPN-noCFS.
= 10 × 2 × 8 seeds.

## Metrics
Success, steps to food, threat under loom, shortcut safe/loom; learning curves (train success in 20-episode bins); ‖ΔW‖ over training.

## Predictions
- V2-1: With shared learner fixes, RANDMOD and FLY rise above v1 floors; if not, the maze itself is too hard / reward too sparse.
- V2-2: +TR > no-trace on success for EMPN and EMPN+SCNC (traces fix credit assignment).
- V2-3: EMPN+SCNC+TR ≥ EMPN+TR (V as RPE broadcast helps).
- V2-4: ALL3+TR > ALL3RAND+TR.
- Bonferroni over the 4 predictions' primary comparisons (and organ vs base as secondary).

## Diagnosis addendum (logged BEFORE v2 full run; seeds 0 only, shared-fix prototype)

| agent | eval success | train food-rate (20-ep bins) | ‖Wq‖ start→end | last-40 train food |
|---|---:|---:|---:|---:|
| FLY | 0.033 | 0.20 | 0.004→0.069 | 0.0 |
| RANDMOD | 0.400 | 0.40 | 0.001→0.052 | 0.5 |
| EMPN | 0.000 | **0.00** | 0.004→0.088 | 0.0 |

Finding: the Q update *does* move weights, but EMPN never reaches food in 200 train episodes (FLY rarely). Floor is exploration / representation, not a silent update. τ grid on EMPN+TR was all 0.0 — retuned on RANDMOD+TR (seeds 900/901) which has a learning signal. Additional shared fixes applied to ALL arms before the full run: feature L2-normalisation, ε-greedy 1.0→0.1, count-based bonus β=0.3/√N(s,a), α=0.1, Wq±5, denom ε=1.

## Diagnosis addendum 2 (BEFORE full v2 run; shared fixes applied to ALL arms)

### Why RANDMOD ~0.88 while FLY/EMPN ~0
| readout | eval success (seed 0, shared TD) | note |
|---|---:|---|
| raw obs (11-d) | 0.75 | Markov state is enough |
| DN only (1304-d) | 0.17 | DN std≈0.004; effective rank≈23; linearly decodes pos (R²=0.99) but after one L2-norm the near-constant dims dominate the learning signal |
| DN + obs, single L2 | 0.22 | obs drowned by DN |
| DN + obs, **block-norm** | **0.60** | each block L2-normalised then concat |
| RANDMOD z-400 | 0.92 | rich, high-variance features |
| DN + z (RANDMOD) | 0.85 | |

Ablation of EMPN parts with the new readout:
| variant | success |
|---|---:|
| EMPN-noCFS | 0.82 |
| CFS + no TBU.store | 0.75 |
| CFS + TBU.store | 0.00 |
| full EMPN | ~0 |

CFS risk avoidance helps; TBU recall goals poison CFS until memories are reliable.

### Shared fixes (every arm, including FLY and RANDMOD)
1. Readout = block-norm(DN) ⊕ block-norm(z if RANDMOD) ⊕ block-norm(obs) ⊕ bias.
2. Explore-until-first-food (ε=1), then ε: 1→0.1; eval ε=0.05.
3. Count bonus β=0.3/√N(s,a); α=0.1; Wq clip ±5; TD denom ε₀=1.
4. CFS (EMPN arms only): gate recall-goals until ≥5 food episodes; scale by model confidence; keep risk term.
5. +TR: pure three-factor dW=η·e·(R−R̄) on policy + CA3/mossy/CA1→CX eligibility; SCNC V mixed into RPE when present. τ tuned on RANDMOD+TR seeds 900/901 only.

Prior partial `runs_v2.jsonl` (7 rows, pre-fix) discarded before the full run.
