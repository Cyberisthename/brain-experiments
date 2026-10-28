# Preregistration: Plasticity boost sweep
Written BEFORE any plasticity-boost result. Queued after the EMPN v2 organ-fix rerun.

## Arms / grid
Plastic arms only: `EMPN+TR`, and the fixed organ arms from the organ-fix rerun (`EMPN+SCNC+TR`, `+CTB+TR`, `+MGM+TR`, `+CVB+TR`, `+ALL3+TR`) with shared gates.
Sweep (orthogonal, seeds 900/901 for tune; then full 0–7 at the chosen pair + a sparse grid):
- η multiplier ∈ {0.5, 1, 2, 4, 8} on the three-factor / TBU eta
- τ ∈ {0.85, 0.9, 0.95} (eligibility decay)
Stability: per-synapse clip ±2, row-norm as in TBU.norm, Wq clip ±5.

## Metrics
- Eval success (60 eps)
- Learning speed: train episodes until rolling-20 success ≥ 50% (∞ if never)
- CA3 clutter, threat-under-loom

## Predictions
- P1: Intermediate η (1–2×) beats 0.5× and 8× on success (8× overfits / destabilizes).
- P2: τ=0.9 ≥ τ=0.85 on learning speed for EMPN+TR.
- Bonferroni over the primary η contrasts at best τ.
