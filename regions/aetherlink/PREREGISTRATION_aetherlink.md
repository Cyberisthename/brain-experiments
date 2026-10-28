# Preregistration: AetherLink lanes test
Written BEFORE any AetherLink result. Queued to run only after EMPN v2 (and after thinking-time if sequential).

## Setup
- Backbone: 25%-compressed REAL CX (CONSOL and MAGP), same reservoir + ridge readout as `compress/`.
- Lanes: N ∈ {1, 2, 4, 8, 16} parallel continuous **ungated linear** projections into real **P-EN** and **E-PG** cells (no hard top-K anywhere).
- Each lane ℓ has its own W_in^(ℓ) (obs→lane) and U^(ℓ) (lane→P-EN/E-PG); lane state is a small tanh RNN (H=32 units) unless noted.
- Consensus/voting readout: average the N lane readouts (soft vote) for the task target; train with shared ridge / joint SGD as in sag adapters.
- Control **WIDE**: one lane with width H_wide such that total trainable params ≈ N=16 lanes (param-matched), N=1 otherwise identical.

## Tasks / seeds
MC, T0, NARMA (grad clip 1, weight clip ±2 as in SAG). 8 seeds. Methods: CONSOL + MAGP.

## Curve fits (prereg)
For mean MC (and separately T0) vs N, fit:
1. log: y = a + b log N
2. power: y = a N^b
3. saturating: y = a (1 − e^{−b N})
Report AIC; primary claim is which family wins on MC for MAGP.

## Predictions
- A1: Performance improves with N at least through N=4 (MC↑, T0↓).
- A2: Saturating or log fit beats power by AIC on MC.
- A3: N=16 lanes ≥ WIDE control on MC (consensus helps beyond raw param count).
- A4: Heading-cell targeting matters: MAGP (more E-PG kept) ≥ CONSOL at matched N.
Bonferroni on A1 (N=1 vs N=4, N=1 vs N=16) and A3 (N=16 vs WIDE), ×2 methods.
