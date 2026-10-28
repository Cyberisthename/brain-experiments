# Preregistration: Synthetic myelination (conduction delays)
Written BEFORE any myelin result. Queued after the plasticity boost.
Note: insects lack true myelin; this is an analogous delay-plasticity rule on edges.

## Rule
- Each module/CX projection edge gets a conduction delay d ∈ {1..5} steps (initial draw from a truncated geometric / categorical).
- Spike/rate at t arrives at t+d.
- **Adaptive myelination**: on coactive pre/post (trace product), d ← max(1, d−δ) with small δ; unused edges drift back toward longer delay.
- Stability: delays integer-clamped to [1,5].

## Arms
1. NODELAY — baseline (d=0 effectively instantaneous)
2. FIXED — delays drawn once, frozen
3. ADAPT — activity-dependent shortening
4. ADAPT-SHUF — same adaptive rule but coactivity credits are shuffled across edges (control)

## Tasks
EMPN v2 maze (FLY + EMPN-noCFS + EMPN+TR with organ fixes) and compress MC/T0 on REAL MAGP 25% CX with delayed recurrent edges.

## Seeds / stats
8 seeds. Bonferroni on ADAPT vs FIXED and ADAPT vs ADAPT-SHUF for maze success and MC.

## Predictions
- M1: FIXED ≠ NODELAY (delays change dynamics; direction free).
- M2: ADAPT > FIXED on maze success and/or MC.
- M3: ADAPT > ADAPT-SHUF (usage-specificity required).
