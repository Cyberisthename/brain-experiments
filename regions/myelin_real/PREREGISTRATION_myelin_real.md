# Preregistration: Real H01 myelination → conduction delays

Written BEFORE any myelin_real task scores. Locked 2026-10-28.

## Data (Part 1 — already streamable; must succeed before Part 2)

- Source: `gs://h01-release/data/20210601/c3/subcompartments/annotations` (public HTTPS).
- Relationship index `skeleton_id` = c3 segment id = `human_module.npz` `root_id`.
- `class_label`: 0=axon, 1000/1001=myelinated axon, 1=dendrite, 3=soma, 5=AIS, 1002/1003=myelinated fragment.
- Per neuron: `myelin_frac = n_myelinated_axon / (n_axon + n_myelinated_axon)`; NaN if no axon points.
- Artifact: `regions/hybrid_human/data/human_myelin.npz` + `SOURCE_myelin.json`.
- Gate: if fewer than 100/2087 neurons have axon annotations, STOP after Part 1 and report why.

## Delay rule (Part 2)

- Integer delays `d ∈ {1..dmax}`, `dmax=5` (same clamp as `regions/myelin`).
- Map: `d = 1 + round((1 - clip(myelin_frac,0,1)) * (dmax-1))` so myelinated → shorter delay.
- NaN myelin_frac (no axon): fill with median myelin_frac over neurons that have axon annotations.
- Per-edge delay comes from the **presynaptic** neuron’s myelin_frac.
- Applied on edges whose presynaptic neuron is in the **human** block only; non-human presynaptic edges stay `d=1` (instant in the ring buffer).

## Networks

1. **HUMAN_LAYERED** — fly + H01 layered graft (`hybrid_human.build_arm("HUMAN_LAYERED", …)`).
2. **FRANKEN** — mouse + human + fly_cx (`frankenstein.build_arm("FRANKEN", …)`).

## Arms

1. **NODELAY** — all delays = 1.
2. **REAL_MYELIN** — delays from real myelin_frac map (frozen).
3. **SHUF_MYELIN** — same delay *values* after shuffling myelin_frac across human neurons once per seed (frozen). Same marginal delay distribution; breaks neuron↔myelin assignment.
4. **ADAPT** — start from random delays in {1..dmax} on human-pre edges; activity-dependent shorten/lengthen as in `regions/myelin` (ADAPT). Use-dependent control, not a real-myelin claim.

## Tasks / seeds / stats

- Tasks: MC ↑, NARMA ↓, PC_HARD ↑ (same cue_frac=0.35, noise=1.5 as frankenstein fast). Easy PC skipped (saturated).
- Seeds: 0..7. Leak=0.2. Spectral radius scaling as in parent runners. Fast kernels.
- Primary contrast: **REAL_MYELIN vs SHUF_MYELIN** per (network × task).
- Also report vs NODELAY and ADAPT.
- Stats: Mann–Whitney U two-sided; Bonferroni over the 6 primary contrasts (2 networks × 3 tasks) for REAL vs SHUF. Exploratory contrasts (vs NODELAY/ADAPT) reported with raw p only.

## Predictions (locked)

- **P1**: REAL_MYELIN ≠ SHUF_MYELIN on at least one task (direction free; myelin map structure matters beyond the delay histogram).
- **P2**: REAL_MYELIN ≥ NODELAY on PC_HARD (myelinated human axons as short-delay highways help pattern completion under noise); free on MC/NARMA.
- **P3**: ADAPT may help MC relative to a random FIXED init (replicating the synthetic myelin ADAPT>ADAPT-SHUF spirit) but is **not** predicted to beat REAL_MYELIN; ADAPT is a mechanism check.

## Honesty rules

- If Part 1 coverage is unusable → stop; no Part 2.
- Report negatives. Do not claim “biology wins” from REAL beating NODELAY alone — SHUF is the structure control.
- Do not starve mouse hybrid / mature jobs (nice, WORKERS≤2).
