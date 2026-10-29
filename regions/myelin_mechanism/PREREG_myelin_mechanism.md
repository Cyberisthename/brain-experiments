# Preregistration: myelin_mechanism — why real H01 myelin beats shuffle on NARMA

**Locked:** 2026-10-29 08:01 America/New_York (EDT), BEFORE any myelin_mechanism eval scores.  
**Dir:** `/workspace/flybrain/regions/myelin_mechanism/`  
**Motivation (prior, not peeking at this study’s scores):** `myelin_real_rep` HUMAN_LAYERED NARMA REAL 0.393 vs SHUF 0.557; LAYER_SHUF 0.441; EI_SHUF 0.556. Layer averages help some; E/I class averages do not.

## Question

Which **synthetic delay-assignment rules** on the **frozen real HUMAN_LAYERED graph** recover ≥50% of the REAL−SHUF NARMA gap without beating REAL, with a one-sided test vs SHUF?

## Graph / delay map (locked)

- Network: `HUMAN_LAYERED` (hybrid_human graft; human module N≈2087 + fly CX). Same builder as myelin_real_rep.  
- Real myelin_frac from `hybrid_human/data/human_myelin.npz`; NaN→median.  
- Map: `d = 1 + round((1−clip(mf,0,1))·(DMAX−1))`, DMAX=5; edge delay = presynaptic human neuron’s d (non-human-pre edges stay 1).  
- Leak=0.2 (primary leak from replication). Tasks: **NARMA-10** (primary), MC / PC_HARD secondary.  
- Ridge α per arm on seeds **90–92**. Eval seeds **800–823** (n=24).

## Descriptive extract (pre-eval; not a hypothesis test)

On the human module alone: Spearman myelin vs out/in-degree, vs BFS depth from L4, mean myelin by layer and by E/I. Saved to `data/desc_corr.npz` / reported in RESULTS. Used only to **motivate** how rules (2)(4) map ranks; rules themselves are fully specified below without fitting to NARMA.

## Arms (same W; only delay assignment changes)

| Arm | Rule |
|---|---|
| `REAL` | real myelin_frac → delay |
| `SHUF` | global permute myelin (exact marginal; randomize assignment) — rule (5) / diversity control |
| `RULE_LAYER` | (1) for each neuron, sample mf from the **empirical mf distribution of its layer** (with replacement), independent of identity within layer |
| `RULE_DEGREE` | (2) assign mf by **isotonic match to in-degree ranks**: sort neurons by indeg; assign the sorted real mf values in the same order as the empirical Spearman sign (if ρ(mf,indeg)>0, high-indeg gets high-mf; else reverse). Ties broken by seed. |
| `RULE_EI_LAYER` | (3) sample mf from empirical distribution within **(sign × layer)** bins |
| `RULE_FF_DEPTH` | (4) BFS depth from L4 on human subgraph; assign mf by isotonic match to depth ranks using sign of Spearman(mf, depth) |

NODELAY is secondary only (not in primary Bonferroni).

## Primary hypotheses (Bonferroni ×4 over rule arms)

For each rule arm R ∈ {RULE_LAYER, RULE_DEGREE, RULE_EI_LAYER, RULE_FF_DEPTH}:

1. One-sided Wilcoxon on paired seeds: NARMA(R) < NARMA(SHUF) (lower better), p_bonf = min(1, 4·p_raw) < 0.05.  
2. **Recovery:** (mean_SHUF − mean_R) / (mean_SHUF − mean_REAL) ≥ 0.50.  
3. **No exceed REAL:** mean_R ≥ mean_REAL − 1e-6 (rule must not beat REAL on mean NARMA).

A rule **passes** iff (1)(2)(3) all hold.  
**Study verdict SUPPORTED** if ≥1 rule passes; else **NOT SUPPORTED**. Report each rule honestly.

## Ops

WORKERS=2; do not kill gated_team / other jobs. Push code+prereg+results. Reservoir ≠ biology claim.
