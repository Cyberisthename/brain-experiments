# Results (12 seeds per condition, chatbot repo commit ef496a0)
Circuit: FlyWire v783 central complex (2,875 neurons, 300,234 signed connections). All wirings scaled to the same spectral radius (0.95); only a linear readout is trained.

## Plain-language version
- **Memory:** the real fly wiring remembers past inputs about **2x better** than random wiring (memory capacity 8.8 vs 4.3), and still clearly better than random wiring with the same per-neuron connection counts (6.5). That was the one big real effect, and it was not predicted.
- **But that didn't make it better at harder tasks.** On NARMA-10 (needs memory + nonlinearity) and on your T0 regression the fly wiring was a bit **worse** than random. Remembering more ≠ computing better here.
- **Adapters:** in your T1 code the adapters only contribute 3 tag strings that get hashed into random seeds. No adapter numbers (y_bits/x_bits/z_bits) are used. So they can't carry knowledge into the task, and they didn't help: no significant difference anywhere.

## Table (mean ± std; Mann–Whitney two-sided p vs REAL)
| metric | fly CX (REAL) | Erdős–Rényi | degree-shuffle | p REAL vs ER | p REAL vs DEG |
|---|---|---|---|---|---|
| NARMA-10 NRMSE (lower better) | 0.504 ± 0.026 | 0.466 ± 0.042 | 0.443 ± 0.037 | 0.010 | 0.0006 |
| Memory capacity (higher better) | **8.84 ± 0.41** | 4.27 ± 0.36 | 6.53 ± 0.64 | 3.7e-5 | 3.7e-5 |
| T0 test MSE (lower better) | 0.0620 ± 0.0107 | 0.0356 ± 0.0023 | 0.0430 ± 0.0091 | 3.7e-5 | 0.0005 |
| T1 MSE, with adapters | 0.1299 ± 0.0022 | 0.1285 ± 0.0022 | 0.1286 ± 0.0023 | 0.030 | 0.053 |
| T1 MSE, without adapters | 0.1303 ± 0.0444 | 0.1283 ± 0.0435 | 0.1294 ± 0.0458 | 0.62 | 0.71 |

T1 raw MSE is dominated by the 3 teachers disagreeing with each other (an unavoidable floor). Floor-subtracted excess error (analyze_adapters.py, paired Wilcoxon):
| wiring | with adapters | without | p |
|---|---|---|---|
| REAL | 0.0019 ± 0.0003 | 0.0048 ± 0.0057 | 0.20 |
| ER | 0.0005 ± 0.0001 | 0.0029 ± 0.0044 | 0.30 |
| DEG | 0.0007 ± 0.0002 | 0.0040 ± 0.0073 | 0.15 |
(The adapter set is one fixed set of 3 teachers, so its tiny std is not "stability from adapters" — it's the same task every seed.)

## Preregistered predictions vs outcome
- H1 (REAL ≈ ER within ~10% on NARMA/MC): **wrong on MC** (REAL +107%), roughly right on NARMA (REAL 8% worse, significant).
- H2 (REAL–DEG gap smaller than REAL–ER): **right for MC** (degree structure explains ~half the boost), wrong for NARMA.
- H3 (all wirings same on T0/T1): **wrong on T0** (REAL worse), ~right on T1.
- H4 (adapters no benefit): **supported**.
- T2/T3 not tested (no input→output data for a reservoir to read out) — as preregistered.
- Degree-shuffle dropped 0 edges in all seeds. ~14 tests, no multiple-comparison correction; the MC and T0 effects survive Bonferroni, NARMA vs ER (p=0.010) and T1-with-adapters (p=0.03) do not.

## Caveats
- One circuit (CX), one reservoir design, one leak/radius/ridge setting, untuned for anyone. Results may flip with other hyperparameters.
- Real wiring is a single fixed graph (seed varies inputs/data); nulls get a new graph each seed.
- Signs come from predicted neurotransmitters; synapse counts used as weights; rate model, not spiking.
- Plot: out/comparison.png. Raw: out/results.csv, out/summary.json, out/t1_excess.csv.

---
# Round 2: fair tuning sweep (PREREGISTRATION_2.md, script round2.py)
## Plain-language version
- **The fly brain's memory advantage was an artifact of the settings.** After each wiring got its own best settings, random wiring had MORE memory: ER 17.5 vs fly 13.6. The fly wiring also lost clearly on NARMA-10 and T0.
- What IS real about the fly wiring is that it's **robust**. Its memory stays at 6–13 across every setting tried, while random wiring collapses to under 1 at many settings (low leak with high spectral radius). The round-1 setting (leak 0.5, ρ 0.95) happened to be a bad spot for random wiring.
- Because the advantage didn't survive tuning, the cell-type knockout study was **not run**, as the preregistration said.
- Structurally, the fly wiring is very different. It has 11x more reciprocal (two-way) connections than ER, and its eigenvalues are concentrated near zero with only a few large modes. In practice it behaves like a lower-dimensional system, and that likely explains both the robustness and the lower peak.

## Tuned test results (12 fresh seeds, best (leak, ρ) per wiring and task chosen on validation)
| task | fly CX | ER | degree-shuffle | p REAL vs ER | p REAL vs DEG |
|---|---|---|---|---|---|
| Memory capacity ↑ | 13.61 ± 0.44 (0.9, 1.1) | **17.53 ± 0.46** (0.9, 1.1) | 14.31 ± 0.67 (0.9, 1.1) | 3.7e-5 | 0.0061 |
| NARMA-10 NRMSE ↓ | 0.484 ± 0.027 (0.5, 1.1) | 0.170 ± 0.022 (0.9, 0.7) | **0.130 ± 0.022** (0.9, 0.9) | 3.7e-5 | 3.7e-5 |
| T0 MSE ↓ | 0.0550 ± 0.0106 (0.1, 1.1) | **0.0158 ± 0.0028** (0.9, 1.1) | 0.0207 ± 0.0042 (0.9, 1.1) | 3.7e-5 | 3.7e-5 |

Validation MC heatmap (mean of 2 tuning seeds; rows = leak, cols = ρ) — also out2/mc_heatmaps.png:
| leak \ ρ | 0.5 | 0.7 | 0.9 | 0.95 | 0.99 | 1.1 |
|---|---|---|---|---|---|---|
| REAL 0.1 / 0.5 / 0.9 | 5.6 / 6.1 / 7.8 | 6.4 / 7.4 / 9.7 | 7.1 / 8.7 / 11.7 | 7.2 / 9.0 / 12.2 | 7.4 / 9.2 / 12.5 | 7.7 / 9.9 / 13.5 |
| ER 0.1 / 0.5 / 0.9 | 6.2 / 9.0 / 13.1 | 4.2 / 8.5 / 15.0 | 0.8 / 4.1 / 16.8 | 0.5 / 3.5 / 17.2 | 0.4 / 3.1 / 17.6 | 0.4 / 1.9 / 18.0 |
| DEG 0.1 / 0.5 / 0.9 | 6.4 / 8.7 / 12.1 | 5.3 / 9.3 / 14.4 | 1.6 / 6.5 / 14.8 | 0.9 / 5.3 / 15.0 | 0.7 / 4.4 / 15.3 | 0.4 / 2.7 / 15.4 |
(full 6×6 grids in out2/sweep.csv)

## Preregistered predictions vs outcome
- H2a (REAL keeps significant MC edge over ER): **wrong** — reversed, ER wins (p=3.7e-5).
- H2b (REAL–DEG gap small): REAL is 5% lower than DEG, and the gap is significant (p=0.006). So "small" was right, but in the opposite direction from what I expected.
- H2c (REAL no better on NARMA/T0): **right**. It's much worse.
- Knockouts: not run, because the preregistered gate failed.
- Structure prediction (more reciprocity and short loops): **right**.

## Structure (one ER and one DEG graph)
| | fly CX | ER | degree-shuffle |
|---|---|---|---|
| reciprocity | 0.405 | 0.036 | 0.066 |
| directed 2-cycles | 60,788 | 5,402 | 9,919 |
| directed 3-cycles | 2,929,802 | 378,565 | 953,636 |
| E/I balance per neuron (mean ± std) | 0.000 ± 0.376 | −0.031 ± 0.201 | −0.039 ± 0.237 |
| fraction of eigenvalues with abs >0.5 (radius scaled to 1) | 0.006 | 0.662 | 0.180 |
| mean abs eigenvalue | 0.037 | 0.602 | 0.275 |
Plot: out2/eigenspectra.png.

## Caveats
- All three wirings picked the corner of the grid (leak 0.9, ρ 1.1) for MC, so the true optima may lie outside the grid. The ER-vs-real ordering at the corner is consistent with the trend, but it isn't proven beyond it.
- Only 2 tuning seeds were used. The real graph is fixed, while the random baselines get a new graph each seed.
- Same caveats as round 1: one rate model, one ridge λ, input scale not tuned, and synapse counts used as weights.

# Round 3 — training the weights on fixed wiring (PREREGISTRATION_3.md)
Run with the compute amendment: 60 Adam updates, 60-step truncated BPTT, full 2,875 neurons, 8 seeds per condition (72/72 jobs). Raw: out3/round3_raw.json, summary out3/round3_summary.json, plot out3/trained_vs_untrained.png. Mann–Whitney n=8 vs 8 (min p = 1.6e-4); Wilcoxon paired n=8 (min p = 0.0078).

| task (metric) | fly CX untrained → trained | ER untrained → trained | deg-shuffle untrained → trained |
|---|---|---|---|
| DELAY (MC over 1..100, ↑) | 8.98 → **8.83** (p=0.25) | 4.12 → 1.94 (p=0.008) | 6.39 → 2.03 (p=0.008) |
| NARMA-10 (NRMSE, ↓) | 0.512 → 0.507 (p=0.008) | 0.470 → 0.478 (p=0.46) | 0.442 → 0.448 (p=0.15) |
| T0 (MSE, ↓) | 0.068 → 0.035 (p=0.008) | 0.037 → 0.00062 (p=0.008) | 0.044 → 0.00037 (p=0.008) |

Fly vs nulls after training: DELAY fly higher than both (p=1.6e-4 each). NARMA fly vs ER p=0.10 (n.s.), vs DEG p=0.010 (fly worse). T0 fly ~60–100× worse (p=1.6e-4).
Important honesty note on DELAY: the fly "win" is NOT training making the fly better. Fly MC barely moved; the null wirings LOST MC because training on lags 1–30 (and BPTT window 40) traded away long-lag memory that the full 1..100 MC counts. Fly was more resistant to that loss.

Robustness after training (worst grid cell ÷ grid mean): DELAY fly 0.73 vs ER 0.036 / DEG 0.048 (higher better); NARMA 1.09 vs 1.68 / 1.19; T0 1.32 vs 4.74 / 3.18 (lower better). Fly more robust on every task (all p=1.6e-4).
Weight movement (rel. L2): fly 0.60 (DELAY) / 0.12 (NARMA) / 0.25 (T0); nulls 0.06–0.15. Spearman init vs trained ≥ 0.94 everywhere.

## Learning-rate check (T0 only; out3/lr_sanity/)
Tuned θ-LR per wiring on seed 100 over {0.001, 0.003, 0.01, 0.03, 0.1}, then 8 seeds at the chosen LR. All three picked 0.03.
- fly 0.00108 vs ER 0.00022 / DEG 0.00029 (MSE; fly vs each p=1.6e-4). At lr 0.01 (main run) fly was 0.035.
- So most of the fly's bad T0 result WAS the optimizer: a higher LR cut fly error ~30×. But fly still ends ~4–5× worse, and only by moving its weights almost completely (rel. L2 0.98 vs ~0.3 for nulls) — i.e. it has to partly un-fly itself to learn T0.
- Worst-grid T0 after LR fix: fly 0.0026 vs ER 0.00085 / DEG 0.0011 (out3/lr_sanity/test.csv).

## Predictions vs outcome
- H3a (training improves every wiring on every task): **wrong.** Training hurt nulls' full MC, didn't significantly move fly MC, and didn't help ER/DEG on NARMA.
- H3b (fly gains most but still ends worse on NARMA/T0): gains-most **wrong** (T0: fly −48% vs nulls −98/−99%); ends-worse **right** for T0, for NARMA only vs DEG (vs ER n.s.).
- H3c (fly keeps better worst-case robustness): **right**, all tasks.
- H3d (weights move <50%, Spearman >0.5): Spearman part **right**; <50% **wrong** for fly DELAY (0.60) and fly T0 at the tuned LR (0.98).

## Caveats
60 updates only (compute amendment); single rate model; DELAY train target (lags 1–30) ≠ eval metric (lags 1–100), which drives the MC result; the LR check covered T0 only.
