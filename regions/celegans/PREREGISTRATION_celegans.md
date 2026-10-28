# Preregistration: C. elegans individual-variation connectome reservoirs

Written BEFORE any C. elegans task scores. Locked 2026-10-28.

## Motivation

Multiple reconstructed *C. elegans* nervous systems exist across individuals and developmental stages (Witvliet et al. 2021 Nature: 8 hermaphrodites L1→adult; Cook et al. 2019: adult hermaphrodite + male). This tests whether any real-vs-random wiring edge is **reproducible across individuals**, and whether it changes with age. Worms are much simpler (~200–300 neurons) than fly/mouse/human modules; report MC normalized by N.

## Data (must succeed before scoring)

| ID | Source | Stage | Sex | Notes |
|---|---|---|---|---|
| W1–W4 | Witvliet 2021 / WormWiring | L1 | herm | brain/nerve-ring reconstructions |
| W5 | Witvliet 2021 | L2 | herm | |
| W6 | Witvliet 2021 | L3 | herm | |
| W7–W8 | Witvliet 2021 | adult | herm | |
| COOK_H | Cook 2019 SI5 corrected Jul 2020 | adult | herm | whole-animal; neurons only |
| COOK_M | Cook 2019 SI5 corrected Jul 2020 | adult | male | whole-animal; neurons only |

- Chemical synapses: weight = synapse count (Witvliet) or EM-section connectivity (Cook).
- Gap junctions: treated as **symmetric** undirected edges (add both directions). Witvliet gap annotations are incomplete (paper note) — still included; flagged in SOURCES.md.
- E/I sign: **unknown** in these tables → **unsigned** positive weights. Documented; not a signed-E/I claim.
- Artifacts: `data/<worm_id>.npz` with `W` (CSR), `names`, `sign` (all +1), `ei_status="unknown"`, meta.
- Gate: if fewer than 6 Witvliet worms parse with N≥100 and nnz≥200 → STOP and report.

## Reservoir / readout (fixed)

- Leaky tanh rate reservoir, leak=0.2, global spectral radius ρ=0.9 (fast_kernels radius cache).
- Ridge readout; **α tuned per arm on seeds 90–92 only** (grid `{1e-8,1e-6,1e-4,1e-2,1e-1,1.0,10.0}`), criterion = mean MC at full train. Locked α then used for eval seeds 0–7. (Lesson from fewshot_delay: fixed α created a fake win.)
- Tasks: MC ↑ (lags 1–100), NARMA-10 NRMSE ↓, PC_HARD ↑ (cue_frac=0.35, noise=1.5).
- Also report **MC/N**.
- Optional FIXED delays: d ∈ {1..5} drawn once per edge (same as regions/myelin FIXED / best_fly). Arms with `_DEL` vs `_NOD`.

## Arms (per worm)

Wiring: `REAL`, `ER`, `SHUF` (degree-shuffle), `DENS` (density+magnitude matched).
Delay: `NOD`, `DEL`.
Full arm name: `{WORM}_{WIRING}_{DELAY}` e.g. `W7_REAL_DEL`.

Controls are always **same N and comparable nnz** as that worm's REAL graph.

## Chemotaxis bonus (Q3)

Native-ish task: 1-D concentration signal `c(t)` (smooth OU / slow sinusoid + noise) drives **real sensory** neurons present in that worm among `{ASEL,ASER,AWCL,AWCR,ASGL,ASGR}` (use whichever exist; require ≥2). Readout ridge from **command/interneurons** present among `{AVAL,AVAR,AVBL,AVBR,AIYL,AIYR,AIZL,AIZR}`. Target = `c(t)` delayed by τ=5 (proxies "remember / act on recent odor"). Metric: test R² ↑.

Controls: same number of **random** input neurons and **random** readout neurons (matched counts), same W. Arms: `CHEMO_REAL_IO` vs `CHEMO_RAND_IO`, crossed with NOD/DEL on REAL wiring only (exploratory; not in primary Bonferroni).

## Questions / predictions (locked)

**Q1.** For each worm, does REAL beat its own ER/SHUF/DENS on MC, NARMA, PC_HARD (± FIXED delays)?
- Per-worm contrasts: Mann–Whitney U two-sided, seeds 0–7.
- Primary family: for each task × delay mode, REAL vs each of {ER,SHUF,DENS} → 3×3×2 = 18 contrasts **per worm**; Bonferroni within worm.
- Cross-worm consistency: for each (task, delay, control), record effect sign of median(REAL)−median(CTRL) per worm; report fraction of worms with REAL better; one-sample Wilcoxon / binomial test on per-worm effect sizes (worms as replicates). **Verdict "consistent"** only if ≥6/8 Witvliet worms share the same sign AND meta p<sub>Bonf</sub><0.05 after correcting over the 18 meta-contrasts.

**Q2.** Development: among Witvliet W1–W8, does the REAL−CTRL gap change with age stage (L1=1,L2=2,L3=3,adult=4)? Spearman ρ of effect size vs stage; exploratory (report raw p).

**Q3.** Chemotaxis bonus: CHEMO_REAL_IO > CHEMO_RAND_IO on R² (direction predicted). Bonferroni over worms × delay (exploratory family separate from Q1).

## Honesty rules

- Report negatives. Do not claim biology wins from REAL>NODELAY alone — controls are ER/SHUF/DENS.
- Do not use eval seeds for α tuning.
- Gap-junction incompleteness in Witvliet is a limitation, not hidden.
- Unsigned weights: no E/I biological claim.
- Do not push to GitHub from this run.
- Keep WORKERS≤2; be nice to other jobs.

## Outputs

- `PREREGISTRATION_celegans.md` (this file), `SOURCES.md`
- `out/runs.jsonl`, `out/alpha_choice.json`, `out/celegans.done`
- `RESULTS_celegans.md` + figure `figs/celegans_effect_by_worm.png`
