# Results: C. elegans individual-variation connectomes

Prereg: `PREREGISTRATION_celegans.md`. Sources: `SOURCES.md`.

## Worm graphs

| Worm | Stage | Sex | N | nnz | source |
|---|---|---|---:|---:|---|
| W1 | L1 | herm | 165 | 835 | Witvliet2021 |
| W2 | L1 | herm | 166 | 1087 | Witvliet2021 |
| W3 | L1 | herm | 166 | 1068 | Witvliet2021 |
| W4 | L1 | herm | 172 | 1374 | Witvliet2021 |
| W5 | L2 | herm | 178 | 1825 | Witvliet2021 |
| W6 | L3 | herm | 179 | 1694 | Witvliet2021 |
| W7 | adult | herm | 185 | 2387 | Witvliet2021 |
| W8 | adult | herm | 184 | 2402 | Witvliet2021 |
| COOK_H | adult | herm | 300 | 5089 | Cook2019 |
| COOK_M | adult | male | 398 | 5757 | Cook2019 |

E/I: unknown → unsigned weights. Witvliet gap junctions incomplete (paper note).

## Q1 — Per-worm REAL vs controls (median over 8 seeds)

MC ↑, NARMA ↓, PC_HARD ↑. Effect = signed so **positive ⇒ REAL better**.


### Summary table (Witvliet + Cook)

| Worm | Stage | Delay | MC_REAL | MC_ER | MC_effect | NARMA_eff(vs ER) | PC_eff(vs ER) |
|---|---|---|---:|---:|---:|---:|---:|
| W1 | L1 | NOD | 5.33 | 6.20 | -0.86 | -0.001 | +0.002 |
| W1 | L1 | DEL | 13.69 | 14.94 | -1.25 | +0.017 | +0.021 |
| W2 | L1 | NOD | 5.06 | 5.67 | -0.61 | -0.016 | -0.030 |
| W2 | L1 | DEL | 13.43 | 14.53 | -1.10 | +0.021 | -0.040 |
| W3 | L1 | NOD | 4.89 | 5.82 | -0.93 | -0.019 | +0.014 |
| W3 | L1 | DEL | 13.46 | 14.43 | -0.97 | +0.030 | -0.028 |
| W4 | L1 | NOD | 4.67 | 5.76 | -1.09 | -0.087 | +0.017 |
| W4 | L1 | DEL | 12.94 | 14.63 | -1.69 | -0.097 | -0.024 |
| W5 | L2 | NOD | 4.70 | 5.55 | -0.85 | -0.069 | -0.019 |
| W5 | L2 | DEL | 13.20 | 14.33 | -1.13 | -0.076 | -0.020 |
| W6 | L3 | NOD | 4.68 | 5.61 | -0.93 | -0.085 | +0.034 |
| W6 | L3 | DEL | 13.31 | 14.64 | -1.33 | -0.086 | +0.022 |
| W7 | adult | NOD | 4.60 | 5.41 | -0.81 | -0.115 | +0.019 |
| W7 | adult | DEL | 13.31 | 14.51 | -1.20 | -0.056 | +0.020 |
| W8 | adult | NOD | 4.62 | 5.30 | -0.69 | -0.058 | -0.005 |
| W8 | adult | DEL | 12.72 | 14.24 | -1.52 | -0.054 | -0.001 |
| COOK_H | adult | NOD | 5.00 | 5.22 | -0.21 | -0.021 | +0.051 |
| COOK_H | adult | DEL | 13.82 | 14.75 | -0.93 | -0.042 | +0.068 |
| COOK_M | adult | NOD | 4.61 | 5.36 | -0.75 | -0.085 | -0.050 |
| COOK_M | adult | DEL | 12.61 | 15.08 | -2.47 | -0.108 | -0.013 |

### Per-worm detail

### W1 (L1, N=165)

| Task | Delay | REAL | ER | SHUF | DENS | REAL−bestCTRL effect |
|---|---|---:|---:|---:|---:|---:|
| MC | NOD | 5.330 | 6.195 | 5.480 | 5.931 | -0.538 |
| MC/N | NOD | 0.0323 |  |  |  |  |
| MC | DEL | 13.692 | 14.945 | 14.130 | 14.744 | -0.914 |
| MC/N | DEL | 0.0830 |  |  |  |  |
| NARMA | NOD | 0.584 | 0.583 | 0.584 | 0.593 | +0.003 |
| NARMA | DEL | 0.440 | 0.458 | 0.402 | 0.454 | -0.002 |
| PC_HARD | NOD | 0.841 | 0.839 | 0.822 | 0.840 | +0.007 |
| PC_HARD | DEL | 0.891 | 0.871 | 0.874 | 0.879 | +0.017 |
### W2 (L1, N=166)

| Task | Delay | REAL | ER | SHUF | DENS | REAL−bestCTRL effect |
|---|---|---:|---:|---:|---:|---:|
| MC | NOD | 5.055 | 5.668 | 5.431 | 5.875 | -0.603 |
| MC/N | NOD | 0.0305 |  |  |  |  |
| MC | DEL | 13.426 | 14.527 | 14.506 | 14.539 | -1.098 |
| MC/N | DEL | 0.0809 |  |  |  |  |
| NARMA | NOD | 0.628 | 0.612 | 0.597 | 0.566 | -0.037 |
| NARMA | DEL | 0.444 | 0.465 | 0.418 | 0.369 | -0.027 |
| PC_HARD | NOD | 0.813 | 0.843 | 0.821 | 0.764 | +0.004 |
| PC_HARD | DEL | 0.843 | 0.883 | 0.830 | 0.859 | -0.015 |
### W3 (L1, N=166)

| Task | Delay | REAL | ER | SHUF | DENS | REAL−bestCTRL effect |
|---|---|---:|---:|---:|---:|---:|
| MC | NOD | 4.889 | 5.819 | 5.499 | 5.922 | -0.858 |
| MC/N | NOD | 0.0295 |  |  |  |  |
| MC | DEL | 13.460 | 14.434 | 14.606 | 14.487 | -1.049 |
| MC/N | DEL | 0.0811 |  |  |  |  |
| NARMA | NOD | 0.634 | 0.614 | 0.577 | 0.564 | -0.049 |
| NARMA | DEL | 0.442 | 0.472 | 0.408 | 0.385 | -0.021 |
| PC_HARD | NOD | 0.812 | 0.798 | 0.786 | 0.814 | +0.013 |
| PC_HARD | DEL | 0.800 | 0.828 | 0.807 | 0.848 | -0.028 |
### W4 (L1, N=172)

| Task | Delay | REAL | ER | SHUF | DENS | REAL−bestCTRL effect |
|---|---|---:|---:|---:|---:|---:|
| MC | NOD | 4.671 | 5.759 | 5.309 | 5.726 | -0.927 |
| MC/N | NOD | 0.0272 |  |  |  |  |
| MC | DEL | 12.942 | 14.629 | 14.163 | 14.669 | -1.544 |
| MC/N | DEL | 0.0752 |  |  |  |  |
| NARMA | NOD | 0.662 | 0.574 | 0.581 | 0.618 | -0.070 |
| NARMA | DEL | 0.468 | 0.371 | 0.424 | 0.473 | -0.046 |
| PC_HARD | NOD | 0.823 | 0.806 | 0.845 | 0.880 | -0.020 |
| PC_HARD | DEL | 0.854 | 0.878 | 0.850 | 0.874 | -0.014 |
### W5 (L2, N=178)

| Task | Delay | REAL | ER | SHUF | DENS | REAL−bestCTRL effect |
|---|---|---:|---:|---:|---:|---:|
| MC | NOD | 4.704 | 5.551 | 5.317 | 5.788 | -0.848 |
| MC/N | NOD | 0.0264 |  |  |  |  |
| MC | DEL | 13.205 | 14.333 | 14.159 | 14.724 | -1.201 |
| MC/N | DEL | 0.0742 |  |  |  |  |
| NARMA | NOD | 0.650 | 0.580 | 0.595 | 0.574 | -0.067 |
| NARMA | DEL | 0.466 | 0.391 | 0.418 | 0.386 | -0.068 |
| PC_HARD | NOD | 0.830 | 0.849 | 0.827 | 0.828 | -0.005 |
| PC_HARD | DEL | 0.853 | 0.872 | 0.834 | 0.844 | +0.003 |
### W6 (L3, N=179)

| Task | Delay | REAL | ER | SHUF | DENS | REAL−bestCTRL effect |
|---|---|---:|---:|---:|---:|---:|
| MC | NOD | 4.681 | 5.614 | 5.785 | 5.662 | -1.006 |
| MC/N | NOD | 0.0262 |  |  |  |  |
| MC | DEL | 13.310 | 14.644 | 14.535 | 14.238 | -1.163 |
| MC/N | DEL | 0.0744 |  |  |  |  |
| NARMA | NOD | 0.664 | 0.579 | 0.581 | 0.624 | -0.069 |
| NARMA | DEL | 0.473 | 0.387 | 0.411 | 0.472 | -0.050 |
| PC_HARD | NOD | 0.844 | 0.811 | 0.843 | 0.844 | +0.012 |
| PC_HARD | DEL | 0.850 | 0.828 | 0.852 | 0.864 | +0.002 |
### W7 (adult, N=185)

| Task | Delay | REAL | ER | SHUF | DENS | REAL−bestCTRL effect |
|---|---|---:|---:|---:|---:|---:|
| MC | NOD | 4.600 | 5.409 | 5.303 | 5.665 | -0.859 |
| MC/N | NOD | 0.0249 |  |  |  |  |
| MC | DEL | 13.307 | 14.506 | 14.146 | 14.862 | -1.198 |
| MC/N | DEL | 0.0719 |  |  |  |  |
| NARMA | NOD | 0.704 | 0.589 | 0.587 | 0.571 | -0.122 |
| NARMA | DEL | 0.455 | 0.399 | 0.429 | 0.376 | -0.053 |
| PC_HARD | NOD | 0.829 | 0.810 | 0.812 | 0.834 | +0.010 |
| PC_HARD | DEL | 0.849 | 0.829 | 0.794 | 0.864 | +0.020 |
### W8 (adult, N=184)

| Task | Delay | REAL | ER | SHUF | DENS | REAL−bestCTRL effect |
|---|---|---:|---:|---:|---:|---:|
| MC | NOD | 4.617 | 5.303 | 5.460 | 5.632 | -0.848 |
| MC/N | NOD | 0.0251 |  |  |  |  |
| MC | DEL | 12.720 | 14.245 | 14.544 | 14.707 | -1.779 |
| MC/N | DEL | 0.0691 |  |  |  |  |
| NARMA | NOD | 0.653 | 0.595 | 0.584 | 0.565 | -0.071 |
| NARMA | DEL | 0.465 | 0.411 | 0.412 | 0.388 | -0.061 |
| PC_HARD | NOD | 0.809 | 0.814 | 0.872 | 0.818 | -0.026 |
| PC_HARD | DEL | 0.855 | 0.856 | 0.886 | 0.865 | -0.014 |
### COOK_H (adult, N=300)

| Task | Delay | REAL | ER | SHUF | DENS | REAL−bestCTRL effect |
|---|---|---:|---:|---:|---:|---:|
| MC | NOD | 5.005 | 5.217 | 5.026 | 5.598 | -0.276 |
| MC/N | NOD | 0.0167 |  |  |  |  |
| MC | DEL | 13.821 | 14.753 | 14.279 | 14.933 | -0.834 |
| MC/N | DEL | 0.0461 |  |  |  |  |
| NARMA | NOD | 0.622 | 0.602 | 0.588 | 0.568 | -0.037 |
| NARMA | DEL | 0.452 | 0.410 | 0.419 | 0.380 | -0.049 |
| PC_HARD | NOD | 0.877 | 0.825 | 0.814 | 0.830 | +0.054 |
| PC_HARD | DEL | 0.887 | 0.820 | 0.843 | 0.845 | +0.051 |
### COOK_M (adult, N=398)

| Task | Delay | REAL | ER | SHUF | DENS | REAL−bestCTRL effect |
|---|---|---:|---:|---:|---:|---:|
| MC | NOD | 4.609 | 5.363 | 5.310 | 5.744 | -0.863 |
| MC/N | NOD | 0.0116 |  |  |  |  |
| MC | DEL | 12.606 | 15.075 | 14.583 | 15.031 | -2.290 |
| MC/N | DEL | 0.0317 |  |  |  |  |
| NARMA | NOD | 0.670 | 0.585 | 0.587 | 0.581 | -0.086 |
| NARMA | DEL | 0.495 | 0.387 | 0.402 | 0.449 | -0.082 |
| PC_HARD | NOD | 0.800 | 0.850 | 0.833 | 0.819 | -0.034 |
| PC_HARD | DEL | 0.835 | 0.848 | 0.860 | 0.824 | -0.009 |

## Q1 meta — cross-worm consistency (Witvliet W1–W8 as replicates)

Bonferroni over 18 primary meta-contrasts (3 tasks × 2 delays × 3 controls).

| Task | Delay | Ctrl | n_worms | n_REAL_better | mean effect | Wilcoxon p | p_Bonf | consistent? |
|---|---|---|---:|---:|---:|---:|---:|---|
| MC | NOD | ER | 8 | 0 | -0.846 | 0.007812 | 0.1406 | no |
| MC | NOD | SHUF | 8 | 0 | -0.630 | 0.007812 | 0.1406 | no |
| MC | NOD | DENS | 8 | 0 | -0.957 | 0.007812 | 0.1406 | no |
| MC | DEL | ER | 8 | 0 | -1.275 | 0.007812 | 0.1406 | no |
| MC | DEL | SHUF | 8 | 0 | -1.091 | 0.007812 | 0.1406 | no |
| MC | DEL | DENS | 8 | 0 | -1.364 | 0.007812 | 0.1406 | no |
| NARMA | NOD | ER | 8 | 0 | -0.056 | 0.007812 | 0.1406 | no |
| NARMA | NOD | SHUF | 8 | 0 | -0.062 | 0.007812 | 0.1406 | no |
| NARMA | NOD | DENS | 8 | 1 | -0.063 | 0.01562 | 0.2812 | no |
| NARMA | DEL | ER | 8 | 3 | -0.038 | 0.1094 | 1 | no |
| NARMA | DEL | SHUF | 8 | 0 | -0.042 | 0.007812 | 0.1406 | no |
| NARMA | DEL | DENS | 8 | 2 | -0.044 | 0.07812 | 1 | no |
| PC_HARD | NOD | ER | 8 | 5 | +0.004 | 0.7422 | 1 | no |
| PC_HARD | NOD | SHUF | 8 | 5 | -0.003 | 0.9453 | 1 | no |
| PC_HARD | NOD | DENS | 8 | 4 | -0.003 | 0.5469 | 1 | no |
| PC_HARD | DEL | ER | 8 | 3 | -0.006 | 0.4609 | 1 | no |
| PC_HARD | DEL | SHUF | 8 | 5 | +0.008 | 0.3828 | 1 | no |
| PC_HARD | DEL | DENS | 8 | 2 | -0.013 | 0.05469 | 0.9844 | no |

**Consistency verdict:** 0/18 primary meta-contrasts meet the prereg bar (≥6/8 same sign + p_Bonf<0.05).
No wiring-specific edge is reproducible across individuals under this bar.

## Q2 — Development (Witvliet effect vs stage)

| Task | Delay | Ctrl | Spearman ρ (effect vs stage) | p |
|---|---|---|---:|---:|
| MC | NOD | ER | +0.319 | 0.4408 |
| MC | NOD | SHUF | -0.792 | 0.0192 |
| MC | NOD | DENS | -0.345 | 0.4029 |
| MC | DEL | ER | -0.294 | 0.4801 |
| MC | DEL | SHUF | -0.255 | 0.5415 |
| MC | DEL | DENS | -0.358 | 0.3845 |
| NARMA | NOD | ER | -0.549 | 0.1586 |
| NARMA | NOD | SHUF | -0.639 | 0.08839 |
| NARMA | NOD | DENS | -0.639 | 0.08839 |
| NARMA | DEL | ER | -0.319 | 0.4408 |
| NARMA | DEL | SHUF | -0.281 | 0.5003 |
| NARMA | DEL | DENS | -0.600 | 0.1157 |
| PC_HARD | NOD | ER | +0.281 | 0.5003 |
| PC_HARD | NOD | SHUF | -0.268 | 0.5208 |
| PC_HARD | NOD | DENS | -0.306 | 0.4603 |
| PC_HARD | DEL | ER | +0.536 | 0.1706 |
| PC_HARD | DEL | SHUF | +0.026 | 0.9521 |
| PC_HARD | DEL | DENS | +0.332 | 0.4217 |

Exploratory (raw p). No claim without a clear monotonic pattern.

## Q3 — Chemotaxis-like bonus

| Worm | Delay | REAL_IO R² | RAND_IO R² | Δ | p_raw |
|---|---|---:|---:|---:|---:|
| W1 | NOD | 0.997 | 0.996 | +0.001 | 0.1605 |
| W1 | DEL | 0.998 | 0.996 | +0.002 | 0.03792 |
| W2 | NOD | 0.997 | 0.997 | +0.000 | 0.1949 |
| W2 | DEL | 0.997 | 0.997 | +0.000 | 0.6454 |
| W3 | NOD | 0.997 | 0.993 | +0.004 | 0.01041 |
| W3 | DEL | 0.998 | 0.985 | +0.013 | 0.001088 |
| W4 | NOD | 0.997 | 0.996 | +0.001 | 0.02813 |
| W4 | DEL | 0.998 | 0.994 | +0.004 | 0.0001554 |
| W5 | NOD | 0.997 | 0.996 | +0.001 | 0.03792 |
| W5 | DEL | 0.998 | 0.996 | +0.001 | 0.04988 |
| W6 | NOD | 0.997 | 0.996 | +0.001 | 0.1605 |
| W6 | DEL | 0.998 | 0.995 | +0.003 | 0.0001554 |
| W7 | NOD | 0.997 | 0.995 | +0.002 | 0.03792 |
| W7 | DEL | 0.998 | 0.996 | +0.001 | 0.02813 |
| W8 | NOD | 0.997 | 0.996 | +0.001 | 0.1304 |
| W8 | DEL | 0.998 | 0.995 | +0.003 | 0.01476 |
| COOK_H | NOD | 0.997 | 0.996 | +0.000 | 0.1949 |
| COOK_H | DEL | 0.997 | 0.996 | +0.001 | 0.1605 |
| COOK_M | NOD | 0.996 | 0.994 | +0.002 | 0.006993 |
| COOK_M | DEL | 0.997 | 0.993 | +0.004 | 0.02067 |

Exploratory: 13/20 worm×delay cells with REAL_IO>RAND_IO and raw p<0.05.

## Honest takeaway

**Question asked:** does *C. elegans* real wiring beat fair random controls **consistently across individuals**?

**Answer (prereg bar):** No. 0/18 primary meta-contrasts meet ≥6/8 same-sign REAL-better + p_Bonf<0.05.

**What is consistent, honestly:**
- On **MC**, REAL loses to ER/SHUF/DENS in **8/8** Witvliet worms (NOD and DEL). Mean effect ≈ −0.85 (NOD) to −1.3 (DEL) MC units. Direction is reproducible; the *advantage* of real wiring is not. Raw Wilcoxon p≈0.0078, but after Bonferroni×18 that is p_Bonf≈0.14 — so we do **not** claim a significant REAL deficit either under the prereg family.
- On **NARMA** and **PC_HARD**, effects flip sign across worms; no reproducible wiring edge.
- **Delays** roughly triple MC for every arm (REAL and controls), matching the myelin/delay lesson elsewhere in the project.
- **Development (Q2):** no clear monotonic REAL−ER gap change L1→adult after looking at Spearman ρ (one raw p<0.05 for MC NOD vs SHUF, exploratory only).
- **Chemotaxis bonus (Q3):** task is **ceilinged** (R²≈0.99 for REAL_IO and RAND_IO). Tiny positive Δs with some raw p<0.05 are not interpretable as a native-task win; need a harder sensory→motor probe before claiming anything.

**Limits:** N≈165–300; E/I unsigned; Witvliet gap junctions incomplete; Cook male graph is SI5-parsed (heuristic muscle filter). MC/N reported because absolute MC is small.

**Bottom line for the parent project:** individual *C. elegans* connectomes reproduce the same pattern we see in fly/mouse/human patches — real wiring does not beat density-matched random on generic reservoir tasks, and that negative result is stable across 8 developmental individuals.
