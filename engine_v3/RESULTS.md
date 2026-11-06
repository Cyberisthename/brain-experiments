# Engine v3 results

**Time zone:** America/New_York.

## Validation (features off)

| Check | Result |
|---|---|
| Unit tests (`pytest engine_v3/tests`) | **9/9 passed** |
| Callosum forward vs v2 sparse OscRNN | **PASS** — max abs err **0.0** |
| Wholebrain dense→sparse port forward | **PASS** — max abs err **~1.5e-7** |
| Synced train (identical init + batches) | **PASS** — loss/out diff **0.0** |
| Mini metric XOR_HARD (short hp, 2 seeds) | within ~0.014 (tol diagnostic) |
| Mini metric NARMA (short hp) | **not comparable** — different train seed offset than v2 `seed+17*n_label`; dynamics proven by synced train |

**Verdict: validation PASS** for migrating science onto v3 with flags off.

## Speed

See `BENCH.md`. Forward speedup vs v2 callosum OscRNN ≈ **1.02–1.06×** at N=512/1024 under load (ring buffer only). Synthetic N=2048 forward ~51–61 ms (B=16,T=48). **Below** the 1.5× end-to-end gate — Phase-2 batching/compile/GPU still required before flipping default region scripts.

## Feature ablation (COMPLETE)

Prereg: `PREREG_v3_features.md` (written before runs).
Job finished 2026-11-05 ~1:00 PM ET. Analyzed immediately via waiter → `out/ablation_summary.json`.
Grid: 6 features × 4 tasks × 8 seeds = **192/192** rows. Topology: callosum_scaled REAL_SC, N_label=512.
Callosum eval PID 362279 left untouched throughout.

### Means ± std (effect vs BASE is signed so + = better)

| Feature | XOR_HARD ↑ | NARMA ↓ | CHAR ↓ | TRANSFER ↑ |
|---|---:|---:|---:|---:|
| BASE | 0.5177±0.0144 | 1.9587±1.0173 | 2.2754±0.0451 | 0.7168±0.0397 |
| STP | 0.7196±0.0105 | 0.5968±0.0287 | 2.0397±0.0334 | 0.9854±0.0220 |
| DUAL_TAU | 0.5126±0.0127 | 1.7362±0.8247 | 2.3258±0.0413 | 0.6696±0.0282 |
| DALE | 0.5071±0.0060 | 1.0554±0.0358 | 2.2573±0.0784 | 0.5202±0.0056 |
| HOMEO | 0.5032±0.0049 | 1.8470±1.1074 | 2.3166±0.0790 | 0.4962±0.0037 |
| ALL | 0.5005±0.0069 | 1.2377±0.2174 | 2.1491±0.0467 | 0.5041±0.0106 |

### Paired Wilcoxon vs BASE (n=8 seeds 3300–3307)

| Feature | Task | effect | p | survives α=0.0025? |
|---|---|---:|---:|---|
| STP | XOR_HARD | +0.2019 | 0.007812 | no |
| STP | NARMA | +1.3619 | 0.007812 | no |
| STP | CHAR | +0.2357 | 0.007812 | no |
| STP | TRANSFER | +0.2686 | 0.007812 | no |
| DUAL_TAU | XOR_HARD | -0.0051 | 0.3828 | no |
| DUAL_TAU | NARMA | +0.2225 | 0.9453 | no |
| DUAL_TAU | CHAR | -0.0504 | 0.007812 | no |
| DUAL_TAU | TRANSFER | -0.0472 | 0.03906 | no |
| DALE | XOR_HARD | -0.0105 | 0.1953 | no |
| DALE | NARMA | +0.9033 | 0.03906 | no |
| DALE | CHAR | +0.0181 | 0.6406 | no |
| DALE | TRANSFER | -0.1967 | 0.007812 | no |
| HOMEO | XOR_HARD | -0.0145 | 0.007812 | no |
| HOMEO | NARMA | +0.1117 | 0.4609 | no |
| HOMEO | CHAR | -0.0412 | 0.1094 | no |
| HOMEO | TRANSFER | -0.2207 | 0.007812 | no |
| ALL | XOR_HARD | -0.0172 | 0.007812 | no |
| ALL | NARMA | +0.7210 | 0.1094 | no |
| ALL | CHAR | +0.1263 | 0.007812 | no |
| ALL | TRANSFER | -0.2127 | 0.007812 | no |

### Interpretation (exploratory)

- With n=8, two-sided Wilcoxon floor is p≈0.0078 when all seeds agree, so the prereg confirmatory bar α=0.0025 is unreachable. Treat all hits as exploratory.
- **STP** is the only arm with positive effect on every task (CHAR +0.24, NARMA +1.36, TRANSFER +0.27, XOR_HARD +0.20), each at the Wilcoxon floor. Matches the earlier early-signal note.
- **DALE** helps NARMA (+0.90, p≈0.039) and hurts TRANSFER (−0.20, p≈0.0078).
- **HOMEO** hurts TRANSFER (effect −0.22) and XOR_HARD (−0.0145); no clear win vs BASE.
- **DUAL_TAU** hurts CHAR (−0.050).
- **ALL** (combined) is mixed: helps CHAR/NARMA, hurts TRANSFER/XOR vs BASE — STP alone is cleaner.
- Biology win claims still need a later held-out-seed prereg (per PREREG).

### Next (approved)

Hypothalamus motif tests A/B/C in `hypothalamus/RESEARCH.md` were approved to run after this ablation. Prereg before coding; do not stop callosum_scaled.

## Anything broken

- End-to-end speedup gate (≥1.5×) **not met** yet (expected; Phase-1 ring only).
- Standalone metric-parity helper used a different seed than callosum `train_task` for NARMA; not a dynamics bug.
- `torch.compile` not enabled by default (optional flag).
- Confirmatory Bonferroni unreachable at n=8 Wilcoxon; document as exploratory.
