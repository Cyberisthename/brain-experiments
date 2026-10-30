# PREREG: human_wholebrain (freeze before seeds 2200–2207)

Date: 2026-10-30 (America/New_York). Frozen before any held-out eval seed is scored.

## Question
Does a recurrent net whose wiring comes from a real human structural connectome
(Lausanne scale-68 group / individuals from Griffa et al. 2019) beat matched
random controls — and does adding myelin/length-based delays help — on tasks that
need memory and cross-hemisphere binding?

## Data (local only; never pushed)
- Group SC + subject FC mean signs + consensus fiber lengths:
  `connectomes/human/griffa70/griffa70_scale68_whole.npz`
- Topology prior: consensus binary edges (fiber_length > 0), dens ≈ 0.28.
- Weights: group-average SC on those edges; individuals use subject SC>0.
- Signs: sign of nanmean FC across 70 subjects (Dale 80/20 fallback if NaN).

## Network
- 68 parcels × `N_LOCAL=2` units → `N=136`.
- Expand each SC edge (i,j) to a full 2×2 unit block (no self loops).
- Within-parcel all-to-all on `mask_intra` (instant).
- Cross-parcel edges: `mask_intra` if same hemisphere and delay==1; else `mask_inter` with delay.
- Left-only readout. Split hemispheric inputs for XOR tasks.
- OscRNN (leak, Adam, softplus×sign on delayed edges) — same family as realistic_callosum.

## Arms (match nonzero edge count / trainable recurrent params to HUMAN_SC)
1. **HUMAN_SC** — consensus topology + group SC weights + FC signs; fixed delay=1 on all delayed edges.
2. **HUMAN_SC_MYELIN** — same topology/signs; delays from fiber length (bin 1..DMAX).
3. **DEGREE_PRESERVING_SHUFFLE** — Maslov–Sneppen rewire of region graph (preserve degree), same weight/sign pool, expand.
4. **WEIGHT_SHUFFLE** — identical unit mask; permute nonzero weights/signs among existing edges.
5. **ER_RANDOM** — ER region graph matched density; same weight/sign pool; expand.
6. **MYELIN_ONLY** — two-hemisphere modular ER (matched within-hemi dens) + length-like delay schedule + sparse random callosal bridge matched to real callosal edge count.
7. **INDIVIDUAL** — subject `seed % 70` SC/FC (presence SC>0); report spread, not a primary win claim.

## Tasks (harder memory to avoid XOR ceiling)
| Task | Metric | Better |
|---|---|---|
| XOR_HARD | accuracy | higher (d=12 hemispheric XOR, left readout) |
| XOR2BIT | mean bit accuracy | higher (2 independent delayed XORs, d=8, 4 in / 2 out) |
| CHAR | CE on Alice | lower |
| NARMA | NRMSE | lower |

## Seeds / hparams
- Held-out eval seeds: **2200–2207** (8). Never used for tuning.
- Frozen hparams = realistic_callosum / oscillation_gating DEFAULT_HP
  (`lr=5e-3, leak=0.4, epochs=45, steps_per_epoch=20, …`), written to `out/hparams.json` before eval.
- Optional smoke / wiring-check only on seed 90 (not counted).

## Stats
- Paired Wilcoxon signed-rank across seeds, per task.
- Primary contrasts (Bonferroni over the set below):
  1. HUMAN_SC_MYELIN vs DEGREE_PRESERVING_SHUFFLE on XOR_HARD, NARMA
  2. HUMAN_SC vs ER_RANDOM on XOR_HARD
  3. HUMAN_SC vs WEIGHT_SHUFFLE on XOR_HARD
  4. HUMAN_SC_MYELIN vs MYELIN_ONLY on XOR_HARD
- α_family = 0.05; m = number of primary tests (6); α = 0.05/m.
- Secondary: means tables; INDIVIDUAL mean±std; CHAR exploratory.

## Verdict rules (preregistered)
- **SUPPORTED** for a contrast if Wilcoxon p < α and direction matches “human/myelin better”.
- **NOT SUPPORTED** otherwise.
- Do not peek at seeds 2200–2207 before this file is committed.

## Runtime
- ~7 arms × 4 tasks × 8 seeds = 224 jobs; 2 workers; target wall ~2–3 h; nohup + resume from `out/runs.jsonl`.
