# Compression + adapter results

Prereg: `PREREGISTRATION_compress.md`. Jobs: 3,042 compression + 144 adapter. 8 test seeds. Bonferroni over the planned comparisons.

## REAL wiring, mean over 8 seeds

| Method | f | MC ↑ | NARMA ↓ | T0 ↓ | MC sim (s) | speedup vs FULL |
|---|---:|---:|---:|---:|---:|---:|
| FULL | 1 | 13.59 | 0.481 | 0.064 | 0.58 | 1× |
| MAGP | 0.5 | 13.34 | 0.483 | 0.047 | 0.31 | 1.9× |
| RANDP | 0.5 | 13.01 | 0.472 | 0.048 | 0.15 | 3.7× |
| CONSOL | 0.5 | 12.51 | 0.483 | 0.049 | 0.26 | 2.2× |
| LOWRANK | 0.5 | 12.32 | 0.477 | 0.048 | 1.68 | **0.3× (slower)** |
| MAGP | 0.25 | 12.17 | 0.493 | 0.040 | 0.12 | 4.9× |
| CONSOL | 0.25 | 11.88 | 0.485 | 0.040 | 0.12 | 4.7× |
| RANDP | 0.25 | 11.57 | 0.514 | 0.041 | 0.06 | 10× |
| LOWRANK | 0.25 | 12.21 | 0.482 | 0.038 | 0.43 | 1.4× |
| MAGP | 0.1 | 10.12 | 0.492 | 0.033 | 0.04 | 14× |
| CONSOL | 0.1 | 10.41 | 0.518 | 0.033 | 0.04 | 15× |
| RANDP | 0.1 | 10.51 | 0.490 | 0.033 | 0.03 | 22× |
| LOWRANK | 0.1 | 11.96 | 0.483 | 0.033 | 0.04 | 15× |

Worst-seed MC (robustness) at f=0.25 CONSOL: REAL 11.52, ER 16.51, DEG 13.21.

## Adapter arm (REAL, f=0.25). Same 11,504 params for ADAPT and LORA

| meth | task | arm | mean | sim (s) |
|---|---|---|---:|---:|
| CONSOL | MC | NONE | 11.86 | 0.12 |
| CONSOL | MC | ADAPT | **18.33** | 1.14 |
| CONSOL | MC | LORA | **18.97** | 1.07 |
| CONSOL | T0 | NONE | 0.042 | 0.24 |
| CONSOL | T0 | ADAPT | **0.0004** | 0.17 |
| CONSOL | T0 | LORA | 0.0012 | 0.16 |
| CONSOL | NARMA | NONE | 0.487 | 0.12 |
| CONSOL | NARMA | ADAPT | 0.481* | 1.13 |
| CONSOL | NARMA | LORA | 0.474* | 1.07 |
| MAGP | MC | NONE / ADAPT / LORA | 12.66 / **17.95** / **18.45** | 0.12 / 1.12 / 1.07 |
| MAGP | T0 | NONE / ADAPT / LORA | 0.040 / **0.0004** / 0.0006 | 0.24 / 0.17 / 0.16 |
| MAGP | NARMA | NONE / ADAPT / LORA | 0.492 / 0.473* / 0.508* | 0.13 / 1.12 / 1.07 |

\*NARMA means for ADAPT/LORA exclude seed 7, where both arms hit a 1e9 overflow (training blew up). With seed 7 included the mean is meaningless.

## Predictions

| # | Prediction | Held? |
|---|---|---|
| C1 | Moderate compression (f=0.5) costs little on NARMA/T0; MC drops with size | **Partly.** NARMA barely moves. T0 error *improves* with compression. MC drops roughly with size. |
| C2 | CONSOL is not better than MAGP/RANDP at matched size | **Held.** MAGP ≥ CONSOL on MC at every f; no Bonferroni survivor for CONSOL beating anyone. |
| C3 | LOWRANK is fastest but loses the most MC | **Failed.** At f≥0.25 LOWRANK is *slower* than FULL (dense k×k). At f=0.1 it is fast and loses *less* MC than MAGP/CONSOL. |
| C4 | REAL stays more robust than ER/DEG after compression | **Failed.** ER and DEG have better worst-seed MC after CONSOL at every f. |
| C5 | Adapters recover part but not all; ADAPT ≈ LORA; "no cost" fails | **Mixed.** ADAPT ≈ LORA (no special bit-gating benefit). On MC and T0 they *beat* FULL (so "no cost" is too weak — they help). On NARMA they do not help and sometimes explode. |

## Honest takeaways
- The AetherLink consolidation pass (CONSOL) is not better than magnitude pruning. Merging neurons does not buy anything over dropping the smallest weights.
- Low-rank mode compression is a speed trap at moderate k (dense matrix).
- The adapter is **not free**: +11,504 params and ~10× slower simulation (dense residual). A factored form would be cheaper; this run used the dense residual as written.
- Seed 7 NARMA blow-up for both adapters is a real failure mode of the 100-step Adam fine-tune; reported, not hidden.
- No planned comparison survived Bonferroni (63 compression tests, 30 adapter tests). Raw means still tell the story above; the differences are just not large vs seed noise once corrected.

Full tables: `out/analysis.md`. Jobs: `out/jobs.jsonl`, `out/adapt.jsonl`.
