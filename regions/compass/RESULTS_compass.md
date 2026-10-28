# Compass (fly CX path integration) results

Prereg: `PREREGISTRATION_compass.md`.

Metric: circular RMSE in radians (↓ better).

## Means ± sd (n seeds)

| Arm | HEADING | DRIFT | GEN |
|---|---:|---:|---:|
| `FLY_CX_NOD` | 1.200 ± 0.176 (n=8) | 1.667 ± 0.312 (n=8) | 1.244 ± 0.152 (n=8) |
| `FLY_CX_DEL` | 1.244 ± 0.186 (n=8) | 1.663 ± 0.313 (n=8) | 1.253 ± 0.135 (n=8) |
| `ER_NOD` | 1.178 ± 0.108 (n=8) | 1.617 ± 0.318 (n=8) | 1.252 ± 0.110 (n=8) |
| `ER_DEL` | 1.174 ± 0.097 (n=8) | 1.608 ± 0.312 (n=8) | 1.269 ± 0.107 (n=8) |
| `SHUF_NOD` | 1.176 ± 0.150 (n=8) | 1.599 ± 0.319 (n=8) | 1.232 ± 0.124 (n=8) |
| `SHUF_DEL` | 1.190 ± 0.169 (n=8) | 1.588 ± 0.313 (n=8) | 1.217 ± 0.114 (n=8) |
| `DENS_NOD` | 1.348 ± 0.137 (n=8) | 1.592 ± 0.268 (n=8) | 1.213 ± 0.113 (n=8) |
| `DENS_DEL` | 1.314 ± 0.148 (n=8) | 1.592 ± 0.296 (n=8) | 1.228 ± 0.094 (n=8) |

## Primary family: FLY_CX_DEL vs delay-matched controls (Bonferroni ×9)

| Metric | Control | Δ (FLY−ctrl) | raw p | bonf p | FLY better? |
|---|---|---:|---:|---:|---|
| HEADING | `ER_DEL` | +0.0697 | 0.1953 | 1 | no |
| HEADING | `SHUF_DEL` | +0.0540 | 0.1953 | 1 | no |
| HEADING | `DENS_DEL` | -0.0703 | 0.25 | 1 | yes |
| DRIFT | `ER_DEL` | +0.0550 | 0.05469 | 0.4922 | no |
| DRIFT | `SHUF_DEL` | +0.0756 | 0.05469 | 0.4922 | no |
| DRIFT | `DENS_DEL` | +0.0717 | 0.1484 | 1 | no |
| GEN | `ER_DEL` | -0.0153 | 0.8438 | 1 | yes |
| GEN | `SHUF_DEL` | +0.0362 | 0.25 | 1 | no |
| GEN | `DENS_DEL` | +0.0253 | 0.7422 | 1 | no |

## Secondary family: FLY_CX_NOD vs no-delay controls (Bonferroni ×9)

| Metric | Control | Δ (FLY−ctrl) | raw p | bonf p | FLY better? |
|---|---|---:|---:|---:|---|
| HEADING | `ER_NOD` | +0.0221 | 0.6406 | 1 | no |
| HEADING | `SHUF_NOD` | +0.0240 | 0.6406 | 1 | no |
| HEADING | `DENS_NOD` | -0.1485 | 0.07812 | 0.7031 | yes |
| DRIFT | `ER_NOD` | +0.0506 | 0.05469 | 0.4922 | no |
| DRIFT | `SHUF_NOD` | +0.0684 | 0.07812 | 0.7031 | no |
| DRIFT | `DENS_NOD` | +0.0751 | 0.1953 | 1 | no |
| GEN | `ER_NOD` | -0.0075 | 0.7422 | 1 | yes |
| GEN | `SHUF_NOD` | +0.0126 | 0.7422 | 1 | no |
| GEN | `DENS_NOD` | +0.0313 | 0.9453 | 1 | no |

## Exploratory: FLY_CX_DEL vs FLY_CX_NOD (raw p)

- HEADING: Δ=+0.0439 (DEL−NOD), raw p=0.05469
- DRIFT: Δ=-0.0038 (DEL−NOD), raw p=0.7422
- GEN: Δ=+0.0090 (DEL−NOD), raw p=0.3828

## Verdict

**Null / no win.** Under this compass proxy, real CX wiring does **not** beat delay-matched ER/SHUF/DENS controls after Bonferroni. Same lesson as generic tasks: once delays and α are matched, random wiring keeps up.

Rows: 192. Raw: `out/runs.jsonl`. Figure: `figs/compass_heading.png`.
