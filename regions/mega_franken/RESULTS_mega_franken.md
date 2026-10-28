# RESULTS: Mega-Frankenstein

**Date:** 2026-10-28  
**Prereg:** `PREREGISTRATION_mega_franken.md`  
**Rows:** 1440 across 60 arms (eval seeds 0–7).  
**Delays:** FIXED + real H01 myelin on human-pre (franken_realmyelin).  
**Octopus block:** **SYNTHETIC** (`octo_vl_synthetic.npz`) — not a real cephalopod connectome.  
**Sizes:** {"FRANKEN": 6833, "FRANKEN+FINCH": 8833, "FRANKEN+WORM": 7133, "MEGA": 11133}  
**Figures:** `mega_franken_mc_vs_sigma.png`, `mega_franken_delta_mc.png`

## Summary

Primary (UNIF) REAL wins after Bonferroni×54: **0 / 54**.
Primary CTRL wins after correction: **0 / 54**.
Secondary (REALISTIC) REAL wins: **0 / 54**.
Mean |Δ MC| (REAL−ER, UNIF) at σ=0.05: 0.913; at σ=0.5: 0.531.

**Honest verdict: null.** Adding finch / worm / synthetic-octo sections does not beat size-matched ER/SHUF replacements after correction at any tested σ under UNIF bridges. Stronger bridges did not unlock a Level-2 composition win on these generic tasks. Raising σ changed mean |Δ MC| by -0.381 (exploratory).

## Means ± SEM — UNIF bridges

| Arm | σ | MC | NARMA | PC_HARD | N |
|---|---:|---:|---:|---:|---:|
| `FRANKEN` | 0.05 | 18.755±0.338 | 0.368±0.010 | 0.881±0.013 | 8 |
| `FRANKEN` | 0.2 | 18.042±0.228 | 0.327±0.006 | 0.885±0.010 | 8 |
| `FRANKEN` | 0.5 | 17.825±0.234 | 0.403±0.009 | 0.897±0.009 | 8 |
| `FRANKEN+FINCH_REAL` | 0.05 | 18.882±0.348 | 0.359±0.012 | 0.865±0.024 | 8 |
| `FRANKEN+FINCH_REAL` | 0.2 | 17.946±0.253 | 0.322±0.010 | 0.865±0.026 | 8 |
| `FRANKEN+FINCH_REAL` | 0.5 | 17.881±0.274 | 0.390±0.014 | 0.868±0.027 | 8 |
| `FRANKEN+FINCH_ER` | 0.05 | 19.850±0.434 | 0.410±0.015 | 0.874±0.028 | 8 |
| `FRANKEN+FINCH_ER` | 0.2 | 19.425±0.377 | 0.398±0.014 | 0.885±0.023 | 8 |
| `FRANKEN+FINCH_ER` | 0.5 | 17.886±0.450 | 0.465±0.017 | 0.901±0.023 | 8 |
| `FRANKEN+FINCH_SHUF` | 0.05 | 18.882±0.314 | 0.351±0.011 | 0.864±0.023 | 8 |
| `FRANKEN+FINCH_SHUF` | 0.2 | 17.925±0.236 | 0.318±0.009 | 0.870±0.020 | 8 |
| `FRANKEN+FINCH_SHUF` | 0.5 | 17.773±0.250 | 0.387±0.013 | 0.879±0.021 | 8 |
| `FRANKEN+WORM_REAL` | 0.05 | 18.597±0.281 | 0.369±0.013 | 0.911±0.013 | 8 |
| `FRANKEN+WORM_REAL` | 0.2 | 17.762±0.367 | 0.458±0.014 | 0.912±0.009 | 8 |
| `FRANKEN+WORM_REAL` | 0.5 | 17.245±0.257 | 0.376±0.011 | 0.911±0.013 | 8 |
| `FRANKEN+WORM_ER` | 0.05 | 18.489±0.273 | 0.364±0.012 | 0.919±0.009 | 8 |
| `FRANKEN+WORM_ER` | 0.2 | 17.842±0.352 | 0.457±0.013 | 0.916±0.008 | 8 |
| `FRANKEN+WORM_ER` | 0.5 | 17.462±0.173 | 0.366±0.011 | 0.918±0.014 | 8 |
| `FRANKEN+WORM_SHUF` | 0.05 | 18.486±0.254 | 0.363±0.011 | 0.920±0.008 | 8 |
| `FRANKEN+WORM_SHUF` | 0.2 | 17.840±0.353 | 0.457±0.013 | 0.917±0.007 | 8 |
| `FRANKEN+WORM_SHUF` | 0.5 | 17.431±0.158 | 0.367±0.009 | 0.922±0.009 | 8 |
| `MEGA_REAL` | 0.05 | 18.359±0.278 | 0.365±0.005 | 0.877±0.025 | 8 |
| `MEGA_REAL` | 0.2 | 17.721±0.275 | 0.457±0.005 | 0.877±0.022 | 8 |
| `MEGA_REAL` | 0.5 | 16.675±0.258 | 0.376±0.006 | 0.897±0.015 | 8 |
| `MEGA_ER` | 0.05 | 20.021±0.320 | 0.423±0.005 | 0.885±0.030 | 8 |
| `MEGA_ER` | 0.2 | 19.451±0.267 | 0.409±0.005 | 0.881±0.032 | 8 |
| `MEGA_ER` | 0.5 | 18.047±0.283 | 0.468±0.004 | 0.901±0.028 | 8 |
| `MEGA_SHUF` | 0.05 | 18.031±0.259 | 0.353±0.007 | 0.858±0.019 | 8 |
| `MEGA_SHUF` | 0.2 | 17.551±0.257 | 0.448±0.005 | 0.858±0.015 | 8 |
| `MEGA_SHUF` | 0.5 | 16.786±0.223 | 0.367±0.005 | 0.885±0.017 | 8 |

## Means ± SEM — REALISTIC bridges

| Arm | σ | MC | NARMA | PC_HARD | N |
|---|---:|---:|---:|---:|---:|
| `FRANKEN` | 0.05 | 18.755±0.338 | 0.368±0.010 | 0.881±0.013 | 8 |
| `FRANKEN` | 0.2 | 18.042±0.228 | 0.327±0.006 | 0.885±0.010 | 8 |
| `FRANKEN` | 0.5 | 17.825±0.234 | 0.403±0.009 | 0.897±0.009 | 8 |
| `FRANKEN+FINCH_REAL` | 0.05 | 19.298±0.282 | 0.369±0.011 | 0.882±0.010 | 8 |
| `FRANKEN+FINCH_REAL` | 0.2 | 18.284±0.189 | 0.324±0.011 | 0.885±0.010 | 8 |
| `FRANKEN+FINCH_REAL` | 0.5 | 18.042±0.206 | 0.392±0.014 | 0.892±0.012 | 8 |
| `FRANKEN+FINCH_ER` | 0.05 | 18.258±0.217 | 0.370±0.014 | 0.913±0.006 | 8 |
| `FRANKEN+FINCH_ER` | 0.2 | 17.570±0.287 | 0.475±0.015 | 0.915±0.006 | 8 |
| `FRANKEN+FINCH_ER` | 0.5 | 18.225±0.274 | 0.454±0.015 | 0.929±0.007 | 8 |
| `FRANKEN+FINCH_SHUF` | 0.05 | 19.218±0.244 | 0.354±0.011 | 0.877±0.011 | 8 |
| `FRANKEN+FINCH_SHUF` | 0.2 | 18.206±0.157 | 0.322±0.011 | 0.879±0.011 | 8 |
| `FRANKEN+FINCH_SHUF` | 0.5 | 18.009±0.205 | 0.390±0.013 | 0.893±0.010 | 8 |
| `FRANKEN+WORM_REAL` | 0.05 | 18.693±0.236 | 0.397±0.008 | 0.893±0.015 | 8 |
| `FRANKEN+WORM_REAL` | 0.2 | 17.760±0.198 | 0.353±0.007 | 0.889±0.011 | 8 |
| `FRANKEN+WORM_REAL` | 0.5 | 17.557±0.232 | 0.420±0.008 | 0.897±0.010 | 8 |
| `FRANKEN+WORM_ER` | 0.05 | 18.626±0.212 | 0.392±0.008 | 0.894±0.012 | 8 |
| `FRANKEN+WORM_ER` | 0.2 | 17.739±0.177 | 0.349±0.006 | 0.895±0.013 | 8 |
| `FRANKEN+WORM_ER` | 0.5 | 17.540±0.232 | 0.420±0.008 | 0.898±0.013 | 8 |
| `FRANKEN+WORM_SHUF` | 0.05 | 18.641±0.216 | 0.393±0.008 | 0.903±0.014 | 8 |
| `FRANKEN+WORM_SHUF` | 0.2 | 17.701±0.172 | 0.349±0.006 | 0.902±0.013 | 8 |
| `FRANKEN+WORM_SHUF` | 0.5 | 17.546±0.215 | 0.417±0.008 | 0.890±0.014 | 8 |
| `MEGA_REAL` | 0.05 | 19.077±0.328 | 0.367±0.009 | 0.883±0.016 | 8 |
| `MEGA_REAL` | 0.2 | 17.818±0.202 | 0.324±0.008 | 0.883±0.015 | 8 |
| `MEGA_REAL` | 0.5 | 17.468±0.268 | 0.383±0.007 | 0.885±0.021 | 8 |
| `MEGA_ER` | 0.05 | 17.952±0.227 | 0.364±0.008 | 0.917±0.015 | 8 |
| `MEGA_ER` | 0.2 | 17.473±0.325 | 0.467±0.009 | 0.915±0.013 | 8 |
| `MEGA_ER` | 0.5 | 17.895±0.289 | 0.444±0.007 | 0.920±0.012 | 8 |
| `MEGA_SHUF` | 0.05 | 18.954±0.312 | 0.351±0.010 | 0.893±0.013 | 8 |
| `MEGA_SHUF` | 0.2 | 17.738±0.229 | 0.321±0.008 | 0.882±0.016 | 8 |
| `MEGA_SHUF` | 0.5 | 17.561±0.233 | 0.380±0.007 | 0.871±0.022 | 8 |

### Primary family (UNIF, Bonferroni ×54)

| Stack | σ | Ctrl | Task | Δ (REAL−ctrl) | p_raw | p_bonf | favor | call |
|---|---:|---|---|---:|---:|---:|---|---|
| FINCH | 0.05 | ER | MC | -0.9680 | 0.007812 | 0.4219 | CTRL | n.s. |
| FINCH | 0.05 | ER | NARMA | -0.0506 | 0.007812 | 0.4219 | REAL | n.s. |
| FINCH | 0.05 | ER | PC_HARD | -0.0087 | 0.1953 | 1 | CTRL | n.s. |
| FINCH | 0.05 | SHUF | MC | +0.0004 | 0.8438 | 1 | REAL | n.s. |
| FINCH | 0.05 | SHUF | NARMA | +0.0082 | 0.007812 | 0.4219 | CTRL | n.s. |
| FINCH | 0.05 | SHUF | PC_HARD | +0.0016 | 0.7422 | 1 | REAL | n.s. |
| FINCH | 0.2 | ER | MC | -1.4791 | 0.007812 | 0.4219 | CTRL | n.s. |
| FINCH | 0.2 | ER | NARMA | -0.0767 | 0.007812 | 0.4219 | REAL | n.s. |
| FINCH | 0.2 | ER | PC_HARD | -0.0194 | 0.07812 | 1 | CTRL | n.s. |
| FINCH | 0.2 | SHUF | MC | +0.0212 | 0.8438 | 1 | REAL | n.s. |
| FINCH | 0.2 | SHUF | NARMA | +0.0041 | 0.01562 | 0.8438 | CTRL | n.s. |
| FINCH | 0.2 | SHUF | PC_HARD | -0.0043 | 0.8438 | 1 | CTRL | n.s. |
| FINCH | 0.5 | ER | MC | -0.0054 | 0.9453 | 1 | CTRL | n.s. |
| FINCH | 0.5 | ER | NARMA | -0.0753 | 0.007812 | 0.4219 | REAL | n.s. |
| FINCH | 0.5 | ER | PC_HARD | -0.0335 | 0.007812 | 0.4219 | CTRL | n.s. |
| FINCH | 0.5 | SHUF | MC | +0.1076 | 0.3125 | 1 | REAL | n.s. |
| FINCH | 0.5 | SHUF | NARMA | +0.0031 | 0.25 | 1 | CTRL | n.s. |
| FINCH | 0.5 | SHUF | PC_HARD | -0.0112 | 0.25 | 1 | CTRL | n.s. |
| WORM | 0.05 | ER | MC | +0.1082 | 0.1953 | 1 | REAL | n.s. |
| WORM | 0.05 | ER | NARMA | +0.0048 | 0.1953 | 1 | CTRL | n.s. |
| WORM | 0.05 | ER | PC_HARD | -0.0070 | 0.9453 | 1 | CTRL | n.s. |
| WORM | 0.05 | SHUF | MC | +0.1106 | 0.3828 | 1 | REAL | n.s. |
| WORM | 0.05 | SHUF | NARMA | +0.0060 | 0.007812 | 0.4219 | CTRL | n.s. |
| WORM | 0.05 | SHUF | PC_HARD | -0.0081 | 0.6406 | 1 | CTRL | n.s. |
| WORM | 0.2 | ER | MC | -0.0797 | 0.1953 | 1 | CTRL | n.s. |
| WORM | 0.2 | ER | NARMA | +0.0016 | 0.3125 | 1 | CTRL | n.s. |
| WORM | 0.2 | ER | PC_HARD | -0.0039 | 0.3828 | 1 | CTRL | n.s. |
| WORM | 0.2 | SHUF | MC | -0.0780 | 0.25 | 1 | CTRL | n.s. |
| WORM | 0.2 | SHUF | NARMA | +0.0016 | 0.3125 | 1 | CTRL | n.s. |
| WORM | 0.2 | SHUF | PC_HARD | -0.0046 | 0.8438 | 1 | CTRL | n.s. |
| WORM | 0.5 | ER | MC | -0.2171 | 0.1953 | 1 | CTRL | n.s. |
| WORM | 0.5 | ER | NARMA | +0.0100 | 0.03906 | 1 | CTRL | n.s. |
| WORM | 0.5 | ER | PC_HARD | -0.0063 | 0.25 | 1 | CTRL | n.s. |
| WORM | 0.5 | SHUF | MC | -0.1860 | 0.3125 | 1 | CTRL | n.s. |
| WORM | 0.5 | SHUF | NARMA | +0.0086 | 0.1484 | 1 | CTRL | n.s. |
| WORM | 0.5 | SHUF | PC_HARD | -0.0113 | 0.1094 | 1 | CTRL | n.s. |
| MEGA | 0.05 | ER | MC | -1.6618 | 0.007812 | 0.4219 | CTRL | n.s. |
| MEGA | 0.05 | ER | NARMA | -0.0588 | 0.007812 | 0.4219 | REAL | n.s. |
| MEGA | 0.05 | ER | PC_HARD | -0.0075 | 0.6406 | 1 | CTRL | n.s. |
| MEGA | 0.05 | SHUF | MC | +0.3284 | 0.03906 | 1 | REAL | n.s. |
| MEGA | 0.05 | SHUF | NARMA | +0.0117 | 0.007812 | 0.4219 | CTRL | n.s. |
| MEGA | 0.05 | SHUF | PC_HARD | +0.0188 | 0.1484 | 1 | REAL | n.s. |
| MEGA | 0.2 | ER | MC | -1.7300 | 0.007812 | 0.4219 | CTRL | n.s. |
| MEGA | 0.2 | ER | NARMA | +0.0478 | 0.007812 | 0.4219 | CTRL | n.s. |
| MEGA | 0.2 | ER | PC_HARD | -0.0040 | 0.8438 | 1 | CTRL | n.s. |
| MEGA | 0.2 | SHUF | MC | +0.1696 | 0.03906 | 1 | REAL | n.s. |
| MEGA | 0.2 | SHUF | NARMA | +0.0087 | 0.02344 | 1 | CTRL | n.s. |
| MEGA | 0.2 | SHUF | PC_HARD | +0.0192 | 0.1953 | 1 | REAL | n.s. |
| MEGA | 0.5 | ER | MC | -1.3718 | 0.007812 | 0.4219 | CTRL | n.s. |
| MEGA | 0.5 | ER | NARMA | -0.0917 | 0.007812 | 0.4219 | REAL | n.s. |
| MEGA | 0.5 | ER | PC_HARD | -0.0041 | 0.7422 | 1 | CTRL | n.s. |
| MEGA | 0.5 | SHUF | MC | -0.1110 | 0.1953 | 1 | CTRL | n.s. |
| MEGA | 0.5 | SHUF | NARMA | +0.0085 | 0.25 | 1 | CTRL | n.s. |
| MEGA | 0.5 | SHUF | PC_HARD | +0.0117 | 0.4609 | 1 | REAL | n.s. |

### Secondary family (REALISTIC, Bonferroni ×54)

| Stack | σ | Ctrl | Task | Δ (REAL−ctrl) | p_raw | p_bonf | favor | call |
|---|---:|---|---|---:|---:|---:|---|---|
| FINCH | 0.05 | ER | MC | +1.0400 | 0.007812 | 0.4219 | REAL | n.s. |
| FINCH | 0.05 | ER | NARMA | -0.0008 | 0.6406 | 1 | REAL | n.s. |
| FINCH | 0.05 | ER | PC_HARD | -0.0308 | 0.007812 | 0.4219 | CTRL | n.s. |
| FINCH | 0.05 | SHUF | MC | +0.0800 | 0.3125 | 1 | REAL | n.s. |
| FINCH | 0.05 | SHUF | NARMA | +0.0149 | 0.007812 | 0.4219 | CTRL | n.s. |
| FINCH | 0.05 | SHUF | PC_HARD | +0.0060 | 0.1484 | 1 | REAL | n.s. |
| FINCH | 0.2 | ER | MC | +0.7137 | 0.01562 | 0.8438 | REAL | n.s. |
| FINCH | 0.2 | ER | NARMA | -0.1509 | 0.007812 | 0.4219 | REAL | n.s. |
| FINCH | 0.2 | ER | PC_HARD | -0.0295 | 0.007812 | 0.4219 | CTRL | n.s. |
| FINCH | 0.2 | SHUF | MC | +0.0779 | 0.25 | 1 | REAL | n.s. |
| FINCH | 0.2 | SHUF | NARMA | +0.0019 | 0.007812 | 0.4219 | CTRL | n.s. |
| FINCH | 0.2 | SHUF | PC_HARD | +0.0064 | 0.25 | 1 | REAL | n.s. |
| FINCH | 0.5 | ER | MC | -0.1830 | 0.1094 | 1 | CTRL | n.s. |
| FINCH | 0.5 | ER | NARMA | -0.0630 | 0.007812 | 0.4219 | REAL | n.s. |
| FINCH | 0.5 | ER | PC_HARD | -0.0377 | 0.007812 | 0.4219 | CTRL | n.s. |
| FINCH | 0.5 | SHUF | MC | +0.0329 | 0.7422 | 1 | REAL | n.s. |
| FINCH | 0.5 | SHUF | NARMA | +0.0016 | 0.05469 | 1 | CTRL | n.s. |
| FINCH | 0.5 | SHUF | PC_HARD | -0.0011 | 1 | 1 | CTRL | n.s. |
| WORM | 0.05 | ER | MC | +0.0677 | 0.1484 | 1 | REAL | n.s. |
| WORM | 0.05 | ER | NARMA | +0.0055 | 0.007812 | 0.4219 | CTRL | n.s. |
| WORM | 0.05 | ER | PC_HARD | -0.0005 | 0.9453 | 1 | CTRL | n.s. |
| WORM | 0.05 | SHUF | MC | +0.0521 | 0.4609 | 1 | REAL | n.s. |
| WORM | 0.05 | SHUF | NARMA | +0.0043 | 0.1953 | 1 | CTRL | n.s. |
| WORM | 0.05 | SHUF | PC_HARD | -0.0103 | 0.4609 | 1 | CTRL | n.s. |
| WORM | 0.2 | ER | MC | +0.0217 | 0.9453 | 1 | REAL | n.s. |
| WORM | 0.2 | ER | NARMA | +0.0043 | 0.01562 | 0.8438 | CTRL | n.s. |
| WORM | 0.2 | ER | PC_HARD | -0.0061 | 0.7422 | 1 | CTRL | n.s. |
| WORM | 0.2 | SHUF | MC | +0.0598 | 0.3828 | 1 | REAL | n.s. |
| WORM | 0.2 | SHUF | NARMA | +0.0042 | 0.1094 | 1 | CTRL | n.s. |
| WORM | 0.2 | SHUF | PC_HARD | -0.0131 | 0.3828 | 1 | CTRL | n.s. |
| WORM | 0.5 | ER | MC | +0.0175 | 0.25 | 1 | REAL | n.s. |
| WORM | 0.5 | ER | NARMA | +0.0006 | 0.5469 | 1 | CTRL | n.s. |
| WORM | 0.5 | ER | PC_HARD | -0.0009 | 0.5469 | 1 | CTRL | n.s. |
| WORM | 0.5 | SHUF | MC | +0.0116 | 0.9453 | 1 | REAL | n.s. |
| WORM | 0.5 | SHUF | NARMA | +0.0033 | 0.007812 | 0.4219 | CTRL | n.s. |
| WORM | 0.5 | SHUF | PC_HARD | +0.0071 | 0.6406 | 1 | REAL | n.s. |
| MEGA | 0.05 | ER | MC | +1.1247 | 0.007812 | 0.4219 | REAL | n.s. |
| MEGA | 0.05 | ER | NARMA | +0.0028 | 0.5469 | 1 | CTRL | n.s. |
| MEGA | 0.05 | ER | PC_HARD | -0.0335 | 0.007812 | 0.4219 | CTRL | n.s. |
| MEGA | 0.05 | SHUF | MC | +0.1232 | 0.5469 | 1 | REAL | n.s. |
| MEGA | 0.05 | SHUF | NARMA | +0.0158 | 0.007812 | 0.4219 | CTRL | n.s. |
| MEGA | 0.05 | SHUF | PC_HARD | -0.0102 | 0.25 | 1 | CTRL | n.s. |
| MEGA | 0.2 | ER | MC | +0.3451 | 0.1094 | 1 | REAL | n.s. |
| MEGA | 0.2 | ER | NARMA | -0.1434 | 0.007812 | 0.4219 | REAL | n.s. |
| MEGA | 0.2 | ER | PC_HARD | -0.0325 | 0.01562 | 0.8438 | CTRL | n.s. |
| MEGA | 0.2 | SHUF | MC | +0.0796 | 0.3828 | 1 | REAL | n.s. |
| MEGA | 0.2 | SHUF | NARMA | +0.0025 | 0.03906 | 1 | CTRL | n.s. |
| MEGA | 0.2 | SHUF | PC_HARD | +0.0005 | 0.7422 | 1 | REAL | n.s. |
| MEGA | 0.5 | ER | MC | -0.4269 | 0.02344 | 1 | CTRL | n.s. |
| MEGA | 0.5 | ER | NARMA | -0.0604 | 0.007812 | 0.4219 | REAL | n.s. |
| MEGA | 0.5 | ER | PC_HARD | -0.0346 | 0.1484 | 1 | CTRL | n.s. |
| MEGA | 0.5 | SHUF | MC | -0.0929 | 0.1094 | 1 | CTRL | n.s. |
| MEGA | 0.5 | SHUF | NARMA | +0.0036 | 0.1094 | 1 | CTRL | n.s. |
| MEGA | 0.5 | SHUF | PC_HARD | +0.0140 | 0.6406 | 1 | REAL | n.s. |

## Notes

- Primary isolates whether **added** real blocks beat ER/SHUF replacements; mouse/human/fly stay real in every arm.
- MEGA > FRANKEN alone is size-confounded and not a primary claim.
- Alpha per arm tuned on seeds 90–92 (MC), locked in `out/alpha_choice.json`.
