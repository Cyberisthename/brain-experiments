# Hypothalamus A/B/C results

**Time zone:** America/New_York.

Rows=256 features=['BASE', 'HOME_O_ERR', 'RAND_MOD', 'RIVAL', 'SHUF_RIVAL', 'SINGLE', 'SYN', 'VOL'] tasks=['CHAR', 'NARMA', 'TRANSFER', 'XOR_HARD'] seeds=[3400, 3401, 3402, 3403, 3404, 3405, 3406, 3407]

Bonferroni α=0.004167 over 12 planned contrasts.

Wilcoxon n=8 floor≈0.0078 > α → confirmatory bar unreachable; mark exploratory.


## Means ± std

| Arm | XOR_HARD ↑ | NARMA ↓ | CHAR ↓ | TRANSFER ↑ |
|---|---:|---:|---:|---:|
| BASE | 0.5082±0.0090 | 1.3337±0.3051 | 2.2706±0.0333 | 0.7086±0.0472 |
| HOME_O_ERR | 0.5164±0.0166 | 1.5077±0.6100 | 2.2727±0.0323 | 0.7120±0.0394 |
| RAND_MOD | 0.5095±0.0098 | 2.2199±1.3730 | 2.2772±0.0354 | 0.7054±0.0256 |
| RIVAL | 0.5119±0.0108 | 1.6676±0.8593 | 2.2454±0.0316 | 0.6899±0.0493 |
| SINGLE | 0.5186±0.0182 | 1.5231±0.5417 | 2.2622±0.0312 | 0.7524±0.0235 |
| SHUF_RIVAL | 0.5070±0.0064 | 1.7124±0.8865 | 2.2927±0.0350 | 0.6988±0.0439 |
| VOL | 0.5074±0.0112 | 2.0405±0.8278 | 2.2654±0.0257 | 0.6872±0.0501 |
| SYN | 0.5131±0.0205 | 1.7174±0.9118 | 2.2752±0.0365 | 0.7085±0.0366 |

## Planned contrasts

| Test | A vs B | Task | effect (+better for A) | p | survives α? | verdict |
|---|---|---|---:|---:|---|---|
| A | HOME_O_ERR vs RAND_MOD | XOR_HARD | +0.0070 | 0.4609 | no | direction_ok_ns |
| A | HOME_O_ERR vs BASE | XOR_HARD | +0.0082 | 0.3828 | no | direction_ok_ns |
| A | HOME_O_ERR vs RAND_MOD | NARMA | +0.7122 | 0.1484 | no | direction_ok_ns |
| A | HOME_O_ERR vs BASE | NARMA | -0.1740 | 0.4609 | no | FAIL_direction |
| B | RIVAL vs SINGLE | XOR_HARD | -0.0068 | 0.6406 | no | FAIL_direction |
| B | RIVAL vs SHUF_RIVAL | XOR_HARD | +0.0049 | 0.05469 | no | direction_ok_ns |
| B | RIVAL vs SINGLE | TRANSFER | -0.0625 | 0.007812 | no | FAIL_direction |
| B | RIVAL vs SHUF_RIVAL | TRANSFER | -0.0089 | 0.8438 | no | FAIL_direction |
| C | VOL vs SYN | NARMA | -0.3231 | 0.25 | no | FAIL_direction |
| C | VOL vs BASE | NARMA | -0.7068 | 0.07812 | no | FAIL_direction |
| C | VOL vs SYN | XOR_HARD | -0.0057 | 0.7422 | no | FAIL_direction |
| C | VOL vs BASE | XOR_HARD | -0.0009 | 0.7422 | no | FAIL_direction |

## Hypothesis verdicts (prereg)

- **Test A:** NOT SUPPORTED (no planned exploratory hit)
- **Test B:** FALSIFIED (RIVAL ≤ SINGLE or ≤ SHUF on a planned task)
- **Test C:** FALSIFIED (SYN ≥ VOL on NARMA)

## Honesty
- Motif abstraction only; not a biological hypothalamus.
- No QPU. Callosum_scaled left untouched.
- STP ablation raw jsonl preserved under engine_v3/out/.

