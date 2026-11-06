# JARVIS default pipeline re-check (~30Q)

- Finished: 2026-11-03 16:06:26 ET (America/New_York)
- Defaults: mode=default(cad_hybrid), retrieval=routed, prefer_dict=True
- Checkpoint: `/workspace/jarvis/checkpoints/medium_v3/best.pt` step 14500
- Wall: 1002.3s

## Scores

| Metric | Got | Expected |
|--------|-----|----------|
| Retrieval pack-routed | **1.000** | 0.967 |
| Retrieval global | 0.967 | 0.933 |
| gen (greedy, global ctx) | **0.300** | 0.300 |
| CAD (global ctx) | **0.433** | 0.433 |
| pack-routed hybrid | **0.933** | 0.900 |
| dictionary /define (dict Qs) | **1.000** (6/6) | 1.000 (6/6) |
| **default pipeline** (routed+CAD hybrid+auto-dict) | **0.967** | ≥ hybrid / CAD |

## Checks

- PASS: retrieval_routed≈0.967
- PASS: cad≈0.433
- PASS: gen≈0.300
- PASS: routed_hybrid≈0.900
- PASS: dict_define=6/6

**all_pass = True**

## How to run chat with these defaults

```bash
cd /workspace/jarvis/chat_rag
OMP_NUM_THREADS=2 nice -n 19 python3 chat.py --once "What does neuron mean?"
# old modes: --mode greedy --retrieval global
```