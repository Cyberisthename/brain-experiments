# human_wholebrain

Whole-brain human structural connectome (Lausanne scale-68) as recurrent wiring vs matched controls.

- Prereg: `PREREG_human_wholebrain.md` (freeze before seeds 2200–2207)
- Harness: `human_wholebrain.py`
- Analyze: `analyze_human_wholebrain.py` → `RESULTS.md`
- Research context: `../CALLOSUM_RESEARCH.md`
- Connectome data stays under `connectomes/` (gitignored); never push

```bash
python human_wholebrain.py --check-wiring
python human_wholebrain.py --smoke
# then:
nohup python human_wholebrain.py --eval --workers 2 > out/eval.log 2>&1 &
python analyze_human_wholebrain.py
```
