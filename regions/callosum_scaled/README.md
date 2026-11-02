# callosum_scaled

Larger-N whole-SC / callosum prior with real fiber-length+myelin delays and
cross-hemisphere transfer tasks. See `PREREG_callosum_scaled.md`.

```bash
python callosum_scaled.py --check-wiring --n-labels 512
python callosum_scaled.py --smoke --n-labels 512
python callosum_scaled.py --time-one --n-labels 512 --arms REAL_SC --tasks TRANSFER
# eval (resume-safe):
nohup python callosum_scaled.py --eval --workers 2 --n-labels 512 > out/eval_512.log 2>&1 &
python analyze_callosum_scaled.py
```

Connectome binaries stay under `connectomes/` (gitignored). Never push data.
