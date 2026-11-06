# JARVIS Tutor Curriculum (pilot)

Programmatic teacher for Lanner's from-scratch JARVIS. **No API teacher. No Llama/Qwen.**

## Layout
- `PREREG.md` — frozen hypothesis, arms, metrics (incl. STaR amendment A1, CPU scale A2)
- `vocab.py`, `teacher.py`, `curriculum.py`, `batching.py` — data + rule-based teacher
- `train_pilot.py` — arms: `tutor`, `no_expl`, `shuffled`, `star`
- `analyze_pilot.py` — frozen analysis → `RESULTS.md`
- `results/` — per-seed JSON + checkpoints

## Run
```bash
cd /workspace/jarvis/tutor
OMP_NUM_THREADS=1 python3 -u train_pilot.py
python3 analyze_pilot.py
# optional student-as-teacher:
RUN_STUDENT_TEACHER=1 OMP_NUM_THREADS=1 python3 -u -c "..."  # or re-run with env after ckpts exist
```

Does **not** launch/stop Kaggle.
