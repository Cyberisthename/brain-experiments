# brain_qa

Local Q&A over flybrain experiment markdown (`RESULTS*`, `PREREG*`, `ANALYSIS*`, `SOURCES.md`).

```bash
cd /workspace/brain_qa
python3 index_corpus.py
python3 ask.py "what was real vs shuffled myelin NARMA?"
python3 eval_qa.py
```

Default answers are **extractive** (BM25 + quotes). See `PLUG_IN.md` to attach a stronger generator later.
