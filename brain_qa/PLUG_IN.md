# Plugging in your new AI (twin)

`brain_qa` is retrieval-first. The default answerer is **extractive** (top BM25 chunks + quoted numbers).

## Drop-in generator

1. Write a class with `generate(self, query, hits) -> dict` where `hits` is a list of `(chunk_dict, score)` and the dict must include at least `answer` (str), and should include `citations` and `numbers`.
2. Point the env var at it:

```bash
export BRAIN_QA_GENERATOR=my_pkg.my_mod:MyGenerator
python ask.py "what was RULE_FF_DEPTH NARMA?"
```

`chunk_dict` fields: `id`, `path`, `i`, `text`.

## Example stub

```python
# my_generator.py
from retriever import extractive_answer

class MyGenerator:
    def __init__(self):
        # load your twin / GGUF / API client here
        pass
    def generate(self, query, hits):
        base = extractive_answer(query, hits)  # keep citations honest
        # optional: rewrite base["answer"] with your model, constrained to cited numbers
        return base
```

## Do not

- Bypass retrieval and invent numbers.
- Commit API keys into this folder.
