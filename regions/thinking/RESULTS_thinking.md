# Thinking-time results

Prereg: `PREREGISTRATION_thinking.md`.

## Status: NO DATA (run bug)

`thinking.py` imports `compress/sag.py`, which asserts `len(CXID)==C.N` at import. AssertionError — stale graph / N mismatch. Zero jobs.

Fix: lazy CX load in sag.py + rebuild CX graph (2875 neurons, ~300234 edges), then nice-rerun.
