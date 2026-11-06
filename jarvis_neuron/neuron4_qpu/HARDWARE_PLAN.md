# Hardware plan — boundary keep/cut QAOA (NOT SUBMITTED)

Balance assumed: **345 s**. Reserve: **≥90 s**. Cap this experiment: **≤40 s**.

## Proposed job (if user later approves)

- Algorithm: QAOA p=1 (prefer) or p=2, parameters from ideal Aer optimum
- Qubits: 16 (1 per decision var), no ancilla
- Shots: 4096
- Circuits: 1 optimized circuit (+ optional 1 idle/verify, still under cap)
- Backend: least-busy with ≥16 qubits at submit time
- Pessimistic estimate: **8–10 s** → leaves ~335 s (≥90 reserve)

## Explicit non-actions (this prepare phase)

- No `IBM_QUANTUM_TOKEN` load/print/log
- No IBM Runtime / Provider job
- No noise-model “fake hardware” billed as hardware

## Verdict

Gate passes; **skip spend** until NEURON-4 produces a harder instance. See RESULTS.md.
