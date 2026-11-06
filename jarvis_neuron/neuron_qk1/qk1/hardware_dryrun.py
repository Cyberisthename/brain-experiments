"""Hardware path: DISABLED. Builds a description of what would be submitted; never imports a provider, never reads a token."""
ENABLED = False
def plan(n_qubits, n_circuits, shots):
    return dict(enabled=ENABLED, n_qubits=n_qubits, n_circuits=n_circuits, shots=shots, note="dry-run only; requires explicit user approval and a separate reviewed change to enable")
def submit(*a, **k):
    raise RuntimeError("QK1 hardware submission is disabled (not approved).")
