"""Ideal (noiseless) shallow fidelity kernel by statevector simulation + classical kernels + simple SVM."""
import numpy as np
def _ry(t): c, s = np.cos(t/2), np.sin(t/2); return np.array([[c, -s], [s, c]])
def feature_state(x, scale=1.0):
    n = len(x); x = np.asarray(x) * scale
    psi = np.array([1.0 + 0j])
    for xi in x: psi = np.kron(psi, _ry(xi) @ np.array([1, 0]))
    idx = np.arange(2**n); bits = [(idx >> (n-1-q)) & 1 for q in range(n)]
    phase = np.zeros(2**n)
    for q in range(n):
        r = (q + 1) % n
        if r == q: continue
        z = (1 - 2*bits[q]) * (1 - 2*bits[r]); phase += x[q] * x[r] * z
    psi = psi * np.exp(-1j * phase / 2)
    for q in range(n):   # second RY layer
        psi = psi.reshape([2]*n); psi = np.moveaxis(np.tensordot(_ry(x[q]), psi, axes=([1], [q])), 0, q).reshape(-1)
    return psi
def fidelity_kernel(X, Y, scale=1.0):
    A = np.array([feature_state(x, scale) for x in X]); B = A if Y is X else np.array([feature_state(y, scale) for y in Y])
    return np.abs(A.conj() @ B.T) ** 2
def rbf_kernel(X, Y, gamma): d = ((X[:, None, :] - Y[None, :, :])**2).sum(-1); return np.exp(-gamma * d)
def svm_fit(K, y, C=1.0, iters=2000, lr=1e-2):
    """binary SVM dual by projected gradient (y in {-1,+1}); adequate for small n."""
    a = np.zeros(len(y)); Q = (y[:, None] * y[None, :]) * K
    for _ in range(iters):
        a = np.clip(a + lr * (1 - Q @ a), 0, C); a -= y * (a @ y) / len(y); a = np.clip(a, 0, C)
    sv = (a > 1e-6) & (a < C - 1e-6); b = np.mean(y[sv] - (K[sv] @ (a * y))) if sv.any() else 0.0
    return a, b
def svm_decision(Kxt, a, y, b): return Kxt @ (a * y) + b
