"""SYNTHETIC FIXTURES ONLY. Not human-validated, not results. Checks plumbing: leakage assertion, kernel PSD, svm runs, cost, hw disabled."""
import sys, numpy as np; sys.path.insert(0, ".")
from qk1 import features, kernels, cost, hardware_dryrun, pipeline
try: features.assert_view({"final": 1}); raise SystemExit("leak check failed")
except AssertionError: pass
rng = np.random.default_rng(0); X = rng.normal(size=(40, 6)); y = np.where(X[:, 0]*X[:, 1] > 0, 1.0, -1.0); groups = [i//4 for i in range(40)]
tr, te = pipeline.grouped_split(groups, 0.25, 1); sc = pipeline.fit_scaler(X[tr]); Xs = sc(X)
assert not (set(np.array(groups)[tr]) & set(np.array(groups)[te]))
K = kernels.fidelity_kernel(Xs[tr], Xs[tr], 0.5); assert np.allclose(K, K.T, atol=1e-9) and np.linalg.eigvalsh(K).min() > -1e-8 and np.allclose(np.diag(K), 1)
a, b = kernels.svm_fit(K, y[tr]); acc = np.mean(np.sign(kernels.svm_decision(kernels.fidelity_kernel(Xs[te], Xs[tr], 0.5), a, y[tr], b)) == y[te])
print("synthetic fidelity-kernel SVM acc (fixture only):", round(float(acc), 3))
print(cost.estimate(200, 140, balance_s=345))
try: hardware_dryrun.submit(); raise SystemExit("hw not disabled")
except RuntimeError: pass
f = features.pair_features({"text": "T1 = 1.85 kHz on 2026-11-01", "source": {"origin_id": "x"}}, {"text": "T1 ≈ 1.50 kHz on 2026-11-04", "source": {"origin_id": "y"}})
assert len(f) == 11; print("ok")
