"""Grouped split -> fit on train only -> predict. Gold labels joined only via evaluator-provided arrays."""
import numpy as np
from .features import assert_view
def grouped_split(groups, test_frac, seed):
    g = sorted(set(groups)); rng = np.random.default_rng(seed); rng.shuffle(g)
    test_g = set(g[:max(1, int(len(g)*test_frac))]); te = np.array([x in test_g for x in groups])
    return ~te, te
def fit_scaler(X_train):
    mu, sd = X_train.mean(0), X_train.std(0) + 1e-9; return lambda X: (X - mu) / sd
def views_only(items):
    for it in items: assert_view(it)
    return items
