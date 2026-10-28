"""T1 post-hoc: subtract the irreducible floor (teachers disagree with each other) so adapter vs no-adapter is comparable."""
import os, numpy as np, pandas as pd
from scipy.stats import wilcoxon
import flybrain_reservoir as F
HERE = os.path.dirname(os.path.abspath(__file__)); df = pd.read_csv(os.path.join(HERE, "out/results.csv"))
aw = F.bench.T1().teacher_words
def floor(s, words):
    Xin = np.random.default_rng(s + 300).standard_normal((1024, 6)) / np.sqrt(6)
    Ys = [np.tanh(Xin @ F.bench.weights_from_word(w))[512:] for w in words]; m = np.mean(Ys, 0)
    return float(np.mean([np.mean((m - Y) ** 2) for Y in Ys]))
rows = []
for s in F.SEEDS:
    rw = [F.bench.random_word(np.random.RandomState(50000 + 3 * s + k)) for k in range(3)]
    fa, fn = floor(s, aw), floor(s, rw)
    for c in ["REAL", "ER", "DEG"]:
        r = df[(df.seed == s) & (df.cond == c)].iloc[0]
        rows.append(dict(seed=s, cond=c, excess_adapters=r.T1_MSE_adapters - fa, excess_noadapters=r.T1_MSE_noadapters - fn))
e = pd.DataFrame(rows); e.to_csv(os.path.join(HERE, "out/t1_excess.csv"), index=False)
for c in ["REAL", "ER", "DEG"]:
    a = e[e.cond == c]
    print(c, "adapters %.4f±%.4f" % (a.excess_adapters.mean(), a.excess_adapters.std()),
          "none %.4f±%.4f" % (a.excess_noadapters.mean(), a.excess_noadapters.std()),
          "Wilcoxon p=%.3g" % wilcoxon(a.excess_adapters, a.excess_noadapters).pvalue)
