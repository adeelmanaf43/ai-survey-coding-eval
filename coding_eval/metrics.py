"""The metrics. These functions take codes or numbers and return numbers: no files, no printing,
so they are easy to test (tests/test_metrics.py)."""
from math import comb

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score


def macro_f1(truth, pred):
    """F1 averaged over categories, every category counting equally."""
    return f1_score(truth, pred, average="macro", zero_division=0)


METRICS = {"accuracy": accuracy_score, "macro_f1": macro_f1, "kappa": cohen_kappa_score}


def scores(truth, pred):
    """Accuracy, macro-F1 and Cohen's kappa, rounded to 3 decimals."""
    return {name: round(f(truth, pred), 3) for name, f in METRICS.items()}


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value for a paired comparison: b answers where only the first method
    is right, c where only the second is. How likely is a split at least this uneven by chance?"""
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def bootstrap(truth, preds, resamples=1000, seed=2026):
    """Point value and 95% interval of every metric, for several methods scored on the SAME resamples (D-020).
    preds: {method name: codes}. Returns ({method: {metric: [point, low, high]}}, all resample values)."""
    labels = sorted(set(truth).union(*map(set, preds.values())))
    code = {name: i for i, name in enumerate(labels)}                  # numbers are faster than strings
    t = np.array([code[x] for x in truth])
    p = {m: np.array([code[x] for x in v]) for m, v in preds.items()}
    rng = np.random.default_rng(seed)
    draws = {m: {k: [] for k in METRICS} for m in p}
    for _ in range(resamples):
        i = rng.integers(0, len(t), len(t))                             # one resample, shared by all methods
        for m in p:
            for k, f in METRICS.items():
                draws[m][k].append(f(t[i], p[m][i]))
    out = {m: {k: [round(f(t, p[m]), 3), round(np.percentile(draws[m][k], 2.5), 3),
                   round(np.percentile(draws[m][k], 97.5), 3)] for k, f in METRICS.items()} for m in p}
    return out, draws


def gap(draws, a, b, point):
    """95% interval for (accuracy of a) minus (accuracy of b), in percentage points."""
    d = np.array(draws[a]["accuracy"]) - np.array(draws[b]["accuracy"])
    return [round(point * 100, 1), round(np.percentile(d, 2.5) * 100, 1), round(np.percentile(d, 97.5) * 100, 1)]


def review_curve(right):
    """right: was Llama right (1/0) for each answer, riskiest first. If a person checks the first k answers,
    returns, for every k: the accuracy of the answers left to the machine, and of the whole dataset
    if every checked answer ends up right (the best case)."""
    right = np.asarray(right)
    n = len(right)
    k = np.arange(n)                                   # answers already checked
    right_left = right[::-1].cumsum()[::-1]            # right answers among those not checked yet
    return right_left / (n - k), (right_left + k) / n


def mean_seconds(seconds, timeout):
    """Mean seconds per answer, leaving out answers recorded at over the timeout: the laptop was
    asleep during those, not working (D-022). Returns (mean, number left out)."""
    sec = pd.Series(seconds, dtype=float)
    return sec[sec <= timeout].mean(), int((sec > timeout).sum())
