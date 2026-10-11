# Final metrics with 95% confidence intervals (roadmap H9).
# Bootstrap: draw 1,000 new samples of the same size from the test answers (with replacement),
# recompute every metric on each, and keep the middle 95% of the values as the interval.
# Prints and saves numbers and category names only, so the output is safe to share and commit.
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score


SPLIT, RESAMPLES, SEED = "test", 1000, 2026
METRICS = {"accuracy": accuracy_score,
           "macro_f1": lambda t, p: f1_score(t, p, average="macro", zero_division=0),
           "kappa": cohen_kappa_score}


def bootstrap(truth, preds):
    """Point value and 95% interval of every metric, for several methods scored on the SAME resamples."""
    labels = sorted(set(truth).union(*map(set, preds.values())))
    # # numbers are faster than strings
    code = {name: i for i, name in enumerate(labels)}
    t = np.array([code[x] for x in truth])
    p = {m: np.array([code[x] for x in v]) for m, v in preds.items()}
    rng = np.random.default_rng(SEED)
    draws = {m: {k: [] for k in METRICS} for m in p}
    for _ in range(RESAMPLES):
        # one resample, shared by all methods
        i = rng.integers(0, len(t), len(t))
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


def show(name, r):
    a, f, k = r["accuracy"], r["macro_f1"], r["kappa"]
    print(f"  {name:<18} {a[0]:6.1%} ({a[1]:.1%}-{a[2]:.1%})   {f[0]:6.1%} ({f[1]:.1%}-{f[2]:.1%})"
          f"   {k[0]:.2f} ({k[1]:.2f}-{k[2]:.2f})")


truth = pd.read_csv(f"data/processed/eval_{SPLIT}.csv", keep_default_na=False)
base = pd.read_csv(
    f"data/processed/baseline_pred_{SPLIT}.csv", keep_default_na=False)
results = {"bootstrap": {"resamples": RESAMPLES,
                         "seed": SEED, "interval": "95%, percentile"}}
print(f"Test set, {len(truth):,} answers. 95% intervals from {RESAMPLES:,} bootstrap resamples (this takes a minute or two)")

for scheme in ("12", "50"):
    llm = pd.read_csv(
        f"data/processed/llm_{SPLIT}_{scheme}.csv", keep_default_na=False)
    df = truth[["id", f"label{scheme}"]].merge(llm[["id", "pred"]], on="id").merge(
        base[["id", f"pred_label{scheme}"]], on="id")
    t = df[f"label{scheme}"]
    preds = {
        "TF-IDF classifier": df[f"pred_label{scheme}"], "Llama 3.1 8B": df["pred"]}
    out, draws = bootstrap(t, preds)
    point = out["TF-IDF classifier"]["accuracy"][0] - \
        out["Llama 3.1 8B"]["accuracy"][0]
    out["classifier_minus_llama_accuracy_points"] = gap(
        draws, "TF-IDF classifier", "Llama 3.1 8B", point)

    # Per-category F1: where each method does well or badly
    cats = t.value_counts()
    f1c = f1_score(t, preds["TF-IDF classifier"],
                   labels=cats.index, average=None, zero_division=0)
    f1l = f1_score(t, preds["Llama 3.1 8B"],
                   labels=cats.index, average=None, zero_division=0)
    out["per_category_f1"] = [{"category": c, "n": int(n), "classifier": round(a, 3), "llama": round(b, 3)}
                              for c, n, a, b in zip(cats.index, cats.values, f1c, f1l)]
    results[f"{scheme}_cat"] = out

    print(f"\n{scheme} categories          Accuracy                Macro-F1                Kappa")
    show("TF-IDF classifier", out["TF-IDF classifier"])
    show("Llama 3.1 8B", out["Llama 3.1 8B"])
    g = out["classifier_minus_llama_accuracy_points"]
    print(
        f"  Classifier minus Llama: {g[0]:+.1f} points of accuracy (95% interval {g[1]:+.1f} to {g[2]:+.1f})")

print("\nF1 per category, 12 categories (n = test answers in the category)")
print(f"  {'Category':<20}{'n':>5}{'Classifier':>12}{'Llama':>8}")
for r in results["12_cat"]["per_category_f1"]:
    print(
        f"  {r['category']:<20}{r['n']:>5}{r['classifier']:>12.2f}{r['llama']:>8.2f}")

# The 100 answers you coded yourself (roadmap H8), next to both methods on exactly those answers
human = pd.read_csv("data/processed/human_test_12.csv", keep_default_na=False)
h = human.merge(truth[["id", "label12"]], on="id").merge(
    pd.read_csv(f"data/processed/llm_{SPLIT}_12.csv", keep_default_na=False)[["id", "pred"]], on="id").merge(
    base[["id", "pred_label12"]], on="id")
out, draws = bootstrap(h["label12"], {
                       "You": h["human"], "Llama 3.1 8B": h["pred"], "TF-IDF classifier": h["pred_label12"]})
point = out["Llama 3.1 8B"]["accuracy"][0] - out["You"]["accuracy"][0]
out["llama_minus_you_accuracy_points"] = gap(
    draws, "Llama 3.1 8B", "You", point)
results["human_sample_12_cat"] = dict(n=len(h), **out)
print(f"\nThe {len(h)} answers you coded, 12 categories   Accuracy             Macro-F1             Kappa")
for name in ("You", "Llama 3.1 8B", "TF-IDF classifier"):
    show(name, out[name])
g = out["llama_minus_you_accuracy_points"]
print(
    f"  Llama minus you: {g[0]:+.1f} points of accuracy (95% interval {g[1]:+.1f} to {g[2]:+.1f})")

Path("results").mkdir(exist_ok=True)
with open(f"results/metrics_{SPLIT}.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved results/metrics_{SPLIT}.json")
