# Score Llama's codes against the human codes, next to the baselines.
# Prints and saves numbers and category names only, so the output is safe to share and commit.
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

SPLIT = "test"

truth = pd.read_csv(f"data/processed/eval_{SPLIT}.csv", keep_default_na=False)
seen = pd.read_csv(
    f"data/processed/baseline_pred_{SPLIT}.csv", keep_default_na=False)[["id", "seen"]]
with open(f"results/baseline_{SPLIT}.json") as f:
    base = json.load(f)


def score(t, p):
    return {"accuracy": round(accuracy_score(t, p), 3),
            "macro_f1": round(f1_score(t, p, average="macro", zero_division=0), 3),
            "kappa": round(cohen_kappa_score(t, p), 3)}


def line(scheme, method, s, new=None):
    new_txt = f"{new:>12.1%}" if new is not None else f"{'':>12}"
    print(
        f"{scheme:<8}{method:<34}{s['accuracy']:>9.1%}{s['macro_f1']:>10.1%}{s['kappa']:>7.2f}{new_txt}")


summary, details = {}, []
print(f"\nResults on the {SPLIT} set")
print(f"{'Scheme':<8}{'Method':<34}{'Accuracy':>9}{'Macro-F1':>10}{'Kappa':>7}{'New answers':>12}")
for scheme in ("12", "50"):
    llm = pd.read_csv(
        f"data/processed/llm_{SPLIT}_{scheme}.csv", keep_default_na=False)
    df = truth.merge(llm, on="id").merge(seen, on="id", how="left")
    label = f"label{scheme}"
    right = df[label] == df["pred"]
    is_new = df["seen"].astype(str) == "False"
    unsure = df["unsure"].astype(str) == "True"

    for r in base["results"]:
        if str(r["scheme"]) == scheme:
            tfidf = "TF-IDF" in r["method"]
            line(scheme, r["method"], r, base["seen_vs_new"]
                 [f"{scheme}_cat"]["new"] if tfidf else None)
    s = score(df[label], df["pred"])
    s["new"] = round(right[is_new].mean(), 3)
    line(scheme, "Llama 3.1 8B, codebook only", s, s["new"])

    s.update({"n_coded": len(df), "unsure_share": round(unsure.mean(), 3),
              "accuracy_when_unsure": round(right[unsure].mean(), 3),
              "accuracy_when_sure": round(right[~unsure].mean(), 3),
              "median_seconds": round(df["seconds"].median(), 1),
              "invalid": int((df["pred"] == "INVALID").sum())})
    mistakes = df[~right].groupby(
        [label, "pred"]).size().sort_values(ascending=False).head(8)
    s["top_mistakes"] = [{"human": h, "llama": p,
                          "count": int(n)} for (h, p), n in mistakes.items()]
    summary[f"{scheme}_cat"] = s
    details.append((scheme, s))

for scheme, s in details:
    print(f"\n{scheme} categories: coded {s['n_coded']} of {len(truth)} answers | median {s['median_seconds']} s per answer"
          f" | invalid replies: {s['invalid']}")
    print(f"  Llama flagged {s['unsure_share']:.0%} as unsure: accuracy {s['accuracy_when_unsure']:.0%} when unsure,"
          f" {s['accuracy_when_sure']:.0%} when sure")
    print("  Most common mistakes (human code -> Llama's code):")
    for m in s["top_mistakes"]:
        print(f"    {m['human']} -> {m['llama']}: {m['count']}")

Path("results").mkdir(exist_ok=True)
with open(f"results/llm_{SPLIT}.json", "w") as f:
    json.dump({"split": SPLIT, "model": "llama3.1:latest",
              "results": summary}, f, indent=2)
print(f"\nSaved results/llm_{SPLIT}.json")
