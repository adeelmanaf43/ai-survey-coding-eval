# Human benchmark: your 100 codes against the BES coders, next to Llama and the
# classifier on the SAME 100 answers. Prints and saves numbers and category names only (safe to share).
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

human = pd.read_csv("data/processed/human_test_12.csv", keep_default_na=False)
truth = pd.read_csv("data/processed/eval_test.csv",
                    keep_default_na=False)[["id", "label12"]]
llama = pd.read_csv("data/processed/llm_test_12.csv",
                    keep_default_na=False)[["id", "pred"]]
clf = pd.read_csv("data/processed/baseline_pred_test.csv",
                  keep_default_na=False)[["id", "pred_label12"]]
df = human.merge(truth, on="id").merge(llama, on="id").merge(clf, on="id")

print(f"Answers you coded: {len(df)}\n")
methods = {}
for name, col in [("You", "human"), ("Llama 3.1 8B", "pred"), ("TF-IDF classifier", "pred_label12")]:
    right = df[col] == df["label12"]
    kappa = cohen_kappa_score(df["label12"], df[col])
    methods[name] = {"right": int(right.sum()), "agreement": round(
        right.mean(), 3), "kappa": round(kappa, 3)}
    print(f"  {name:<18} agrees with the BES coders on {right.sum()} of {len(df)} ({right.mean():.0%}), kappa {kappa:.2f}")

same_as_llama = (df["human"] == df["pred"]).mean()
print(
    f"  You and Llama chose the same code for {same_as_llama:.0%} of these answers")

sec = df["seconds"]
# cost depends on total time, so use the mean
per_1000 = sec.mean() * 1000 / 3600
print(
    f"\nYour time: median {sec.median():.0f} s per answer (mean {sec.mean():.0f} s), {sec.sum() / 60:.0f} min in total")
print(
    f"  At your average pace, coding 1,000 answers by hand takes about {per_1000:.1f} hours")
diff = df[df["human"] != df["label12"]].groupby(
    ["label12", "human"]).size().sort_values(ascending=False).head(8)
print("\nWhere you and the BES coders differed most (BES code -> your code):")
for (bes, you), n in diff.items():
    print(f"  {bes} -> {you}: {n}")

Path("results").mkdir(exist_ok=True)
with open("results/human_benchmark.json", "w") as f:
    json.dump({"scheme": 12, "sample": "random 100 of the 1,200 test answers (seed 2026)", "n": len(df),
               "methods": methods, "same_code_as_llama": round(same_as_llama, 3),
               "seconds_median": round(sec.median(), 1), "seconds_mean": round(sec.mean(), 1),
               "hours_per_1000_answers": round(per_1000, 1),
               "differences": [{"bes": b, "you": y, "count": int(n)} for (b, y), n in diff.items()]}, f, indent=2)
print("\nSaved results/human_benchmark.json")
