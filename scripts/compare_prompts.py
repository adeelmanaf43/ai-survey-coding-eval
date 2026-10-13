# Prompt A (definitions only) vs prompt B (definitions plus examples, the frozen prompt): roadmap H5, D-023.
# Both prompts coded the same answers, so the comparison is paired: only answers where exactly one
# prompt is right can show a difference (McNemar's exact test). Prints numbers and category names only.
import argparse
import json
from math import comb
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

FROZEN_B = {"12": "459a2e64", "50": "b0fcfe68"}   # D-013

parser = argparse.ArgumentParser()
parser.add_argument("--split", choices=["dev", "test"], default="dev")
SPLIT = parser.parse_args().split


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value: how likely is a split at least this uneven if both prompts are equally good?"""
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


truth = pd.read_csv(f"data/processed/eval_{SPLIT}.csv", keep_default_na=False)
results = {"split": SPLIT}
print(f"Prompt A (definitions only) vs prompt B (definitions plus examples), {SPLIT} set")
for scheme in ("12", "50"):
    df = truth[["id", f"label{scheme}"]].rename(columns={f"label{scheme}": "human"})
    for name, suffix in (("A", "_A"), ("B", "")):
        run = pd.read_csv(f"data/processed/llm_{SPLIT}_{scheme}{suffix}.csv", keep_default_na=False)
        run = run.drop_duplicates("id", keep="last")[["id", "pred", "seconds", "prompt_tokens", "version"]]
        df = df.merge(run.rename(columns=lambda c: c if c == "id" else f"{c}_{name}"), on="id")

    out = {"answers": len(df)}
    print(f"\n{scheme} categories: {len(df)} answers coded by both prompts (of {len(truth)})")
    print(f"  {'Prompt':<8}{'Version':<10}{'Accuracy':>9}{'Macro-F1':>10}{'Kappa':>7}{'Typical s':>11}{'Prompt tokens':>15}")
    for name in ("A", "B"):
        p = df[f"pred_{name}"]
        out[name] = {"version": ", ".join(sorted(df[f"version_{name}"].astype(str).unique())),
                     "accuracy": round(accuracy_score(df["human"], p), 3),
                     "macro_f1": round(f1_score(df["human"], p, average="macro", zero_division=0), 3),
                     "kappa": round(cohen_kappa_score(df["human"], p), 3),
                     "median_seconds": round(pd.to_numeric(df[f"seconds_{name}"]).median(), 1),
                     "median_prompt_tokens": int(pd.to_numeric(df[f"prompt_tokens_{name}"]).median()),
                     "invalid": int((p == "INVALID").sum())}
        r = out[name]
        print(f"  {name:<8}{r['version']:<10}{r['accuracy']:>9.1%}{r['macro_f1']:>10.1%}{r['kappa']:>7.2f}"
              f"{r['median_seconds']:>11.1f}{r['median_prompt_tokens']:>15,}")
    if out["B"]["version"] != FROZEN_B[scheme]:
        print(f"  [!] prompt B rows have version {out['B']['version']}, not the frozen {FROZEN_B[scheme]}")

    right_a, right_b = df["pred_A"] == df["human"], df["pred_B"] == df["human"]
    only_b, only_a = int((right_b & ~right_a).sum()), int((right_a & ~right_b).sum())
    out.update({"same_code_share": round((df["pred_A"] == df["pred_B"]).mean(), 3),
                "only_B_right": only_b, "only_A_right": only_a,
                "mcnemar_exact_p": round(mcnemar_exact(only_b, only_a), 3)})
    print(f"  Same code from both prompts: {out['same_code_share']:.0%} of answers")
    print(f"  Only B right: {only_b} | only A right: {only_a} | McNemar exact p = {out['mcnemar_exact_p']:.3f}")

    # Where the examples changed the outcome (human code -> the wrong prompt's code)
    for label, mask, wrong in (("examples helped (B right, A wrong)", right_b & ~right_a, "pred_A"),
                               ("examples hurt (A right, B wrong)", right_a & ~right_b, "pred_B")):
        pairs = df[mask].groupby(["human", wrong]).size().sort_values(ascending=False).head(5)
        key = "helped" if "helped" in label else "hurt"
        out[f"examples_{key}"] = [{"human": h, "wrong_code": w, "count": int(n)} for (h, w), n in pairs.items()]
        print(f"  Where the {label}: " + ("; ".join(f"{h} -> {w} ({n})" for (h, w), n in pairs.items()) or "none"))
    results[f"{scheme}_cat"] = out

Path("results").mkdir(exist_ok=True)
with open(f"results/prompt_comparison_{SPLIT}.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved results/prompt_comparison_{SPLIT}.json")
