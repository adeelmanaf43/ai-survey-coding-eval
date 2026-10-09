# Day 4: If a person checks only the answers where LLama and the classifier disagree,
# how much checking does that take, and how accurate is the final result?

import json
from pathlib import Path

import pandas as pd

SPLIT = "dev"

truth = pd.read_csv(f"data/processed/eval_{SPLIT}.csv", keep_default_na=False)
base = pd.read_csv(
    f"data/processed/baseline_pred_{SPLIT}.csv", keep_default_na=False)


summary = {}
for scheme in ("12", "50"):
    llm = pd.read_csv(
        f"data/processed/llm_{SPLIT}_{scheme}.csv", keep_default_na=False)[["id", "pred"]]
    df = (truth[["id", f"label{scheme}"]]
          .merge(llm, on="id")
          .merge(base[["id", f"pred_label{scheme}"]], on="id"))
    human, llama, clf = df[f"label{scheme}"], df["pred"], df[f"pred_label{scheme}"]

    agree = llama == clf
    llama_right = llama == human
    clf_right = clf == human

    # Assumption: the person who checks a disagreement gives the same code as the BES coder (best case)

    final_right = (agree & llama_right) | ~ agree

    s = {"n": len(df),
         "agree": int(agree.sum()),
         "accuracy_when_agree": round(llama_right[agree].mean(), 3),
         "disagree": int((~agree).sum()),
         "llama_right_when_disagree": round(llama_right[~agree].mean(), 3),
         "classifier_right_when_disagree": round(clf_right[~agree].mean(), 3),
         "llama_errors": int((~llama_right).sum()),
         "llama_errors_caught": int((~llama_right & ~agree).sum()),
         "accuracy_llama_alone": round(llama_right.mean(), 3),
         "accuracy_with_review": round(final_right.mean(), 3)}

    summary[f"{scheme}_cat"] = s

    print(f"\n{scheme} categories ({s['n']} answers)")
    print(
        f" Agree on {s['agree']} answers ({s['agree'] / s['n']:.0%}: right {s['accuracy_when_agree']:.1%} of the time)")
    print(
        f" Disagree on {s['disagree']} answers ({s['disagree'] / s['n']:.0%}): Llama right{s['llama_right_when_disagree']:.0%},"
        f" classifier right {s['classifier_right_when_disagree']:.0f}")
    print(f" LLama's {s['llama_errors']} mistakes: {s['llama_errors_caught']} among the disagreements (a person would see them),"
          f" {s['llama_errors'] - s['llama_errors_caught']} where both methods agree (missed)")
    print(
        f" Accuracy: Llama alnoe {s['accuracy_llama_alone']:.1%} | with a person checking the {s['disagree']} disagreements"
        f" {s['accuracy_with_review']:.1%}")


Path("results").mkdir(exist_ok=True)
with open(f"results/second_opionion_{SPLIT}.json", "w") as f:
    json.dump(
        {"split": SPLIT, "assumption": "a person checking a disagreement gives the BES coder's code",
         "results": summary, }, f, indent=2)

print(f"\nSaved results/second_opinion_{SPLIT}.json")
