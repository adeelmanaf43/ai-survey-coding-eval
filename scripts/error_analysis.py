# Error analysis (roadmap H6): where Llama's codes differ from the BES codes on the test set (D-025).
#   python scripts\error_analysis.py          confusion matrices + a blind reading sheet of 30 mistakes
#   python scripts\error_analysis.py --count  after you've filled in the sheet: count your verdicts
# The printed output and the files in results/ hold category names and counts only. The reading sheet
# holds answer text, so it stays in data/processed (git-ignored): read it on your laptop, never paste it.
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SPLIT, SEED, N_READ = "test", 2026, 30
SHEET = Path("data/processed/errors_to_read.csv")
# which code is the BES one: don't open it
KEY = Path("data/processed/errors_key.csv")
WHY = {"1": "several issues", "2": "vague answer",
       "3": "overlapping categories", "4": "clear error"}

parser = argparse.ArgumentParser()
parser.add_argument("--count", action="store_true",
                    help="count the verdicts in the filled-in sheet")
args = parser.parse_args()
truth = pd.read_csv(f"data/processed/eval_{SPLIT}.csv", keep_default_na=False)

if not args.count:
    Path("results").mkdir(exist_ok=True)
    print(f"Llama's mistakes on the {SPLIT} set (BES code -> Llama's code)")
    for scheme in ("12", "50"):
        llm = pd.read_csv(
            f"data/processed/llm_{SPLIT}_{scheme}.csv", keep_default_na=False)
        df = truth[["id", "text", f"label{scheme}"]].rename(columns={f"label{scheme}": "bes"}).merge(
            llm[["id", "pred"]], on="id")
        # The confusion matrix: one row per BES code, one column per Llama code, counts in the cells
        pd.crosstab(df["bes"].rename("BES code"), df["pred"].rename("Llama's code")).to_csv(
            f"results/confusion_llama_{SPLIT}_{scheme}.csv")
        wrong = df[df["bes"] != df["pred"]]
        print(
            f"\n{scheme} categories: {len(wrong)} mistakes in {len(df):,} answers. Most common mix-ups:")
        for (b, p), n in wrong.groupby(["bes", "pred"]).size().sort_values(ascending=False).head(10).items():
            print(f"  {b} -> {p}: {n}")
        if scheme != "12":
            continue
        print(
            f"\n  {'BES category':<20}{'answers':>8}{'Llama right':>13}   most common wrong code")
        for cat, n in df["bes"].value_counts().items():
            mine = df[df["bes"] == cat]
            missed = mine.loc[mine["pred"] != cat, "pred"].value_counts()
            top = f"{missed.index[0]} ({missed.iloc[0]})" if len(
                missed) else "-"
            print(
                f"  {cat:<20}{n:>8}{(mine['pred'] == cat).mean():>13.0%}   {top}")

        # The blind reading sheet: 30 random mistakes, with the two codes in random order as X and Y
        if SHEET.exists():
            print(
                f"\n{SHEET} already exists, not overwritten (delete it to start again)")
            continue
        pick = wrong.sample(n=min(N_READ, len(wrong)), random_state=SEED)
        bes_is_x = np.random.default_rng(SEED).random(len(pick)) < 0.5
        rows = range(1, len(pick) + 1)
        pd.DataFrame({"n": rows, "answer": pick["text"].to_numpy(),
                      "code_X": np.where(bes_is_x, pick["bes"], pick["pred"]),
                      "code_Y": np.where(bes_is_x, pick["pred"], pick["bes"]),
                      "verdict": "", "why": ""}).to_csv(SHEET, index=False, encoding="utf-8-sig")
        pd.DataFrame({"n": rows, "bes_is": np.where(
            bes_is_x, "X", "Y")}).to_csv(KEY, index=False)
        print(
            f"\nReading sheet: {SHEET} ({len(pick)} of Llama's {len(wrong)} mistakes, 12 categories)")
        print(
            "It holds real answers: open it only on your laptop, never commit or paste it.")
        print("Fill in 'verdict' (X, Y, B or N) and 'why' (1-4), save, then run: python scripts\\error_analysis.py --count")
    print(
        f"\nSaved results/confusion_llama_{SPLIT}_12.csv and _50.csv (counts only)")

else:
    sheet = pd.read_csv(SHEET, dtype=str, keep_default_na=False, sep=None, engine="python",
                        encoding="utf-8-sig", encoding_errors="replace")
    df = sheet.merge(pd.read_csv(KEY, dtype=str), on="n")
    # A verdict can be a letter (X, Y, B, N) or the name of the code you chose, e.g. "Health"
    v = df["verdict"].str.strip().str.lower()
    df["verdict"] = np.select([v == df["code_X"].str.strip().str.lower(), v == df["code_Y"].str.strip().str.lower(),
                               v.isin(["x", "y", "b", "n"]), v == "both", v == "neither"],
                              ["X", "Y", v.str.upper(), "B", "N"], "")
    df["why"] = df["why"].str.strip()
    bad_verdict = df.loc[df["verdict"] == "", "n"]
    bad_why = df.loc[~df["why"].isin(list(WHY)), "n"]
    if len(bad_verdict) or len(bad_why):
        if len(bad_verdict):
            print(
                f"Rows {', '.join(bad_verdict)}: verdict must be X, Y, B, N or the name of the code you chose")
        if len(bad_why):
            print(f"Rows {', '.join(bad_why)}: why must be a single number from 1 to 4"
                  " (keep your own reasons in a separate 'note' column)")
        raise SystemExit("Fix those cells, save, and run this again.")
    # Unblind: was the code you picked the BES code or Llama's?
    df["result"] = np.select([df["verdict"] == "B", df["verdict"] == "N", df["verdict"] == df["bes_is"]],
                             ["both fine", "neither", "BES right"], "Llama right")
    df["why"] = df["why"].map(WHY)
    order = ["BES right", "Llama right", "both fine", "neither"]
    counts = df["result"].value_counts().reindex(order, fill_value=0)
    print(
        f"Blind review of {len(df)} of Llama's 12-category mistakes ({SPLIT} set)")
    for result, k in counts.items():
        print(f"  {result:<13}{k:>4}  ({k / len(df):.0%})")
    print("\nWhy the codes differ, by verdict:")
    table = pd.crosstab(df["why"], df["result"]).reindex(
        columns=order, fill_value=0)
    print(table.to_string())
    out = {"split": SPLIT, "scheme": "12", "reviewed": len(df), "verdicts": counts.to_dict(),
           "why": df["why"].value_counts().to_dict(),
           "why_by_verdict": {w: row.to_dict() for w, row in table.iterrows()}}
    with open(f"results/error_review_{SPLIT}_12.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved results/error_review_{SPLIT}_12.json (counts only)")
