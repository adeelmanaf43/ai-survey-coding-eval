# Review rule (roadmap H10): a person checks the riskiest answers first. How accurate are the answers
# left to the machine, and what does it cost per 1,000 answers? Saves a chart and numbers only.
from matplotlib.ticker import PercentFormatter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

import matplotlib
matplotlib.use("Agg")                     # draw straight to a file, no window

SPLIT = "test"
# assumption: cost of one hour of a human coder. Change it to your own figure
HOURLY_COST_GBP = 15
# assumption for a range: checking a disagreement may take 3x an average answer
HARD_FACTOR = 3

truth = pd.read_csv(f"data/processed/eval_{SPLIT}.csv", keep_default_na=False)
base = pd.read_csv(
    f"data/processed/baseline_pred_{SPLIT}.csv", keep_default_na=False)
with open("results/human_benchmark.json") as f:
    # your average time per answer (D-019)
    human_seconds = json.load(f)["seconds_mean"]

# Llama's laptop time per answer: the mean, because cost depends on total time (D-019)
llm_seconds = {f"{s}_cat": pd.read_csv(f"data/processed/llm_{SPLIT}_{s}.csv")["seconds"].mean()
               for s in ("12", "50")}

results = {"assumptions": {"person_checking_gives_the_BES_code": True, "human_seconds_per_answer": human_seconds,
                           "hourly_cost_gbp": HOURLY_COST_GBP, "hard_case_factor": HARD_FACTOR}}
curves = {}
for scheme in ("12", "50"):
    llm = pd.read_csv(
        f"data/processed/llm_{SPLIT}_{scheme}.csv", keep_default_na=False)
    df = (truth[["id", f"label{scheme}"]].merge(llm[["id", "pred", "unsure"]], on="id")
          .merge(base[["id", f"pred_label{scheme}", f"conf_label{scheme}"]], on="id"))

    # The rule: a person checks every answer where Llama and the classifier disagree, or Llama is unsure
    df["check"] = (df["pred"] != df[f"pred_label{scheme}"]) | (
        df["unsure"].astype(str) == "True")
    # Riskiest first: answers the rule flags, then the rest from the least to the most confident classifier
    df = df.sort_values(
        ["check", f"conf_label{scheme}"], ascending=[False, True])
    # Llama's code (= the agreed code once they agree)
    right = (df["pred"] == df[f"label{scheme}"]).to_numpy()
    n = len(df)
    # number of answers a person has checked
    k = np.arange(n)
    # right answers among those NOT checked yet
    right_left = right[::-1].cumsum()[::-1]
    # accuracy of the answers left to the machine
    auto = right_left / (n - k)
    # whole dataset, if every check fixes the answer
    overall = (right_left + k) / n
    rule = int(df["check"].sum())
    curves[scheme] = (k / n, auto, rule)

    # 0%, 1%, ... 25% checked: the chart as a table
    steps = [round(p / 100 * n) for p in range(0, 26)]
    results[f"{scheme}_cat"] = {
        "rule": {"checked": rule, "checked_share": round(rule / n, 3), "machine_share_accuracy": round(auto[rule], 3),
                 "overall_accuracy_best_case": round(overall[rule], 3)},
        "curve": [{"checked_share": round(s / n, 3), "machine_share_accuracy": round(auto[s], 3),
                   "overall_accuracy_best_case": round(overall[s], 3)} for s in steps]}
    print(f"{scheme} categories: a person checks {rule} of {n:,} answers ({rule / n:.1%}). The other {1 - rule / n:.1%}"
          f" are {auto[rule]:.1%} right (Llama alone: {auto[0]:.1%}). Whole dataset: {overall[rule]:.1%} (best case)")

# Cost per 1,000 answers
print(
    f"\nCost per 1,000 answers (your pace {human_seconds:.0f} s per answer, GBP {HOURLY_COST_GBP} per hour assumed)")
hand_hours = 1000 * human_seconds / 3600
print(
    f"  Code everything by hand: {hand_hours:.1f} h of human time, about GBP {hand_hours * HOURLY_COST_GBP:.0f}")
results["cost_per_1000"] = {"all_by_hand": {"human_hours": round(
    hand_hours, 2), "gbp": round(hand_hours * HOURLY_COST_GBP, 2)}}
for scheme in ("12", "50"):
    share = results[f"{scheme}_cat"]["rule"]["checked_share"]
    minutes = 1000 * share * human_seconds / 60
    laptop = 1000 * llm_seconds[f"{scheme}_cat"] / 3600
    low, high = minutes / 60 * HOURLY_COST_GBP, minutes * \
        HARD_FACTOR / 60 * HOURLY_COST_GBP
    print(f"  {scheme} categories with the rule: {minutes:.0f}-{minutes * HARD_FACTOR:.0f} min of human time,"
          f" about GBP {low:.0f}-{high:.0f}, plus {laptop:.1f} h of unattended laptop time")
    results["cost_per_1000"][f"rule_{scheme}_cat"] = {"human_minutes": [round(minutes, 1), round(minutes * HARD_FACTOR, 1)],
                                                      "gbp": [round(low, 2), round(high, 2)], "laptop_hours": round(laptop, 2)}

# The chart (colours from a colour-blind-checked palette; text stays in grey/black ink)
INK, SECONDARY, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
COLOURS = {"12": "#2a78d6", "50": "#eb6834"}
fig, ax = plt.subplots(figsize=(8, 4.8), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
for scheme, (x, y, rule) in curves.items():
    keep = x <= 0.25
    ax.plot(x[keep], y[keep], color=COLOURS[scheme], lw=2,
            solid_capstyle="round", label=f"{scheme} categories")
    ax.plot(x[rule], y[rule], "o", ms=8, color=COLOURS[scheme],
            mec=SURFACE, mew=2, zorder=3)
    ax.annotate(f"{y[rule]:.1%}", (x[rule], y[rule]), xytext=(8, -16) if scheme == "50" else (-44, 6),
                textcoords="offset points", color=INK, fontsize=9)
x50, y50, rule50 = curves["50"]
ax.annotate("Dots: a person checks only the answers where\nLlama and the classifier disagree",
            xy=(x50[rule50], y50[rule50]), xytext=(0.085, 0.962), color=SECONDARY, fontsize=9,)
ax.set_xlim(0, 0.25)
ax.set_ylim(0.94, 1.002)
ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
ax.set_xlabel("Share of answers a person checks", color=SECONDARY)
ax.set_ylabel("Accuracy of the answers left to the machine", color=SECONDARY)
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(AXIS)
ax.tick_params(colors=MUTED, length=0)
ax.legend(frameon=False, loc="lower right", labelcolor=INK)
ax.set_title("Checking about 5% of answers leaves the rest about 99% right", loc="left",
             color=INK, fontsize=12, fontweight="bold", pad=24)
ax.text(0, 1.02, f"Llama 3.1 8B on {len(truth):,} British Election Study test answers; riskiest answers checked first",
        transform=ax.transAxes, color=SECONDARY, fontsize=9)
fig.tight_layout()
fig.savefig("results/review_rule.png", facecolor=SURFACE)

with open("results/review_rule.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved results/review_rule.png and results/review_rule.json")
