"""Review rule: a person checks the riskiest answers first. How accurate are the answers
left to the machine, and what does the checking cost per 1,000 answers? Saves a chart and numbers only."""
import numpy as np
import pandas as pd

from .config import CONFIG, RESULTS, SCHEMES
from .data import classifier_codes, eval_set, llm_codes, llm_path, load_json, save_json
from .metrics import mean_seconds, review_curve


def review_rule(split="test"):
    """results/review_rule.json and results/review_rule.png"""
    hourly, factor = CONFIG["cost"]["hourly_gbp"], CONFIG["cost"]["hard_case_factor"]
    timeout = CONFIG["llm"]["timeout_seconds"]
    truth, base = eval_set(split), classifier_codes(split)
    # your average time per answer (D-019)
    human_seconds = load_json("human_benchmark.json")["seconds_mean"]

    # Llama's laptop time per answer: the mean, because cost depends on total time (D-019).
    # A call gives up after the timeout, so a longer time means the laptop was asleep
    # during that answer, not working: leave those answers out (D-022)
    llm_seconds, llm_left_out = {}, {}
    for s in SCHEMES:
        sec = pd.read_csv(llm_path(split, s))["seconds"]
        llm_seconds[f"{s}_cat"], llm_left_out[f"{s}_cat"] = mean_seconds(
            sec, timeout)

    results = {"assumptions": {"person_checking_gives_the_BES_code": True, "human_seconds_per_answer": human_seconds,
                               "hourly_cost_gbp": hourly, "hard_case_factor": factor,
                               f"llm_answers_left_out_over_{timeout}_s": llm_left_out}}
    curves = {}
    for scheme in SCHEMES:
        df = (truth[["id", f"label{scheme}"]].merge(llm_codes(split, scheme)[["id", "pred", "unsure"]], on="id")
              .merge(base[["id", f"pred_label{scheme}", f"conf_label{scheme}"]], on="id"))
        # The rule: a person checks every answer where Llama and the classifier disagree, or Llama is unsure
        df["check"] = (df["pred"] != df[f"pred_label{scheme}"]) | (
            df["unsure"].astype(str) == "True")
        # Riskiest first: answers the rule flags, then the rest from the least to the most confident classifier (D-021)
        df = df.sort_values(
            ["check", f"conf_label{scheme}"], ascending=[False, True])
        # Llama's code (= the agreed code once they agree)
        right = (df["pred"] == df[f"label{scheme}"]).to_numpy()
        n = len(df)
        # left to the machine; whole dataset if every check fixes the answer
        auto, overall = review_curve(right)
        rule = int(df["check"].sum())
        curves[scheme] = (np.arange(n) / n, auto, rule)

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
        f"\nCost per 1,000 answers (your pace {human_seconds:.0f} s per answer, GBP {hourly} per hour assumed)")
    hand_hours = 1000 * human_seconds / 3600
    print(
        f"  Code everything by hand: {hand_hours:.1f} h of human time, about GBP {hand_hours * hourly:.0f}")
    results["cost_per_1000"] = {"all_by_hand": {"human_hours": round(
        hand_hours, 2), "gbp": round(hand_hours * hourly, 2)}}
    for scheme in SCHEMES:
        share = results[f"{scheme}_cat"]["rule"]["checked_share"]
        minutes = 1000 * share * human_seconds / 60
        laptop = 1000 * llm_seconds[f"{scheme}_cat"] / 3600
        low, high = minutes / 60 * hourly, minutes * factor / 60 * hourly
        print(f"  {scheme} categories with the rule: {minutes:.0f}-{minutes * factor:.0f} min of human time,"
              f" about GBP {low:.0f}-{high:.0f}, plus {laptop:.1f} h of unattended laptop time")
        results["cost_per_1000"][f"rule_{scheme}_cat"] = {"human_minutes": [round(minutes, 1), round(minutes * factor, 1)],
                                                          "gbp": [round(low, 2), round(high, 2)],
                                                          "laptop_hours": round(laptop, 2)}
    draw_chart(curves, len(truth))
    save_json("review_rule.json", results)


def draw_chart(curves, n_answers):
    """The chart (colours from a colour-blind-checked palette; text stays in grey/black ink)."""
    import matplotlib
    # draw straight to a file, no window
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

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
    ax.set_ylabel("Accuracy of the answers left to the machine",
                  color=SECONDARY)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, length=0)
    ax.legend(frameon=False, loc="lower right", labelcolor=INK)
    ax.set_title("Checking about 5% of answers leaves the rest about 99% right", loc="left",
                 color=INK, fontsize=12, fontweight="bold", pad=24)
    ax.text(0, 1.02, f"Llama 3.1 8B on {n_answers:,} British Election Study test answers; riskiest answers checked first",
            transform=ax.transAxes, color=SECONDARY, fontsize=9)
    fig.tight_layout()
    RESULTS.mkdir(exist_ok=True)
    fig.savefig(RESULTS / "review_rule.png", facecolor=SURFACE)
    plt.close(fig)
    print("Saved results/review_rule.png")
