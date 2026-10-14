"""Every file in results/ from the saved codes; run_all() recomputes them in one go.
Prints and saves numbers and category names only, so the output is safe to share and commit.
The one exception is review_sheet(): it writes answer text, so it saves to data/processed (git-ignored)."""
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score

from .config import CONFIG, PROCESSED, RESULTS, ROOT, SCHEMES, SEED
from .data import classifier_codes, eval_set, human_codes, llm_codes, llm_path, load_json, save_json
from .metrics import bootstrap, gap, mcnemar_exact, scores

LLAMA, CLASSIFIER = "Llama 3.1 8B", "TF-IDF classifier"
# the blind reading sheet: real answers, never commit or paste it
SHEET = PROCESSED / "errors_to_read.csv"
# which code is the BES one: don't open it
KEY = PROCESSED / "errors_key.csv"
WHY = {"1": "several issues", "2": "vague answer",
       "3": "overlapping categories", "4": "clear error"}


# ---------- Llama's scores (H7) ----------

def _line(scheme, method, s, new=None):
    new_txt = f"{new:>12.1%}" if new is not None else f"{'':>12}"
    print(
        f"{scheme:<8}{method:<34}{s['accuracy']:>9.1%}{s['macro_f1']:>10.1%}{s['kappa']:>7.2f}{new_txt}")


def llm_scores(split):
    """Llama's codes against the BES codes, next to the baselines: results/llm_{split}.json."""
    truth = eval_set(split)
    seen = classifier_codes(split)[["id", "seen"]]
    baseline_file = RESULTS / f"baseline_{split}.json"
    base = load_json(baseline_file.name) if baseline_file.exists() else {
        "results": []}   # for printing only

    summary, details = {}, []
    print(f"Results on the {split} set")
    print(f"{'Scheme':<8}{'Method':<34}{'Accuracy':>9}{'Macro-F1':>10}{'Kappa':>7}{'New answers':>12}")
    for scheme in SCHEMES:
        df = truth.merge(llm_codes(split, scheme), on="id").merge(
            seen, on="id", how="left")
        label = f"label{scheme}"
        right = df[label] == df["pred"]
        # not in the classifier's training data, word for word
        is_new = df["seen"].astype(str) == "False"
        unsure = df["unsure"].astype(str) == "True"

        for r in base["results"]:
            if str(r["scheme"]) == scheme:
                tfidf = "TF-IDF" in r["method"]
                _line(scheme, r["method"], r, base["seen_vs_new"]
                      [f"{scheme}_cat"]["new"] if tfidf else None)
        s = scores(df[label], df["pred"])
        s["new"] = round(right[is_new].mean(), 3)
        _line(scheme, "Llama 3.1 8B, codebook only", s, s["new"])

        s.update({"n_coded": len(df), "unsure_share": round(unsure.mean(), 3),
                  "accuracy_when_unsure": round(right[unsure].mean(), 3),
                  "accuracy_when_sure": round(right[~unsure].mean(), 3),
                  "median_seconds": round(df["seconds"].median(), 1),
                  "invalid": int((df["pred"] == "INVALID").sum())})
        mistakes = df[~right].groupby(
            [label, "pred"]).size().sort_values(ascending=False).head(8)
        s["top_mistakes"] = [{"human": h, "llama": p, "count": int(n)} for (
            h, p), n in mistakes.items()]
        summary[f"{scheme}_cat"] = s
        details.append((scheme, s))

    for scheme, s in details:
        print(f"\n{scheme} categories: coded {s['n_coded']} of {len(truth)} answers | median {s['median_seconds']} s"
              f" per answer | invalid replies: {s['invalid']}")
        print(f"  Llama flagged {s['unsure_share']:.0%} as unsure: accuracy {s['accuracy_when_unsure']:.0%} when unsure,"
              f" {s['accuracy_when_sure']:.0%} when sure")
        print("  Most common mistakes (human code -> Llama's code):")
        for m in s["top_mistakes"]:
            print(f"    {m['human']} -> {m['llama']}: {m['count']}")
    save_json(f"llm_{split}.json", {"split": split,
              "model": CONFIG["llm"]["model"], "results": summary})


# ---------- Second opinion (H7) ----------

def second_opinion(split):
    """If a person checks only the answers where Llama and the classifier disagree, how much checking
    does that take, and how accurate is the final result? (D-014) results/second_opinion_{split}.json"""
    truth, base = eval_set(split), classifier_codes(split)
    summary = {}
    for scheme in SCHEMES:
        df = (truth[["id", f"label{scheme}"]]
              .merge(llm_codes(split, scheme)[["id", "pred"]], on="id")
              .merge(base[["id", f"pred_label{scheme}"]], on="id"))
        human, llama, clf = df[f"label{scheme}"], df["pred"], df[f"pred_label{scheme}"]
        agree = llama == clf
        llama_right = llama == human
        clf_right = clf == human
        # Assumption: the person who checks a disagreement gives the same code as the BES coder (best case)
        final_right = (agree & llama_right) | ~agree

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

        print(f"{scheme} categories ({s['n']} answers)")
        print(
            f"  Agree on {s['agree']} answers ({s['agree'] / s['n']:.0%}): right {s['accuracy_when_agree']:.1%} of the time")
        print(f"  Disagree on {s['disagree']} answers ({s['disagree'] / s['n']:.0%}): Llama right"
              f" {s['llama_right_when_disagree']:.0%}, classifier right {s['classifier_right_when_disagree']:.0%}")
        print(f"  Llama's {s['llama_errors']} mistakes: {s['llama_errors_caught']} among the disagreements (a person would"
              f" see them), {s['llama_errors'] - s['llama_errors_caught']} where both methods agree (missed)")
        print(f"  Accuracy: Llama alone {s['accuracy_llama_alone']:.1%} | with a person checking the {s['disagree']}"
              f" disagreements {s['accuracy_with_review']:.1%}")
    save_json(f"second_opinion_{split}.json",
              {"split": split, "assumption": "a person checking a disagreement gives the BES coder's code",
               "results": summary})


# ---------- Prompt A vs prompt B (H5, D-023) ----------

def prompt_comparison(split="dev"):
    """Prompt A (definitions only) vs prompt B (definitions plus examples, the frozen prompt).
    Both coded the same answers, so the comparison is paired: only answers where exactly one prompt
    is right can show a difference (McNemar's exact test). results/prompt_comparison_{split}.json"""
    frozen = CONFIG["llm"]["frozen_versions"]
    truth = eval_set(split)
    results = {"split": split}
    print(
        f"Prompt A (definitions only) vs prompt B (definitions plus examples), {split} set")
    for scheme in SCHEMES:
        df = truth[["id", f"label{scheme}"]].rename(
            columns={f"label{scheme}": "human"})
        for name in ("A", "B"):
            run = llm_codes(split, scheme, prompt=name).drop_duplicates(
                "id", keep="last")
            run = run[["id", "pred", "seconds", "prompt_tokens", "version"]]
            df = df.merge(run.rename(columns=lambda c: c if c ==
                          "id" else f"{c}_{name}"), on="id")

        out = {"answers": len(df)}
        print(
            f"\n{scheme} categories: {len(df)} answers coded by both prompts (of {len(truth)})")
        print(f"  {'Prompt':<8}{'Version':<10}{'Accuracy':>9}{'Macro-F1':>10}{'Kappa':>7}{'Typical s':>11}"
              f"{'Prompt tokens':>15}")
        for name in ("A", "B"):
            p = df[f"pred_{name}"]
            out[name] = {"version": ", ".join(sorted(df[f"version_{name}"].astype(str).unique())),
                         **scores(df["human"], p),
                         "median_seconds": round(pd.to_numeric(df[f"seconds_{name}"]).median(), 1),
                         "median_prompt_tokens": int(pd.to_numeric(df[f"prompt_tokens_{name}"]).median()),
                         "invalid": int((p == "INVALID").sum())}
            r = out[name]
            print(f"  {name:<8}{r['version']:<10}{r['accuracy']:>9.1%}{r['macro_f1']:>10.1%}{r['kappa']:>7.2f}"
                  f"{r['median_seconds']:>11.1f}{r['median_prompt_tokens']:>15,}")
        if out["B"]["version"] != frozen[scheme]:
            print(
                f"  [!] prompt B rows have version {out['B']['version']}, not the frozen {frozen[scheme]}")

        right_a, right_b = df["pred_A"] == df["human"], df["pred_B"] == df["human"]
        only_b, only_a = int((right_b & ~right_a).sum()), int(
            (right_a & ~right_b).sum())
        out.update({"same_code_share": round((df["pred_A"] == df["pred_B"]).mean(), 3),
                    "only_B_right": only_b, "only_A_right": only_a,
                    "mcnemar_exact_p": round(mcnemar_exact(only_b, only_a), 3)})
        print(
            f"  Same code from both prompts: {out['same_code_share']:.0%} of answers")
        print(
            f"  Only B right: {only_b} | only A right: {only_a} | McNemar exact p = {out['mcnemar_exact_p']:.3f}")

        # Where the examples changed the outcome (human code -> the wrong prompt's code)
        for key, mask, wrong in (("helped", right_b & ~right_a, "pred_A"), ("hurt", right_a & ~right_b, "pred_B")):
            pairs = df[mask].groupby(["human", wrong]).size(
            ).sort_values(ascending=False).head(5)
            out[f"examples_{key}"] = [
                {"human": h, "wrong_code": w, "count": int(n)} for (h, w), n in pairs.items()]
            when = "B right, A wrong" if key == "helped" else "A right, B wrong"
            print(f"  Where the examples {key} ({when}): "
                  + ("; ".join(f"{h} -> {w} ({n})" for (h, w), n in pairs.items()) or "none"))
        results[f"{scheme}_cat"] = out
    save_json(f"prompt_comparison_{split}.json", results)


# ---------- Human benchmark (H8) ----------

def human_benchmark(split="test"):
    """Your blind codes against the BES coders, next to Llama and the classifier on the SAME answers (D-018).
    results/human_benchmark.json"""
    truth = eval_set(split)
    df = (human_codes().merge(truth[["id", "label12"]], on="id")
          .merge(llm_codes(split, "12")[["id", "pred"]], on="id")
          .merge(classifier_codes(split)[["id", "pred_label12"]], on="id"))

    print(f"Answers you coded: {len(df)}\n")
    methods = {}
    for name, col in [("You", "human"), (LLAMA, "pred"), (CLASSIFIER, "pred_label12")]:
        right = df[col] == df["label12"]
        kappa = cohen_kappa_score(df["label12"], df[col])
        methods[name] = {"right": int(right.sum()), "agreement": round(
            right.mean(), 3), "kappa": round(kappa, 3)}
        print(f"  {name:<18} agrees with the BES coders on {right.sum()} of {len(df)} ({right.mean():.0%}), kappa {kappa:.2f}")
    same_as_llama = (df["human"] == df["pred"]).mean()
    print(
        f"  You and Llama chose the same code for {same_as_llama:.0%} of these answers")

    sec = df["seconds"]
    # cost depends on total time, so use the mean (D-019)
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

    save_json("human_benchmark.json",
              {"scheme": 12, "sample": f"random {len(df)} of the {len(truth):,} {split} answers (seed {SEED})",
               "n": len(df), "methods": methods, "same_code_as_llama": round(same_as_llama, 3),
               "seconds_median": round(sec.median(), 1), "seconds_mean": round(sec.mean(), 1),
               "hours_per_1000_answers": round(per_1000, 1),
               "differences": [{"bes": b, "you": y, "count": int(n)} for (b, y), n in diff.items()]})


# ---------- 95% confidence intervals (H9) ----------

def _show(name, r):
    a, f, k = r["accuracy"], r["macro_f1"], r["kappa"]
    print(f"  {name:<18} {a[0]:6.1%} ({a[1]:.1%}-{a[2]:.1%})   {f[0]:6.1%} ({f[1]:.1%}-{f[2]:.1%})"
          f"   {k[0]:.2f} ({k[1]:.2f}-{k[2]:.2f})")


def confidence_intervals(split="test"):
    """Every metric with a 95% bootstrap interval, F1 per category, and your codes next to both methods.
    Bootstrap: draw 1,000 new samples of the same size from the answers (with replacement), recompute every
    metric on each, and keep the middle 95% of the values as the interval (D-020). results/metrics_{split}.json"""
    truth, base = eval_set(split), classifier_codes(split)
    resamples = CONFIG["bootstrap"]["resamples"]
    results = {"bootstrap": {"resamples": resamples,
                             "seed": SEED, "interval": "95%, percentile"}}
    print(f"{split.capitalize()} set, {len(truth):,} answers. 95% intervals from {resamples:,} bootstrap resamples"
          " (this takes a minute or two)")

    for scheme in SCHEMES:
        df = (truth[["id", f"label{scheme}"]].merge(llm_codes(split, scheme)[["id", "pred"]], on="id")
              .merge(base[["id", f"pred_label{scheme}"]], on="id"))
        t = df[f"label{scheme}"]
        preds = {CLASSIFIER: df[f"pred_label{scheme}"], LLAMA: df["pred"]}
        out, draws = bootstrap(t, preds, resamples, SEED)
        point = out[CLASSIFIER]["accuracy"][0] - out[LLAMA]["accuracy"][0]
        out["classifier_minus_llama_accuracy_points"] = gap(
            draws, CLASSIFIER, LLAMA, point)

        # F1 per category: where each method does well or badly
        cats = t.value_counts()
        f1c = f1_score(t, preds[CLASSIFIER],
                       labels=cats.index, average=None, zero_division=0)
        f1l = f1_score(t, preds[LLAMA], labels=cats.index,
                       average=None, zero_division=0)
        out["per_category_f1"] = [{"category": c, "n": int(n), "classifier": round(a, 3), "llama": round(b, 3)}
                                  for c, n, a, b in zip(cats.index, cats.values, f1c, f1l)]
        results[f"{scheme}_cat"] = out

        print(
            f"\n{scheme} categories          Accuracy                Macro-F1                Kappa")
        _show(CLASSIFIER, out[CLASSIFIER])
        _show(LLAMA, out[LLAMA])
        g = out["classifier_minus_llama_accuracy_points"]
        print(
            f"  Classifier minus Llama: {g[0]:+.1f} points of accuracy (95% interval {g[1]:+.1f} to {g[2]:+.1f})")

    print("\nF1 per category, 12 categories (n = answers in the category)")
    print(f"  {'Category':<20}{'n':>5}{'Classifier':>12}{'Llama':>8}")
    for r in results["12_cat"]["per_category_f1"]:
        print(
            f"  {r['category']:<20}{r['n']:>5}{r['classifier']:>12.2f}{r['llama']:>8.2f}")

    # The answers you coded yourself (H8), next to both methods on exactly those answers
    if (PROCESSED / "human_test_12.csv").exists() and split == "test":
        h = (human_codes().merge(truth[["id", "label12"]], on="id")
             .merge(llm_codes(split, "12")[["id", "pred"]], on="id")
             .merge(base[["id", "pred_label12"]], on="id"))
        out, draws = bootstrap(h["label12"], {"You": h["human"], LLAMA: h["pred"], CLASSIFIER: h["pred_label12"]},
                               resamples, SEED)
        point = out[LLAMA]["accuracy"][0] - out["You"]["accuracy"][0]
        out["llama_minus_you_accuracy_points"] = gap(
            draws, LLAMA, "You", point)
        results["human_sample_12_cat"] = dict(n=len(h), **out)
        print(
            f"\nThe {len(h)} answers you coded, 12 categories   Accuracy             Macro-F1             Kappa")
        for name in ("You", LLAMA, CLASSIFIER):
            _show(name, out[name])
        g = out["llama_minus_you_accuracy_points"]
        print(
            f"  Llama minus you: {g[0]:+.1f} points of accuracy (95% interval {g[1]:+.1f} to {g[2]:+.1f})")
    save_json(f"metrics_{split}.json", results)


# ---------- Error analysis (H6, D-025) ----------

def _llama_vs_bes(truth, split, scheme):
    return truth[["id", "text", f"label{scheme}"]].rename(columns={f"label{scheme}": "bes"}).merge(
        llm_codes(split, scheme)[["id", "pred"]], on="id")


def confusion_matrices(split="test"):
    """One row per BES code, one column per Llama code, counts in the cells:
    results/confusion_llama_{split}_{scheme}.csv"""
    truth = eval_set(split)
    RESULTS.mkdir(exist_ok=True)
    print(f"Llama's mistakes on the {split} set (BES code -> Llama's code)")
    for scheme in SCHEMES:
        df = _llama_vs_bes(truth, split, scheme)
        pd.crosstab(df["bes"].rename("BES code"), df["pred"].rename("Llama's code")).to_csv(
            RESULTS / f"confusion_llama_{split}_{scheme}.csv")
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
    print("Saved " + " and ".join(
        f"results/confusion_llama_{split}_{s}.csv" for s in SCHEMES) + " (counts only)")


def review_sheet(split="test"):
    """The blind reading sheet: random 12-category mistakes, with the BES code and Llama's code in random
    order as X and Y, so you can't tell which is which. It holds answer text: data/processed only."""
    if SHEET.exists():
        print(
            f"{SHEET.relative_to(ROOT)} already exists, not overwritten (delete it to start again)")
        return
    df = _llama_vs_bes(eval_set(split), split, "12")
    wrong = df[df["bes"] != df["pred"]]
    pick = wrong.sample(
        n=min(CONFIG["review"]["sheet_size"], len(wrong)), random_state=SEED)
    bes_is_x = np.random.default_rng(SEED).random(len(pick)) < 0.5
    rows = range(1, len(pick) + 1)
    pd.DataFrame({"n": rows, "answer": pick["text"].to_numpy(),
                  "code_X": np.where(bes_is_x, pick["bes"], pick["pred"]),
                  "code_Y": np.where(bes_is_x, pick["pred"], pick["bes"]),
                  "verdict": "", "why": ""}).to_csv(SHEET, index=False, encoding="utf-8-sig")
    pd.DataFrame({"n": rows, "bes_is": np.where(
        bes_is_x, "X", "Y")}).to_csv(KEY, index=False)
    print(
        f"Reading sheet: {SHEET.relative_to(ROOT)} ({len(pick)} of Llama's {len(wrong)} mistakes, 12 categories)")
    print("It holds real answers: open it only on your laptop, never commit or paste it.")
    print("Fill in 'verdict' (X, Y, B or N) and 'why' (1-4), save, then run: python -m coding_eval metrics")


def review_counts(split="test"):
    """Count your verdicts on the filled-in reading sheet and unblind them with the key:
    results/error_review_{split}_12.json. Returns False (and saves nothing) if the sheet isn't ready."""
    sheet = pd.read_csv(SHEET, dtype=str, keep_default_na=False, sep=None, engine="python",
                        encoding="utf-8-sig", encoding_errors="replace")
    if (sheet["verdict"].str.strip() == "").all():
        print("The reading sheet isn't filled in yet: fill it in, then run this again.")
        return False
    df = sheet.merge(pd.read_csv(KEY, dtype=str), on="n")
    # A verdict can be a letter (X, Y, B, N) or the name of the code you chose, e.g. "Health"
    v = df["verdict"].str.strip().str.lower()
    df["verdict"] = np.select([v == df["code_X"].str.strip().str.lower(), v == df["code_Y"].str.strip().str.lower(),
                               v.isin(["x", "y", "b", "n"]), v == "both", v == "neither"],
                              ["X", "Y", v.str.upper(), "B", "N"], "")
    df["why"] = df["why"].str.strip().str[:1]
    bad_verdict = df.loc[df["verdict"] == "", "n"]
    bad_why = df.loc[~df["why"].isin(list(WHY)), "n"]
    if len(bad_verdict) or len(bad_why):
        if len(bad_verdict):
            print(
                f"Rows {', '.join(bad_verdict)}: verdict must be X, Y, B, N or the name of the code you chose")
        if len(bad_why):
            print(f"Rows {', '.join(bad_why)}: why must be a single number from 1 to 4"
                  " (keep your own reasons in a separate 'note' column)")
        print("Fix those cells, save, and run this again.")
        return False

    # Unblind: was the code you picked the BES code or Llama's?
    df["result"] = np.select([df["verdict"] == "B", df["verdict"] == "N", df["verdict"] == df["bes_is"]],
                             ["both fine", "neither", "BES right"], "Llama right")
    df["why"] = df["why"].map(WHY)
    order = ["BES right", "Llama right", "both fine", "neither"]
    counts = df["result"].value_counts().reindex(order, fill_value=0)
    print(
        f"Blind review of {len(df)} of Llama's 12-category mistakes ({split} set)")
    for result, k in counts.items():
        print(f"  {result:<13}{k:>4}  ({k / len(df):.0%})")
    print("\nWhy the codes differ, by verdict:")
    table = pd.crosstab(df["why"], df["result"]).reindex(
        columns=order, fill_value=0)
    print(table.to_string())
    save_json(f"error_review_{split}_12.json",
              {"split": split, "scheme": "12", "reviewed": len(df), "verdicts": counts.to_dict(),
               "why": df["why"].value_counts().to_dict(),
               "why_by_verdict": {w: row.to_dict() for w, row in table.iterrows()}})
    return True


# ---------- Everything at once ----------

def run_all():
    """Recompute every file in results/ from the saved codes. A step whose input files are missing
    is skipped with a note, so the command also runs on a fresh clone (where it skips everything)."""
    from .review_rule import review_rule      # imported here: it loads matplotlib, which is slow to import

    if not PROCESSED.exists():
        print(f"No {PROCESSED.relative_to(ROOT).as_posix()} folder, so there is nothing to recompute. The codes in it"
              " come from British Election Study answers, which stay on the author's laptop (see the README).")
        return
    done, skipped = [], []

    def step(title, needs, function, *args):
        missing = [p.relative_to(ROOT).as_posix()
                   for p in needs if not p.exists()]
        if missing:
            skipped.append(f"{title}: missing {', '.join(missing)}")
            return
        print(f"\n=== {title} " + "=" * max(3, 66 - len(title)))
        if function(*args) is False:
            skipped.append(f"{title}: not ready (see above)")
        else:
            done.append(title)

    def inputs(split):
        """The evaluation set, the classifier's codes and Llama's codes (prompt B) for one split."""
        return PROCESSED / f"eval_{split}.csv", PROCESSED / f"baseline_pred_{split}.csv", [llm_path(split, s) for s in SCHEMES]

    for split in ("dev", "test"):
        truth, classifier, llama = inputs(split)
        step(f"Llama's scores, {split} set", [
             truth, classifier, *llama], llm_scores, split)
        step(f"Second opinion, {split} set", [
             truth, classifier, *llama], second_opinion, split)
        prompt_a = [llm_path(split, s, prompt="A") for s in SCHEMES]
        # prompt A was run on dev only (D-024)
        if split == "dev" or all(p.exists() for p in prompt_a):
            step(f"Prompt A vs B, {split} set", [
                 truth, *llama, *prompt_a], prompt_comparison, split)

    # the rest is about the test set only
    truth, classifier, llama = inputs("test")
    step("Human benchmark", [truth, classifier, *llama,
         PROCESSED / "human_test_12.csv"], human_benchmark)
    step("95% confidence intervals", [
         truth, classifier, *llama], confidence_intervals)
    step("Review rule and cost", [
         truth, classifier, *llama, RESULTS / "human_benchmark.json"], review_rule)
    step("Confusion matrices", [truth, *llama], confusion_matrices)
    step("Blind review counts", [truth, *llama, SHEET, KEY], review_counts)

    print("\n" + "=" * 72 +
          f"\nDone: {len(done)} steps recomputed." + (" Skipped:" if skipped else ""))
    for s in skipped:
        print(f"  {s}")
