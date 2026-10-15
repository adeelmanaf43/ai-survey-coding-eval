"""Two baselines the AI model has to beat, and the seen vs new split (D-011).
Prints and saves numbers only (no answers), so the output is safe to share and commit."""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from .config import CONFIG, PROCESSED, ROOT
from .data import eval_set, save_json
from .metrics import scores


def normalise(text):
    """Lower case, punctuation removed: used to decide whether an answer was seen word for word."""
    return text.astype(str).str.lower().str.replace(r"[\W_]", " ", regex=True).str.strip()


def training_answers():
    """Human-coded answers from the two waves before wave 25, one row per answer."""
    import pyreadstat

    paths, waves = CONFIG["paths"], CONFIG["baseline"]["train_waves"]
    strings_file, panel_file = str(
        ROOT / paths["strings"]), str(ROOT / paths["panel"])
    _, smeta = pyreadstat.read_sav(strings_file, metadataonly=True)
    _, pmeta = pyreadstat.read_sav(panel_file, metadataonly=True)
    text_cols = [f"MII_textW{w}" for w in waves]
    code_cols = [f"{c}W{w}" for w in waves for c in (
        "mii_cat", "small_mii_cat")]
    missing = [c for c in text_cols if c not in smeta.column_names] + \
        [c for c in code_cols if c not in pmeta.column_names]
    if missing:
        raise SystemExit(f"Columns not found: {missing}")
    # category names from wave 25, so every wave is spelled the same
    wave = CONFIG["eval_set"]["wave"]
    names50 = {
        float(k): v for k, v in pmeta.variable_value_labels[f"mii_catW{wave}"].items()}
    names12 = {float(
        k): v for k, v in pmeta.variable_value_labels[f"small_mii_catW{wave}"].items()}

    print("Reading training waves (the panel file can take a minute) ...")
    strings, _ = pyreadstat.read_sav(strings_file, usecols=["id"] + text_cols)
    panel, _ = pyreadstat.read_sav(panel_file, usecols=["id"] + code_cols)
    both = strings.merge(panel, on="id")
    train = pd.concat([pd.DataFrame({"id": both["id"],
                                     "text": both[f"MII_textW{w}"].fillna("").astype(str).str.strip(),
                                     "label50": both[f"mii_catW{w}"].map(names50),
                                     "label12": both[f"small_mii_catW{w}"].map(names12)})
                       for w in waves], ignore_index=True)
    train = train[train["text"].str.contains(
        r"\w", regex=True) & train["label50"].notna()].copy()
    train.loc[train["label50"].str.lower().eq(
        "uncoded") & train["label12"].isna(), "label12"] = "Uncoded"
    return train


def train_and_predict(split="test"):
    """Train on waves 23-24, code one split, and save the results and the predictions."""
    train = training_answers()
    # Leave out everyone in the evaluation set (dev and test), so the classifier never
    # sees their earlier answers (D-010)
    evaluation_ids = set(eval_set("dev")["id"]) | set(eval_set("test")["id"])
    left_out = train["id"].isin(evaluation_ids)
    print(
        f"Left out {left_out.sum():,} training answers from {train.loc[left_out, 'id'].nunique():,}"
        f"of the {len(evaluation_ids):,} evaluation respondents")
    train = train[~left_out]
    evalset = eval_set(split)

    rows = []
    for scheme, n_cat in (("label12", 12), ("label50", 50)):
        data = train.dropna(subset=[scheme])
        # Baseline 1: always guess the most common category in the training data
        most_common = data[scheme].value_counts().idxmax()
        rows.append({"scheme": n_cat, "method": f"Always {most_common}",
                     **scores(evalset[scheme], [most_common] * len(evalset))})
        # Baseline 2: TF-IDF on small letter chunks (handles typos) + logistic regression with default settings
        model = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True),
                              LogisticRegression(max_iter=2000))
        model.fit(data["text"], data[scheme])
        evalset[f"pred_{scheme}"] = model.predict(evalset["text"])
        evalset[f"conf_{scheme}"] = model.predict_proba(
            evalset["text"]).max(axis=1)   # how sure it is (0 to 1)
        rows.append({"scheme": n_cat, "method": "TF-IDF + logistic regression",
                     **scores(evalset[scheme], evalset[f"pred_{scheme}"])})

    print(f"\nResults on the {split} set ({len(evalset)} answers)")
    print(f"{'Scheme':<8}{'Method':<34}{'Accuracy':>9}{'Macro-F1':>10}{'Kappa':>7}")
    for r in rows:
        print(
            f"{r['scheme']:<8}{r['method']:<34}{r['accuracy']:>9.1%}{r['macro_f1']:>10.1%}{r['kappa']:>7.2f}")

    # Seen or new? Does the exact answer (ignoring case and punctuation) appear in the training data
    seen = normalise(evalset["text"]).isin(set(normalise(train["text"])))
    seen_vs_new = {"seen_share": round(seen.mean(), 3)}
    print(
        f"\n Seen in training, word for word: {seen.sum()} of {len(evalset)} answers ({seen.mean():.0%})")
    for scheme, n_cat in (("label12", 12), ("label50", 50)):
        right = evalset[scheme] == evalset[f"pred_{scheme}"]
        seen_vs_new[f"{n_cat}_cat"] = {"seen": round(
            right[seen].mean(), 3), "new": round(right[~seen].mean(), 3)}
        print(f"   {n_cat} categories, classifier accuracy: seen {right[seen].mean():.1%} | "
              f"new {right[~seen].mean():.1%} ({(~seen).sum()} new answers)")

    save_json(f"baseline_{split}.json", {"split": split, "n_eval": len(evalset), "train_waves": CONFIG["baseline"]["train_waves"],
                                         "n_train": len(train), "results": rows, "seen_vs_new": seen_vs_new})
    evalset["seen"] = seen
    evalset[["id", "seen", "label12", "pred_label12", "conf_label12", "label50", "pred_label50", "conf_label50"]].to_csv(
        PROCESSED / f"baseline_pred_{split}.csv", index=False)
