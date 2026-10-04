# Session 10, Hour 2: two baselines the AI model has to beat.
# Prints and saves numbers only (no answers), so the output is safe to share and commit

import json
from pathlib import Path

import pandas as pd
import pyreadstat
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.pipeline import make_pipeline

STRINGS = "data/raw/BES2024_W30Strings_v30.1.sav"
PANEL = "data/raw/BES2024_W31_Panel_v31.05.sav"
# the two human-coded waves just before wave 25
TRAIN_WAVES = [23, 24]
SPLIT = "dev"                 # change to "test" only once, in the final evaluation


# 1. Check the column exist, and take category names from wave 25 so every wave is spelled the same

_, smeta = pyreadstat.read_sav(STRINGS, metadataonly=True)
_, pmeta = pyreadstat.read_sav(PANEL, metadataonly=True)
text_cols = [f"MII_textW{w}" for w in TRAIN_WAVES]
code_cols = [f"{c}W{w}" for w in TRAIN_WAVES for c in (
    "mii_cat", "small_mii_cat")]

missing = [c for c in text_cols if c not in smeta.column_names] + \
    [c for c in code_cols if c not in pmeta.column_names]

if missing:
    raise SystemExit(f"Columns not found: {missing}")

names50 = {float(k): v for k,
           v in pmeta.variable_value_labels["mii_catW25"].items()}
names12 = {float(k): v for k,
           v in pmeta.variable_value_labels["small_mii_catW25"].items()}


# 2. Training answers: text from the strings file + human codes from the panel, one row per answer
print("Reading training waves (the panel file can take a minute) ...")
strings, _ = pyreadstat.read_sav(STRINGS, usecols=["id"] + text_cols)
panel, _ = pyreadstat.read_sav(PANEL, usecols=["id"] + code_cols)

both = strings.merge(panel, on="id")


train = pd.concat([pd.DataFrame({"id": both["id"],
                                 "text": both[f"MII_textW{w}"].fillna("").astype(str).str.strip(),
                                 "label50": both[f"mii_catW{w}"].map(names50),
                                 "label12": both[f"small_mii_catW{w}"].map(names12)})
                   for w in TRAIN_WAVES], ignore_index=True)

train = train[train['text'].str.contains(
    r"\w", regex=True) & train["label50"].notna()].copy()
train.loc[train["label50"].str.lower().eq(
    "uncoded") & train["label12"].isna(), "label12"] = "Uncoded"


# 3. Leave out everyone in our evaluation set, so the model never sees their earlier answers
evalset = pd.read_csv(f"data/processed/eval_{SPLIT}.csv")


def score(truth, pred):
    return {"accuracy": round(accuracy_score(truth, pred), 3),
            "macro_f1": round(f1_score(truth, pred, average="macro", zero_division=0), 3),
            "kappa": round(cohen_kappa_score(truth, pred), 3)}


rows = []
for scheme, n_cat in (("label12", 12), ("label50", 50)):
    data = train.dropna(subset=[scheme])

    # Baseline 1: always guess the most common category in the training data
    most_common = data[scheme].value_counts().idxmax()
    rows.append({"scheme": n_cat, "method": f"Always {most_common}",
                 **score(evalset[scheme], [most_common] * len(evalset))})

    # Baseline2: TF-IDF on small letter chunks (handles typos) + logistic regression default settings
    model = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True),
                          LogisticRegression(max_iter=2000))
    model.fit(data["text"], data[scheme])
    evalset[f"pred_{scheme}"] = model.predict(evalset["text"])
    rows.append({"scheme": n_cat, "method": "TF-IDF + logistic regression",
                 **score(evalset[scheme], evalset[f"pred_{scheme}"])})


# 4. Print and save: numbers to results/ (commit), predictions to data/ (git-ignored, for error analysis later)
print(f"\nResults on the {SPLIT} set ({len(evalset)} answers)")
print(f"{'Scheme':<8}{'Method':<34}{'Accuracy':>9}{'Macro-F1':>10}{'Kappa':>7}")
for r in rows:
    print(f"{r['scheme']:<8}{r['method']:<34}{r['accuracy']:>9.1%}{r['macro_f1']:>10.1%}{r['kappa']:>7.2f}")
Path("results").mkdir(exist_ok=True)
with open(f"results/baseline_{SPLIT}.json", "w") as f:
    json.dump({"split": SPLIT, "n_eval": len(evalset), "train_waves": TRAIN_WAVES,
               "n_train": len(train), "results": rows}, f, indent=2)
evalset[["id", "label12", "pred_label12", "label50", "pred_label50"]].to_csv(
    f"data/processed/baseline_pred_{SPLIT}.csv", index=False)
print(f"\nSaved results/baseline_{SPLIT}.json")


# 5. Seen or new? Does the exact answer (ignoring case and punctuation) appear in the training data
def normalise(text):
    return text.astype(str).str.lower().str.replace(r"[\W_]", " ", regex=True).str.strip()


seen = normalise(evalset['text']).isin(set(normalise(train["text"])))
seen_vs_new = {"seen_share": round(seen.mean(), 3)}

print(
    f"\n Seen in training, word for word: {seen.sum()} of {len(evalset)} answers ({seen.mean():.0%})")

for scheme, n_cat in (("label12", 12), ("label50", 50)):
    right = evalset[scheme] == evalset[f"pred_{scheme}"]
    seen_vs_new[f"{n_cat}_cat"] = {"seen": round(right[seen].mean(), 3),
                                   "new": round(right[~seen].mean(), 3)}
    print(f"   {n_cat} categories, classifier accuracy: seen {right[seen].mean():.1%} | "
          f"new {right[~seen].mean():.1%} ({(-seen).sum()} new answers)")


# Add the split to the saved results, and a 'seen' column to the saved predictions
with open(f"results/baseline_{SPLIT}.json") as f:
    saved = json.load(f)

saved["seen_vs_new"] = seen_vs_new
with open(f"results/baseline_{SPLIT}.json", "w") as f:
    json.dump(saved, f, indent=2)

evalset["seen"] = seen

evalset[["id", "seen", "label12", "pred_label12", "label50",
         "pred_label50"]].to_csv(
             f"data/processed/baseline_pred_{SPLIT}.csv", index=False
)
