# Builds the wave 25 evaluation set: dev 300 + test 1,200 answers.
# Saves to data/processed (git-ignored). Prints counts only, never answers.
from pathlib import Path

import pandas as pd
import pyreadstat
from sklearn.model_selection import train_test_split

STRINGS = "data/raw/BES2024_W30Strings_v30.1.sav"
PANEL = "data/raw/BES2024_W31_Panel_v31.05.sav"
SEED = 2026

# 1. Read only the columns we need (3 of the panel's 13,381)
print("Reading files (the panel file can take a minute)...")
strings, _ = pyreadstat.read_sav(STRINGS, usecols=["id", "MII_textW25"])
panel, meta = pyreadstat.read_sav(PANEL, usecols=["id", "mii_catW25", "small_mii_catW25"])

# 2. Keep real answers: trim spaces, drop blanks and answers with no letters or digits
strings["text"] = strings["MII_textW25"].fillna("").astype(str).str.strip()
answers = strings[strings["text"].str.contains(r"\w", regex=True)].copy()
print(f"W25 answers with text:          {len(answers):,}")

# 3. Join each answer to its human codes by respondent id
df = answers.merge(panel, on="id", how="inner")
print(f"  ...found in the panel file:   {len(df):,}")
df = df[df["mii_catW25"].notna()].copy()
print(f"  ...with a 50-category code:   {len(df):,}")

# 4. Code numbers -> category names, from the file's value labels
labels50 = {float(k): v for k, v in meta.variable_value_labels["mii_catW25"].items()}
labels12 = {float(k): v for k, v in meta.variable_value_labels["small_mii_catW25"].items()}
df["label50"] = df["mii_catW25"].map(labels50)
df["label12"] = df["small_mii_catW25"].map(labels12)
df["one_word"] = df["text"].str.split().str.len() == 1

# 5. "Uncoded" answers get a 50-category code but no 12-category one: keep them as their own class
uncoded = df["label50"].str.lower() == "uncoded"
df.loc[uncoded & df["label12"].isna(), "label12"] = "Uncoded"
missing = df[df["label12"].isna()]
if len(missing):
    print("[!] No 12-category code (expected none):", missing["label50"].value_counts().to_dict())
df = df.dropna(subset=["label12"])

# 6. Stratified sample: dev and test keep the same category mix as the whole wave
def strata(labels, min_size):
    counts = labels.value_counts()
    groups = labels.where(labels.map(counts) >= min_size, "rare")  # tiny categories pooled, for sampling only
    if (groups == "rare").sum() == 1:                                # one row alone can't be split
        groups = groups.replace("rare", counts.index[0])
    return groups

sample, _ = train_test_split(df, train_size=1500, random_state=SEED, stratify=strata(df["label50"], 10))
dev, test = train_test_split(sample, train_size=300, random_state=SEED, stratify=strata(sample["label50"], 5))

# 7. Save locally (data/ is git-ignored) and print a safe summary
out = Path("data/processed")
out.mkdir(parents=True, exist_ok=True)
columns = ["id", "text", "label50", "label12", "one_word"]
dev[columns].to_csv(out / "eval_dev.csv", index=False)
test[columns].to_csv(out / "eval_test.csv", index=False)

print(f"\nPool: {len(df):,} | dev: {len(dev)} | test: {len(test)}")
print(f"50-scheme categories: pool {df['label50'].nunique()}, dev {dev['label50'].nunique()}, test {test['label50'].nunique()}")
print(f"One-word answers: {df['one_word'].mean():.0%} of the pool")
print(pd.DataFrame({"dev": dev["label12"].value_counts(), "test": test["label12"].value_counts()}).fillna(0).astype(int))
