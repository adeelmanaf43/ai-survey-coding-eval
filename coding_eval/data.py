"""Reading and writing project files, and building the wave 25 evaluation set.
Answer-level files live in data/processed (git-ignored); results/ holds only aggregate numbers."""
import json

import pandas as pd

from .config import CONFIG, PROCESSED, RESULTS, ROOT, SEED


def eval_set(split):
    """The dev or test answers with their BES codes."""
    return pd.read_csv(PROCESSED / f"eval_{split}.csv", keep_default_na=False)


def llm_path(split, scheme, prompt="B"):
    """Prompt B (the frozen prompt) keeps the original file names; prompt A gets its own files (D-023)."""
    return PROCESSED / f"llm_{split}_{scheme}{'_A' if prompt == 'A' else ''}.csv"


def llm_codes(split, scheme, prompt="B"):
    """Llama's saved codes for one split, scheme and prompt."""
    return pd.read_csv(llm_path(split, scheme, prompt), keep_default_na=False)


def classifier_codes(split):
    """The TF-IDF classifier's saved codes and confidence."""
    return pd.read_csv(PROCESSED / f"baseline_pred_{split}.csv", keep_default_na=False)


def human_codes():
    """Your 100 blind codes from the human benchmark (H8)."""
    return pd.read_csv(PROCESSED / "human_test_12.csv", keep_default_na=False)


def save_json(name, obj):
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / name, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"Saved results/{name}")


def load_json(name):
    with open(RESULTS / name) as f:
        return json.load(f)


def build_eval_set():
    """Join wave 25 answers to their human codes and draw a stratified sample: dev 300, test 1,200 (D-006).
    Prints counts only, never answers."""
    import pyreadstat
    from sklearn.model_selection import train_test_split

    paths, cfg = CONFIG["paths"], CONFIG["eval_set"]
    w = cfg["wave"]
    text, cat50, cat12 = f"MII_textW{w}", f"mii_catW{w}", f"small_mii_catW{w}"

    # 1. Read only the columns we need (3 of the panel's 13,381)
    print("Reading files (the panel file can take a minute)...")
    strings, _ = pyreadstat.read_sav(
        str(ROOT / paths["strings"]), usecols=["id", text])
    panel, meta = pyreadstat.read_sav(
        str(ROOT / paths["panel"]), usecols=["id", cat50, cat12])

    # 2. Keep real answers: trim spaces, drop blanks and answers with no letters or digits
    strings["text"] = strings[text].fillna("").astype(str).str.strip()
    answers = strings[strings["text"].str.contains(r"\w", regex=True)].copy()
    print(f"W{w} answers with text:          {len(answers):,}")

    # 3. Join each answer to its human codes by respondent id
    df = answers.merge(panel, on="id", how="inner")
    print(f"  ...found in the panel file:   {len(df):,}")
    df = df[df[cat50].notna()].copy()
    print(f"  ...with a 50-category code:   {len(df):,}")

    # 4. Code numbers -> category names, from the file's value labels
    labels50 = {float(k): v for k,
                v in meta.variable_value_labels[cat50].items()}
    labels12 = {float(k): v for k,
                v in meta.variable_value_labels[cat12].items()}
    df["label50"] = df[cat50].map(labels50)
    df["label12"] = df[cat12].map(labels12)
    df["one_word"] = df["text"].str.split().str.len() == 1

    # 5. "Uncoded" answers get a 50-category code but no 12-category one: keep them as their own class (D-005)
    uncoded = df["label50"].str.lower() == "uncoded"
    df.loc[uncoded & df["label12"].isna(), "label12"] = "Uncoded"
    missing = df[df["label12"].isna()]
    if len(missing):
        print("[!] No 12-category code (expected none):",
              missing["label50"].value_counts().to_dict())
    df = df.dropna(subset=["label12"])

    # 6. Stratified sample: dev and test keep the same category mix as the whole wave
    def strata(labels, min_size):
        counts = labels.value_counts()
        # tiny categories pooled, for sampling only
        groups = labels.where(labels.map(counts) >= min_size, "rare")
        # one row alone can't be split
        if (groups == "rare").sum() == 1:
            groups = groups.replace("rare", counts.index[0])
        return groups

    sample, _ = train_test_split(df, train_size=cfg["dev"] + cfg["test"], random_state=SEED,
                                 stratify=strata(df["label50"], 10))
    dev, test = train_test_split(
        sample, train_size=cfg["dev"], random_state=SEED, stratify=strata(sample["label50"], 5))

    # 7. Save locally (data/ is git-ignored) and print a safe summary
    PROCESSED.mkdir(parents=True, exist_ok=True)
    columns = ["id", "text", "label50", "label12", "one_word"]
    dev[columns].to_csv(PROCESSED / "eval_dev.csv", index=False)
    test[columns].to_csv(PROCESSED / "eval_test.csv", index=False)

    print(f"\nPool: {len(df):,} | dev: {len(dev)} | test: {len(test)}")
    print(
        f"50-scheme categories: pool {df['label50'].nunique()}, dev {dev['label50'].nunique()}, test {test['label50'].nunique()}")
    print(f"One-word answers: {df['one_word'].mean():.0%} of the pool")
    print(pd.DataFrame({"dev": dev["label12"].value_counts(
    ), "test": test["label12"].value_counts()}).fillna(0).astype(int))
