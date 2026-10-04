import json
from pathlib import Path

import pandas as pd
import pyreadstat


PANEL = "data/raw/BES2024_W31_Panel_v31.05.sav"
dev = pd.read_csv("data/processed/eval_dev.csv")


# 1. Every category name exactly as BES spells it

_, meta = pyreadstat.read_sav(PANEL, metadataonly=True)


def names(column):
    # skip special codes
    return {label for code, label in meta.variable_value_labels[column].items() if 0 < float(code) < 100}


names50 = sorted(names("mii_catW25"))
names12 = sorted(names("small_mii_catW25"))

# 2. Example sheet: up to 3 dev answers per category. Real answers -> stays in data/ (git-ignored)
lines = []
for label12, group12 in dev.groupby("label12"):
    lines.append(f"\n=== {label12} ({len(group12)} dev answers) ===")
    for label50, group50 in group12.groupby("label50"):
        lines.append(f"  --- {label50} ({len(group50)})")
        lines += [f"        {text}" for text in group50["text"].head(3)]

Path("data/processed/dev_examples.txt").write_text("\n".join(lines), encoding="utf-8")
print("Wrote data/processed/dev_examples.txt (local only, never commit or paste it)")


# 3. Codebook skeletons (never overwrites a file you have started)
Path("codebooks").mkdir(exist_ok=True)
for file, category_names in [("codebooks/codebook_12.json", names12),
                             ("codebooks/codebook_50.json", names50)]:
    if Path(file).exists():
        print(f"{file} already exists, not overwritten")
        continue
    skeleton = [{"name": n, "definition": "", "examples": []}
                for n in category_names]
    Path(file).write_text(json.dumps(
        skeleton, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Created {file} with {len(category_names)} categories")
