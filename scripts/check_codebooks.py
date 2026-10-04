# CHecks: names match the data, every category is defined, no dev answers copied word for word.
import json

import pandas as pd

dev = pd.read_csv("data/processed/eval_dev.csv")
dev_texts = set(dev["text"].str.lower().str.strip())

for file, column in [("codebooks/codebook_12.json", "label12"), ("codebooks/codebook_50.json", "label50")]:
    with open(file, encoding="utf-8") as f:
        book = json.load(f)

    names = {c["name"] for c in book}
    print(f"\n{file}: {len(book)} categories")
    print("  dev labels missing from codebook: ",
          sorted(set(dev[column]) - names) or "none")

    print("   empty definitions:", [c["name"]
          for c in book if not c["definition"].strip()] or "none")

    copied = [c["name"] for c in book for e in c["examples"]
              if e.lower().strip() in dev_texts and len(e.split()) >= 4]
    print("   categories with a copied dev answer (rewrite it):", copied or "none")
