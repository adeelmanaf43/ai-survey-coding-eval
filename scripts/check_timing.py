# Why is the 50-category laptop time so high? Reads only Llama's timing columns: safe to share.
import pandas as pd

for scheme in ("12", "50"):
    df = pd.read_csv(f"data/processed/llm_test_{scheme}.csv")
    sec = df["seconds"]
    print(f"\n{scheme} categories: {len(df):,} answers, {sec.sum() / 3600:.1f} h in total")
    print(f"  median {sec.median():.1f} s, mean {sec.mean():.1f} s, slowest {sec.max():,.0f} s")
    print(f"  answers over 60 s: {(sec > 60).sum()}, together {sec[sec > 60].sum() / 3600:.1f} h;"
          f" all the other answers: {sec[sec <= 60].sum() / 3600:.1f} h")
    print("  10 slowest (row in the file, seconds, prompt tokens Ollama had to read):")
    for row, r in df.nlargest(10, "seconds").iterrows():
        print(f"    row {row:>4}   {r['seconds']:>9,.0f} s   {r.get('prompt_tokens', '-')} tokens")
