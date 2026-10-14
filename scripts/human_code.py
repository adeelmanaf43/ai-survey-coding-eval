# Human benchmark: you code 100 test answers yourself, blind and timed.
# It shows the answer text only, never the BES code or any model's code. Your codes are saved
# in data/processed (git-ignored), so nothing leaves your computer.


import csv
import json
import time
from pathlib import Path

import pandas as pd

N, SEED = 100, 2026
OUT = Path("data/processed/human_test_12.csv")


with open("codebooks/codebook_12.json", encoding="utf-8") as f:
    codebook = json.load(f)
names = [c["name"] for c in codebook]


def show_names():
    cells = [f"{i:>2}. {name:<20}" for i, name in enumerate(names, start=1)]
    for j in range(0, len(cells), 3):
        print("  " + "".join(cells[j:j + 3]))


def show_definitions():
    for i, c in enumerate(codebook, start=1):
        print(f"  {i:>2}. {c['name']}: {c['definition']}")


test = pd.read_csv("data/processed/eval_test.csv", keep_default_na=False)
# the same 100 answers every time
sample = test.sample(n=N, random_state=SEED)[["id", "text"]]
done = set(pd.read_csv(OUT, keep_default_na=False)
           ["id"]) if OUT.exists() else set()
todo = sample[~sample["id"].isin(done)]


print("Same rules Llama got: code the FIRST issue mentioned. Use Uncoded only if the answer is")
print("unclear, too vague, or not about politics or society. Don't look anything up.\n")
show_definitions()
print(f"\n{len(done)} of {N} done, {len(todo)} to go. The clock runs while an answer is on screen,")
print("so type q to pause instead of walking away.")

new_file = not OUT.exists()
with open(OUT, "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    if new_file:
        writer.writerow(["id", "human", "seconds"])
    for k, row in enumerate(todo.itertuples(), start=len(done) + 1):
        print("\n" + "-" * 66)
        print(f"Answer {k} of {N}:\n\n    {row.text}\n")
        show_names()
        start = time.perf_counter()
        while True:
            choice = input(
                "\nYour code (1-13, ? = definitions, q = pause): ").strip().lower()
            if choice == "q":
                raise SystemExit(
                    "Saved. Run the script again to continue where you stopped.")
            if choice == "?":
                show_definitions()
            elif choice.isdigit() and 1 <= int(choice) <= len(names):
                break
            else:
                print("Type a number from 1 to 13.")
        seconds = round(time.perf_counter() - start, 1)
        writer.writerow([row.id, names[int(choice) - 1], seconds])
        f.flush()   # saved after every answer
        print(f"  -> {names[int(choice) - 1]} ({seconds:.0f} s)")

print(f"\nAll {N} done. Next: python -m coding_eval metrics")
