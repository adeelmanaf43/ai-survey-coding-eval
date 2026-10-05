# Session 11: Llama codes the evaluation answers with your codebooks, 100% locally through Ollama.
# Progress lines show counts and timings only (no answers), so the output is safe to share.
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import pandas as pd
import requests

MODEL = "llama3.1:latest"
OLLAMA = "http://localhost:11434/api/chat"
SPLIT = "dev"     # change to "test" only once, in the final evaluation
SEED = 2026
QUESTION = ("As far as you're concerned, what is the SINGLE MOST important issue "
            "facing the country at the present time?")


def system_prompt(codebook):
    lines = ["You code answers to this British survey question, asked in May 2023:",
             f'"{QUESTION}"',
             "",
             "Assign each answer to exactly one category from the codebook below.",
             "Rules:",
             "- Follow the definitions and examples.",
             "- If the answer names several issues, code the first one mentioned.",
             '- Use "Uncoded" only if the answer is unclear, too vague, or not about politics or society.',
             '- Set "unsure" to true if you are not confident.',
             "",
             "Codebook:"]
    for c in codebook:
        examples = "; ".join(f'"{e}"' for e in c["examples"])
        lines.append(f"- {c['name']}: {c['definition']}" +
                     (f" Examples: {examples}" if examples else ""))
    lines += ["",
              'Reply with JSON only, for example {"category": "Health", "unsure": false}.']
    return "\n".join(lines)


parser = argparse.ArgumentParser()
parser.add_argument("--limit", type=int,
                    help="code only the first N answers (quick test)")
args = parser.parse_args()

answers = pd.read_csv(
    f"data/processed/eval_{SPLIT}.csv", keep_default_na=False)
if args.limit:
    answers = answers.head(args.limit)

# one scheme at a time, so Ollama can reuse the codebook it has already read
for scheme in ("12", "50"):
    with open(f"codebooks/codebook_{scheme}.json", encoding="utf-8") as f:
        codebook = json.load(f)
    prompt = system_prompt(codebook)
    # changes if the codebook or prompt changes
    version = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:8]
    schema = {"type": "object",
              "properties": {"category": {"type": "string", "enum": [c["name"] for c in codebook]},
                             "unsure": {"type": "boolean"}},
              "required": ["category", "unsure"]}

    out = Path(f"data/processed/llm_{SPLIT}_{scheme}.csv")
    done = pd.read_csv(out, keep_default_na=False) if out.exists(
    ) else pd.DataFrame(columns=["id", "version"])
    if (done["version"].astype(str) != version).any():
        raise SystemExit(
            f"{out} was made with a different codebook or prompt. Delete that file to start fresh.")
    todo = answers[~answers["id"].isin(done["id"])]
    print(f"\n{scheme} categories: {len(done)} already coded, {len(todo)} to go (prompt version {version})")

    new_file = not out.exists()
    with open(out, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["id", "pred", "unsure", "seconds", "prompt_tokens", "cached_tokens",
                             "output_tokens", "model", "version"])
        started = time.time()
        for i, row in enumerate(todo.itertuples(), start=1):
            payload = {"model": MODEL, "stream": False, "format": schema, "keep_alive": "30m",
                       "messages": [{"role": "system", "content": prompt},
                                    {"role": "user", "content": f'Answer: "{row.text}"'}],
                       "options": {"temperature": 0, "seed": SEED, "num_ctx": 8192, "num_predict": 50}}
            t0 = time.time()
            reply = requests.post(OLLAMA, json=payload, timeout=600).json()
            if "error" in reply:
                raise SystemExit(f"Ollama error: {reply['error']}")
            try:
                result = json.loads(reply["message"]["content"])
                pred, unsure = result["category"], result["unsure"]
            except (json.JSONDecodeError, KeyError, TypeError):
                pred, unsure = "INVALID", ""
            writer.writerow([row.id, pred, unsure, round(time.time() - t0, 2), reply.get("prompt_eval_count"),
                             reply.get("prompt_eval_cached_count"), reply.get("eval_count"), MODEL, version])
            f.flush()   # saved after every answer: if anything stops the run, just start it again
            if i % 10 == 0 or i == len(todo):
                per = (time.time() - started) / i
                print(
                    f"  {i}/{len(todo)} | {per:.1f} s per answer | about {per * (len(todo) - i) / 60:.0f} min left")

print("\nDone. Next: python scripts\\score_llm.py")
