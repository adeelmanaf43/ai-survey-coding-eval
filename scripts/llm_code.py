# Llama codes the evaluation answers with your codebooks, 100% locally through Ollama.
# Progress lines show counts and timings only (no answers), so the output is safe to share.
#   python scripts\llm_code.py                          prompt B on the test set (the frozen final run)
#   python scripts\llm_code.py --split dev --prompt A   prompt A on the dev set (roadmap H5, D-023)
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
SPLIT = "test"    # final evaluation: the test set is used once ("dev" while building, or use --split dev)
SEED = 2026
QUESTION = ("As far as you're concerned, what is the SINGLE MOST important issue "
            "facing the country at the present time?")


def system_prompt(codebook, with_examples=True):
    """Prompt B (with_examples=True) is the frozen prompt; prompt A is the same without the examples."""
    lines = ["You code answers to this British survey question, asked in May 2023:",
             f'"{QUESTION}"',
             "",
             "Assign each answer to exactly one category from the codebook below.",
             "Rules:",
             "- Follow the definitions and examples." if with_examples else "- Follow the definitions.",
             "- If the answer names several issues, code the first one mentioned.",
             '- Use "Uncoded" only if the answer is unclear, too vague, or not about politics or society.',
             '- Set "unsure" to true if you are not confident.',
             "",
             "Codebook:"]
    for c in codebook:
        examples = "; ".join(f'"{e}"' for e in c["examples"]) if with_examples else ""
        lines.append(f"- {c['name']}: {c['definition']}" +
                     (f" Examples: {examples}" if examples else ""))
    lines += ["",
              'Reply with JSON only, for example {"category": "Health", "unsure": false}.']
    return "\n".join(lines)


parser = argparse.ArgumentParser()
parser.add_argument("--split", choices=["dev", "test"], default=SPLIT,
                    help="which evaluation set to code (default: the SPLIT setting above)")
parser.add_argument("--prompt", choices=["A", "B"], default="B",
                    help="B: definitions plus examples, the frozen prompt (default); A: definitions only")
parser.add_argument("--limit", type=int,
                    help="code only the first N answers (quick test)")
args = parser.parse_args()
SPLIT = args.split

answers = pd.read_csv(
    f"data/processed/eval_{SPLIT}.csv", keep_default_na=False)
if args.limit:
    answers = answers.head(args.limit)

# one scheme at a time, so Ollama can reuse the codebook it has already read
for scheme in ("12", "50"):
    with open(f"codebooks/codebook_{scheme}.json", encoding="utf-8") as f:
        codebook = json.load(f)
    prompt = system_prompt(codebook, with_examples=args.prompt == "B")
    # changes if the codebook or prompt changes
    version = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:8]
    schema = {"type": "object",
              "properties": {"category": {"type": "string", "enum": [c["name"] for c in codebook]},
                             "unsure": {"type": "boolean"}},
              "required": ["category", "unsure"]}

    # prompt B keeps its original file names; prompt A gets its own files, so nothing is overwritten
    out = Path(f"data/processed/llm_{SPLIT}_{scheme}{'_A' if args.prompt == 'A' else ''}.csv")
    done = pd.read_csv(out, keep_default_na=False) if out.exists(
    ) else pd.DataFrame(columns=["id", "version"])
    if (done["version"].astype(str) != version).any():
        raise SystemExit(
            f"{out} was made with a different codebook or prompt. Delete that file to start fresh.")
    todo = answers[~answers["id"].isin(done["id"])]
    print(f"\n{scheme} categories, prompt {args.prompt}, {SPLIT} set: {len(done)} already coded,"
          f" {len(todo)} to go (prompt version {version})")

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
                       "options": {"temperature": 0, "seed": SEED, "num_ctx": 8192, "num_gpu": 0, "num_predict": 50}}
            # up to 3 tries: after a crash, Ollama reloads the model by itself
            for attempt in range(1, 4):
                t0 = time.time()
                try:
                    reply = requests.post(
                        OLLAMA, json=payload, timeout=600).json()
                except requests.exceptions.RequestException as e:
                    reply = {"error": str(e)}
                if "error" not in reply:
                    break
                print(
                    f"  Ollama error, try {attempt} of 3: {str(reply['error'])[:150]}")
                time.sleep(30)
            else:
                raise SystemExit(
                    "Ollama failed 3 times in a row. Restart Ollama, then run the script again.")
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

print("\nDone. Next: python scripts\\" + ("score_llm.py" if args.prompt == "B" else "compare_prompts.py"))
