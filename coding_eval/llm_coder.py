"""Llama codes the answers with your codebooks, 100% locally through Ollama.
Progress lines show counts and timings only (no answers), so the output is safe to share."""
import csv
import hashlib
import json
import time

import pandas as pd
import requests

from .config import CODEBOOKS, CONFIG, ROOT, SCHEMES
from .data import eval_set, llm_path

QUESTION = ("As far as you're concerned, what is the SINGLE MOST important issue "
            "facing the country at the present time?")


def load_codebook(scheme):
    with open(CODEBOOKS / f"codebook_{scheme}.json", encoding="utf-8") as f:
        return json.load(f)


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
        examples = "; ".join(
            f'"{e}"' for e in c["examples"]) if with_examples else ""
        lines.append(f"- {c['name']}: {c['definition']}" +
                     (f" Examples: {examples}" if examples else ""))
    lines += ["",
              'Reply with JSON only, for example {"category": "Health", "unsure": false}.']
    return "\n".join(lines)


def prompt_version(prompt):
    """A short fingerprint of the prompt: it changes whenever the codebook or prompt text changes."""
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:8]


def answer_schema(codebook):
    """Structured output: Llama may only answer with a codebook name and a true/false 'unsure' flag."""
    return {"type": "object",
            "properties": {"category": {"type": "string", "enum": [c["name"] for c in codebook]},
                           "unsure": {"type": "boolean"}},
            "required": ["category", "unsure"]}


def parse_reply(reply):
    """Llama's code and 'unsure' flag from an Ollama reply, or ("INVALID", "") if it can't be read."""
    try:
        result = json.loads(reply["message"]["content"])
        return result["category"], result["unsure"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return "INVALID", ""


def ask_ollama(payload, tries=3, wait=30):
    """One request, retried after a crash (D-016). Returns the reply and the seconds the successful try took."""
    for attempt in range(1, tries + 1):
        t0 = time.time()
        try:
            reply = requests.post(
                CONFIG["llm"]["url"], json=payload, timeout=CONFIG["llm"]["timeout_seconds"]).json()
        except requests.exceptions.RequestException as e:
            reply = {"error": str(e)}
        if "error" not in reply:
            return reply, round(time.time() - t0, 2)
        print(
            f"  Ollama error, try {attempt} of {tries}: {str(reply['error'])[:150]}")
        time.sleep(wait)
    raise SystemExit(
        f"Ollama failed {tries} times in a row. Restart Ollama, then run the same command again.")


def code_answers(split="test", prompt="B", limit=None):
    """Code every answer of one split with Llama, one scheme at a time, saving after every answer.
    A stopped run resumes where it stopped; a file made with another prompt version is never mixed in."""
    llm = CONFIG["llm"]
    answers = eval_set(split)
    if limit:
        answers = answers.head(limit)
    for scheme in SCHEMES:     # one scheme at a time, so Ollama can reuse the codebook it has already read
        codebook = load_codebook(scheme)
        system = system_prompt(codebook, with_examples=prompt == "B")
        version = prompt_version(system)
        out = llm_path(split, scheme, prompt)
        done = pd.read_csv(out, keep_default_na=False) if out.exists(
        ) else pd.DataFrame(columns=["id", "version"])
        if (done["version"].astype(str) != version).any():
            raise SystemExit(f"{out.relative_to(ROOT)} was made with a different codebook or prompt."
                             " Delete that file to start fresh.")
        todo = answers[~answers["id"].isin(done["id"])]
        print(f"\n{scheme} categories, prompt {prompt}, {split} set: {len(done)} already coded,"
              f" {len(todo)} to go (prompt version {version})")

        new_file = not out.exists()
        with open(out, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(["id", "pred", "unsure", "seconds", "prompt_tokens", "cached_tokens",
                                 "output_tokens", "model", "version"])
            started = time.time()
            for i, row in enumerate(todo.itertuples(), start=1):
                payload = {"model": llm["model"], "stream": False, "format": answer_schema(codebook), "keep_alive": "30m",
                           "messages": [{"role": "system", "content": system},
                                        {"role": "user", "content": f'Answer: "{row.text}"'}],
                           "options": {"temperature": llm["temperature"], "seed": CONFIG["seed"], "num_ctx": llm["num_ctx"],
                                       "num_gpu": llm["num_gpu"], "num_predict": llm["num_predict"]}}
                reply, seconds = ask_ollama(payload)
                pred, unsure = parse_reply(reply)
                writer.writerow([row.id, pred, unsure, seconds, reply.get("prompt_eval_count"),
                                 reply.get("prompt_eval_cached_count"), reply.get("eval_count"), llm["model"], version])
                f.flush()      # saved after every answer: if anything stops the run, just start it again
                if i % 10 == 0 or i == len(todo):
                    per = (time.time() - started) / i
                    print(
                        f"  {i}/{len(todo)} | {per:.1f} s per answer | about {per * (len(todo) - i) / 60:.0f} min left")
    print("\nDone. Next: python -m coding_eval metrics")
