# Why does Ollama crash? Send the same request several times, changing ONE thing each time.
# Safe to share: the tests use an invented answer, the first test answer is never printed,
# and only technical lines from Ollama's own log file are shown.
import hashlib
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

HOST = "http://localhost:11434"
MODEL = "llama3.1:latest"
DAY1_VERSION = "0.35.0"   # the Ollama version you had on Day 1
REAL = {"temperature": 0, "seed": 2026, "num_ctx": 8192,
        "num_predict": 50}   # same options as llm_code.py
QUESTION = ("As far as you're concerned, what is the SINGLE MOST important issue "
            "facing the country at the present time?")


# copied from llm_code.py, so the prompt version must come out as 459a2e64
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


with open("codebooks/codebook_12.json", encoding="utf-8") as f:
    codebook = json.load(f)
PROMPT = system_prompt(codebook)
SCHEMA = {"type": "object",
          "properties": {"category": {"type": "string", "enum": [c["name"] for c in codebook]},
                         "unsure": {"type": "boolean"}},
          "required": ["category", "unsure"]}


def ask(answer, options, with_codebook=True, with_json_rules=True):
    """One request to Ollama. Returns (worked, category or error message, seconds)."""
    if with_codebook:
        messages = [{"role": "system", "content": PROMPT}, {
            "role": "user", "content": f'Answer: "{answer}"'}]
    else:
        messages = [{"role": "user", "content": "Reply with one word: hello"}]
    payload = {"model": MODEL, "stream": False, "keep_alive": "10m",
               "messages": messages, "options": options}
    if with_json_rules:
        payload["format"] = SCHEMA
    t0 = time.time()
    try:
        reply = requests.post(f"{HOST}/api/chat",
                              json=payload, timeout=900).json()
    except requests.exceptions.RequestException as e:
        reply = {"error": str(e)}
    seconds = time.time() - t0
    if "error" in reply:
        time.sleep(10)   # give Ollama a moment to clean up after a crash
        return False, str(reply["error"]), seconds
    text = reply.get("message", {}).get("content", "")
    try:
        text = json.loads(text)["category"]
    except (json.JSONDecodeError, KeyError, TypeError):
        pass
    return True, str(text), seconds


def runs_on():
    """CPU or GPU, from Ollama's list of loaded models."""
    try:
        for m in requests.get(f"{HOST}/api/ps", timeout=10).json().get("models", []):
            if m.get("size"):
                gpu = (m.get("size_vram") or 0) / m["size"]
                return "on 100% CPU" if gpu < 0.01 else f"on {gpu:.0%} GPU"
    except requests.exceptions.RequestException:
        pass
    return ""


# ---------- 1. versions ----------
try:
    version = requests.get(f"{HOST}/api/version", timeout=10).json()["version"]
except requests.exceptions.RequestException:
    raise SystemExit(
        "Can't reach Ollama. Open the Ollama app (icon near the clock), then run this again.")
print(f"Ollama version now: {version} (Day 1: {DAY1_VERSION})")
print(
    f"Prompt version: {hashlib.sha1(PROMPT.encode('utf-8')).hexdigest()[:8]} (should be 459a2e64)")
print('Tests A-E send the invented answer "cost of living". This takes 5 to 15 minutes.\n')

# ---------- 2. tests: change one thing at a time ----------
TESTS = [("A", "short prompt, no codebook", dict(options={}, with_codebook=False, with_json_rules=False)),
         ("B", "your exact setup", dict(options=REAL)),
         ("C", "B without the JSON rules", dict(
             options=REAL, with_json_rules=False)),
         ("D", "B on the CPU only (num_gpu 0)",
          dict(options={**REAL, "num_gpu": 0})),
         ("E", "B with less memory (num_ctx 4096)", dict(options={**REAL, "num_ctx": 4096}))]
worked, first_error = {}, None
for key, label, kwargs in TESTS:
    if key in "CDE" and worked["B"]:
        break   # B works, so there is nothing to fix
    ok, result, seconds = ask("cost of living", **kwargs)
    worked[key] = ok
    print(f"{key}  {label:<36} {'works' if ok else 'CRASH':<6} {seconds:4.0f} s  {runs_on() if ok else ''}")
    if not ok and first_error is None:
        first_error = result
if first_error:
    print(f"   Ollama's error: {first_error[:110]}")

if worked["B"]:
    setup, change, cause = REAL, None, None
elif worked.get("D"):
    setup, change = {
        **REAL, "num_gpu": 0}, 'in the "options" of the payload, add  "num_gpu": 0,  before "num_predict": 50'
    cause = "the graphics card (GPU): on the CPU alone it works"
elif worked.get("E"):
    setup, change = {
        **REAL, "num_ctx": 4096}, 'change  "num_ctx": 8192  to  "num_ctx": 4096  (the 50-category prompt still fits)'
    cause = "memory: with a smaller context window it works"
else:
    setup = None

# ---------- 3. the setup that works: first test answer + same codes as the dev run? ----------
first_ok, same = None, None
if setup:
    first = str(pd.read_csv("data/processed/eval_test.csv",
                keep_default_na=False)["text"].iloc[0])
    first_ok = ask(first, setup)[0]
    unusual = sum(not (ch.isascii() and ch.isprintable()) for ch in first)
    print(
        f"F  first test answer ({len(first)} characters, {unusual} unusual) {'works' if first_ok else 'CRASH'}")
    dev = pd.read_csv("data/processed/eval_dev.csv",
                      keep_default_na=False).head(10)
    before = (pd.read_csv("data/processed/llm_dev_12.csv", keep_default_na=False)
              .drop_duplicates("id", keep="last").set_index("id")["pred"])
    same = sum(ask(row.text, setup)[1] == before.get(
        row.id) for row in dev.itertuples())
    print(
        f"G  first 10 dev answers: {same}/10 get the same code as in your dev run")

# ---------- 4. Ollama's own log: versions over time + technical lines ----------
folder = Path(os.environ.get("LOCALAPPDATA", "")) / "Ollama"
logs = sorted(folder.glob("server*.log"),
              key=lambda p: p.stat().st_mtime) if folder.is_dir() else []
if logs:
    print(f"\nOllama versions in its log files ({folder}):")
    for p in logs:
        found = re.findall(
            r"\(version ([^)\s]+)\)", p.read_text(encoding="utf-8", errors="replace"))
        when = datetime.fromtimestamp(
            p.stat().st_mtime).strftime("%a %d %b %H:%M")
        print(
            f"  {p.name:<13} last written {when}: version {', '.join(dict.fromkeys(found)) or '?'}")
    lines = logs[-1].read_text(encoding="utf-8", errors="replace").splitlines()
    start = max([i for i, line in enumerate(lines)
                if "Listening on" in line], default=0)
    key_line = re.compile(r"level=(ERROR|WARN)|panic|exit status|terminated|out of memory|CUDA error|"
                          r"GGML_ASSERT|vulkan|exception|what\(\)|inference compute|offload|system memory", re.I)
    picked = [line.strip()[:220] for line in lines[start:]
              if key_line.search(line) and not any(s in line for s in ("prompt=", "content", "[GIN]"))]
    print(
        f"Key lines from {logs[-1].name} since Ollama last started (technical only):")
    for line in list(dict.fromkeys(picked))[-25:]:
        print("  " + line)
else:
    print(f"\n(No Ollama log files found in {folder})")

# ---------- 5. verdict ----------
print("\nWHAT THIS MEANS")
if version != DAY1_VERSION:
    print(
        f"- Ollama has updated itself since Day 1 ({DAY1_VERSION} -> {version}).")
if setup is None:
    if worked.get("C"):
        print("- Only C works, but your method needs the JSON rules. So it's a bug in Ollama, not your script or data.")
    elif worked["A"]:
        print("- Short prompts work, but every version of your codebook request crashes. It's Ollama, not your data.")
    else:
        print("- Every request crashed, even a short one. So the problem is Ollama itself, not your script or data.")
    if version != DAY1_VERSION:
        print(f"- Reinstall {DAY1_VERSION}: download OllamaSetup.exe from "
              f"https://github.com/ollama/ollama/releases/tag/v{DAY1_VERSION}")
        print("  and run it (no need to uninstall first). Then run this script again.")
    else:
        print("- Paste me this whole output.")
elif not first_ok:
    print("- The setup works with an invented answer but crashes on the first test answer.")
    print("  Don't change anything; paste me this whole output.")
elif same < 9:
    print(
        f"- It works, but only {same}/10 dev answers got the same code as before, so the setup now")
    print("  behaves differently. Paste me this whole output before starting the run.")
else:
    if change:
        print(f"- FIX FOUND. The crash comes from {cause}.")
        print(f"  In scripts\\llm_code.py, {change}.")
    else:
        print("- Your exact setup works now, so the crash was temporary.")
    print(f"- {same}/10 dev answers got the same code as in your dev run"
          + (", so results stay comparable." if same == 10 else ": a small difference, fine to go ahead."))
    print("- Start the overnight run: python scripts\\llm_code.py")
