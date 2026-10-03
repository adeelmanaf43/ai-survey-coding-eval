import json
import statistics
import time
import requests


MODEL = "llama3.1:latest"
URL = "http://localhost:11434/api/chat"

CATEGORIES = ["Europe", "Immigration", "Economy", "Health", "Terrorism",
              "Inequality", "Environment", "Austerity/spending", "Negativity",
              "Other lib-auth", "Other left-right", "Other", "Uncoded"]

# Invented answers, not BES data, so this file is safe to commit
ANSWERS = ["cost of living", "the NHS is falling apart", "immigration is out of control",
           "brexit and what comes after it", "climate change", "dunno really, everything",
           "the economy then the health service", "immigration", "energy bills",
           "the war in ukraine", "knif crime", "cuts to school and councils", "pensions not keeping up",
           "too much political correctness", "greedy big corporations", "terror attacks",
           "scottish independence"]


SYSTEM = (
    "You code answers to the survey question 'What is the single most important "
    "issue facing the country?' Choose exactly one category from: "
    + ", ".join(CATEGORIES) +
    " . If several issues are named, code the first one. "
    "Set unsure to true if you are not confident. Reply in JSON.")

SCHEMA = {"type": "object",
          "properties": {"category": {"type": "string", "enum": CATEGORIES},
                         "unsure": {"type": "boolean"}},
          "required": ["category", "unsure"]
          }
times, valid = [], 0
for i, answer in enumerate(ANSWERS, start=1):
    payload = {
        "model": MODEL, "stream": False, "think": False, "format": SCHEMA,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": answer}],
        "options": {"temperature": 0, "num_ctx": 8192}
    }

    start = time.perf_counter()
    reply = requests.post(URL, json=payload, timeout=600).json()
    seconds = time.perf_counter() - start
    if "error" in reply:
        raise SystemError(reply["error"])

    result = json.loads(reply["message"]["content"])

    if result.get("category") in CATEGORIES:
        valid += 1

    times.append(seconds)
    print(f"{i:>2}. {seconds: 5.1f}s {answer:<38} {result}")


# call 1 includes loading the model into memory so leave it out
warm = times[1:]
print(
    f"\nFirst call: {times[0]:.1f}s | median of the rest: {statistics.median(warm): .1f}s per answer")
print(f"Valid categories:{valid} / {len(ANSWERS)}")
