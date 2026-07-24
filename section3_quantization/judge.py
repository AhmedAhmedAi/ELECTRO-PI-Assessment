"""Blind LLM-judge: Gemini scores both models' answers without knowing
which is which, on standard rubrics, then picks the better answer per prompt.

Design (bias controls):
- BLIND: the judge sees "Answer A" / "Answer B", never model names.
- Position bias: A/B assignment is randomized per prompt (seeded, so the
  blinding map is reproducible); the map is only joined back after judging.
- Rubrics (the common LLM-judge set): instruction_following, truthfulness,
  conciseness, writing_style, helpfulness -- each 1-5 per answer, with the
  judge naming the issues it found.
- Verdict per prompt on the 7-point pairwise scale:
  A-much-better / A-better / A-slightly-better / tie /
  B-slightly-better / B-better / B-much-better.

Outputs: results/judge_results.json (raw, de-blinded) and
         results/judge_summary.md (tables).

Usage:  .venv/bin/python judge.py          # needs GOOGLE_API_KEY (or falls
                                           # back to ../section1_livekit/.env)
"""

import json
import os
import random
import re
import ssl
import time
import urllib.request
from pathlib import Path

import certifi

# This machine's python.org install has no CA certs wired into urllib
# (curl works, urllib doesn't) -- use certifi's bundle explicitly.
_SSL = ssl.create_default_context(cafile=certifi.where())

JUDGE_MODEL = "gemini-3.5-flash"  # newest flash on this API key
MODELS = ["Qwen3.5-4B-BF16", "Qwen3.5-4B-Q4_K_M"]
RUBRICS = ["instruction_following", "truthfulness", "conciseness",
           "writing_style", "helpfulness"]
VERDICTS = ["A_much_better", "A_better", "A_slightly_better", "tie",
            "B_slightly_better", "B_better", "B_much_better"]

HERE = Path(__file__).parent


def api_key() -> str:
    key = os.getenv("GOOGLE_API_KEY", "")
    if not key:  # convenience fallback: reuse Section 1's key
        env = HERE.parent / "section1_livekit" / ".env"
        if env.exists():
            m = re.search(r"^GOOGLE_API_KEY=(.+)$", env.read_text(), re.M)
            key = m.group(1).strip() if m else ""
    if not key:
        raise SystemExit("GOOGLE_API_KEY not set (env or ../section1_livekit/.env)")
    return key


def load_answer(model: str, prompt_file: str) -> str:
    text = (HERE / "results" / f"quality_{model}" / prompt_file).read_text()
    m = re.search(r"^> .*?$(.*?)\[ Prompt:", text, re.S | re.M)
    return m.group(1).strip() if m else text.strip()


JUDGE_INSTRUCTIONS = f"""You are a strict, impartial evaluator comparing two
anonymous AI answers (A and B) to the same user prompt. You do not know which
systems produced them. Judge only what is on the page.

Score BOTH answers 1-5 on each rubric (5 = excellent):
- instruction_following: did it do exactly what the prompt asked (format,
  length, constraints)?
- truthfulness: factual/logical correctness; penalize errors and invention.
- conciseness: appropriately brief; penalize padding and rambling.
- writing_style: fluency and natural quality in the prompt's language
  (if the prompt is Arabic dialect, judge dialect authenticity strictly).
- helpfulness: would this answer satisfy the user?

For each rubric, briefly name the specific issue(s) you found, or "none".
Then give ONE overall pairwise verdict from exactly this list:
{", ".join(VERDICTS)}
with a one-sentence justification.

Return ONLY JSON with this shape:
{{"rubrics": {{"<rubric>": {{"score_a": int, "score_b": int, "issues": "..."}}}},
  "verdict": "<one of the labels>", "why": "..."}}"""


def judge_one(key: str, prompt_text: str, ans_a: str, ans_b: str) -> dict:
    body = {
        "contents": [{"role": "user", "parts": [{"text":
            f"{JUDGE_INSTRUCTIONS}\n\n## User prompt\n{prompt_text}\n\n"
            f"## Answer A\n{ans_a}\n\n## Answer B\n{ans_b}"}]}],
        "generationConfig": {"responseMimeType": "application/json",
                             "temperature": 0},
    }
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{JUDGE_MODEL}:generateContent",
        data=json.dumps(body).encode(),
        headers={"x-goog-api-key": key, "Content-Type": "application/json"},
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=90, context=_SSL) as r:
                data = json.load(r)
            return json.loads(data["candidates"][0]["content"]["parts"][0]["text"])
        except Exception as e:  # transient API hiccups: retry, then give up loudly
            if attempt == 2:
                raise
            print(f"  retry ({e})")
            time.sleep(3)


def main() -> None:
    key = api_key()
    rng = random.Random(42)  # reproducible blinding
    results = []

    for pf in sorted((HERE / "prompts").glob("*.txt")):
        prompt_text = pf.read_text().strip()
        a_is = rng.choice(MODELS)                 # who plays "A" this round
        b_is = MODELS[1] if a_is == MODELS[0] else MODELS[0]
        print(f"judging {pf.name}  (A={a_is.split('-')[-1]})")

        j = judge_one(key, prompt_text,
                      load_answer(a_is, pf.name), load_answer(b_is, pf.name))

        # De-blind: fold A/B back onto model names.
        per_model = {}
        for r in RUBRICS:
            entry = j["rubrics"].get(r, {})
            per_model.setdefault(a_is, {})[r] = entry.get("score_a")
            per_model.setdefault(b_is, {})[r] = entry.get("score_b")
        winner = ("tie" if j["verdict"] == "tie"
                  else a_is if j["verdict"].startswith("A") else b_is)
        margin = j["verdict"].split("_", 1)[1] if j["verdict"] != "tie" else "tie"

        results.append({"prompt": pf.name, "blinding": {"A": a_is, "B": b_is},
                        "scores": per_model, "issues": {r: j["rubrics"].get(r, {}).get("issues")
                                                        for r in RUBRICS},
                        "verdict_raw": j["verdict"], "winner": winner,
                        "margin": margin, "why": j.get("why", "")})

    out = HERE / "results" / "judge_results.json"
    out.write_text(json.dumps({"judge_model": JUDGE_MODEL, "results": results},
                              indent=2, ensure_ascii=False))

    # ---- summary tables ----
    lines = [f"# Blind judge summary ({JUDGE_MODEL}, temp 0, seeded blinding)\n"]
    lines.append("| prompt | winner | margin | why |")
    lines.append("|---|---|---|---|")
    for r in results:
        lines.append(f"| {r['prompt']} | {r['winner'].split('-')[-1]} | "
                     f"{r['margin']} | {r['why'][:110]} |")

    lines.append("\n| rubric | " + " | ".join(m.split("-")[-1] for m in MODELS) + " |")
    lines.append("|---|---|---|")
    for rub in RUBRICS:
        row = []
        for m in MODELS:
            scores = [x["scores"][m][rub] for x in results
                      if x["scores"][m].get(rub) is not None]
            row.append(f"{sum(scores)/len(scores):.1f}" if scores else "-")
        lines.append(f"| {rub} | {row[0]} | {row[1]} |")

    (HERE / "results" / "judge_summary.md").write_text("\n".join(lines) + "\n")
    print("\nwrote results/judge_results.json and results/judge_summary.md")


if __name__ == "__main__":
    main()
