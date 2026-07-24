# Electro Pi — AI Engineer Technical Test

Submission by Ahmed. One folder per section. Each section runs within
**10 minutes of cloning** — the exact commands are below; copy-paste them
from the repo root.

> **Please read each section's own `README.md` and `NOTES.md`.** Every
> section implements **more than the spec asked for** — extra techniques,
> measurements, and findings (e.g. Section 1's aborted-turn purge and
> filler race, Section 2's hybrid retrieval + measured guardrail,
> Section 3's blind LLM judge and TTFT study, Section 4's slot sweep and
> Docker build forensics). The internal READMEs are where all of that is
> documented and evidenced; this file only gets you running.

| Section | Folder | Write-up | Status |
|---|---|---|---|
| 1 — LiveKit voice agent | [section1_livekit/](section1_livekit/) | [NOTES.md](section1_livekit/NOTES.md) | ✅ complete (+ bonus 1.2) |
| 2 — LangChain RAG | [section2_rag/](section2_rag/) | [NOTES.md](section2_rag/NOTES.md) | ✅ complete |
| 3 — Quantization | [section3_quantization/](section3_quantization/) | [NOTES.md](section3_quantization/NOTES.md) | ✅ complete |
| 4 — Model deployment | [section4_deployment/](section4_deployment/) | [NOTES.md](section4_deployment/NOTES.md) | ✅ complete |

## Prerequisites

- Python 3.11+ (spec asks 3.10+)
- `brew install llama.cpp` — Sections 3 & 4 (native runs)
- Docker Desktop — Section 4's container runs only
- API keys (each section's README/`.env.example` has the details):

| Key | Needed by |
|---|---|
| `OPENAI_API_KEY` | Section 1 (STT always; can also cover LLM + TTS) |
| `GOOGLE_API_KEY` | Section 1 (default LLM) · Section 2 · Section 3 (judge step only) |
| `ELEVENLABS_API_KEY` | Section 1 (default TTS; optional — see below) |

**Only have an OpenAI key?** Section 1 runs fully on it: set
`LLM_PROVIDER=openai` and `TTS_PROVIDER=openai` in its `.env`.

## Section 1 — voice agent (~5 min to talking)

```bash
cd section1_livekit
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env                  # paste your keys in
.venv/bin/python agent.py console     # voice: mic + speakers (use headphones)
.venv/bin/python smoke_test.py        # or headless: text in, tool calls out
```

While the console runs, open **http://localhost:8899** — a live view of the
call (recognized speech, reply typing in sync with the voice, barge-in and
tool-call markers). The page opens mic-muted — click the mic button to talk. Evidence of a full session, including a live provider
failover: [transcript.md](section1_livekit/transcript.md) — or watch the
[8-minute video demo](https://drive.google.com/file/d/1qBo52Xbigp-0x06KQy6bGPYHbL8cvdy8/view?usp=sharing).

## Section 2 — RAG (0 min to read, ~6 min to re-run)

The entry point is **[analysis.ipynb](section2_rag/analysis.ipynb)** — the
whole pipeline stage by stage, **shipped pre-executed with real outputs**,
so the results are readable immediately. To re-run everything end-to-end:

```bash
cd section2_rag
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
echo 'GOOGLE_API_KEY=...' > .env
.venv/bin/jupyter execute analysis.ipynb    # ~4 min; or run cells in Jupyter Lab
```

The corpus is a committed snapshot of **electropi.ai itself** — grading
never depends on the live site, and you can verify every citation.

## Section 3 — quantization (0 min to read, one command to reproduce)

Also notebook-first: **[analysis.ipynb](section3_quantization/analysis.ipynb)**
ships pre-executed — download → local `llama-quantize` → RAM/speed/TTFT/quality
measurements, with raw artifacts in `results/`. Full reproduction:

```bash
brew install llama.cpp
cd section3_quantization
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/jupyter execute analysis.ipynb
```

Setup is minutes; the *compute* is ~15 min plus a one-time 8 GB BF16
download (skipped if the file is already present). The blind-judge step
needs `GOOGLE_API_KEY` in the environment and is skipped without it.

## Section 4 — deployment (~5 min to first token)

Serves the Q4_K_M GGUF quantized in Section 3. Fastest path (native,
Metal):

```bash
cd section4_deployment
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
# then, from another terminal:
curl -N -X POST localhost:8000/generate -H 'Content-Type: application/json' \
  -d '{"prompt": "Why is the sky blue?", "max_tokens": 128}'
```

The spec's Docker path, and the concurrent (continuous-batching) stack:

```bash
docker build -t qwen-serve .
docker run --rm -p 8000:8000 \
  -v "$(pwd)/../section3_quantization/models:/models:ro" qwen-serve

docker compose up --build             # llama-server engine + API gateway
```

The section README explains which of the five run modes fits your machine
(Mac / Linux CPU / Linux GPU) and shows the load-test results for each.

**If `section3_quantization/models/Qwen3.5-4B-Q4_K_M.gguf` is missing**
(the GGUFs are multi-GB and may not travel with the repo): either run the
Section 3 notebook, which produces it locally — that's the section's whole
point — or, to unblock Section 4 alone, download a ready-made Q4_K_M of
[Qwen3.5-4B](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF) (~2.5 GB)
into that folder under that exact filename.

## Layout

```
section1_livekit/       STT → LLM → TTS agent, tools, failover, live webview
section2_rag/           LangChain + Chroma RAG over electropi.ai, guardrail, judge
section3_quantization/  Qwen3.5-4B BF16 → Q4_K_M locally, measured trade-offs
section4_deployment/    FastAPI + SSE serving of the Section 3 GGUF, Docker, load tests
```

Each folder is self-contained: its own `.venv`, `requirements.txt`,
`README.md` (quickstart + techniques + honest limitations) and `NOTES.md`
(the spec's written questions).
