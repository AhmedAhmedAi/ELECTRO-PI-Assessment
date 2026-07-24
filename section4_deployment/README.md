# Section 4 — Model Deployment

Serves the **Q4_K_M GGUF I quantized in Section 3** (Qwen3.5-4B, 2.51 GiB)
behind a FastAPI REST API with token-by-token SSE streaming, a working
Dockerfile + compose stack, and a load test that measures five serving
configurations. The spec write-up (what changes for 50 concurrent users)
is in [NOTES.md](NOTES.md).

## Why FastAPI + llama-cpp-python (the spec asks to justify)

The artifact being deployed is the GGUF from Section 3, and that decides
the stack:

- **GGUF's native runtime is llama.cpp.** vLLM's GGUF support is
  experimental and not recommended by its own docs; SGLang does not take
  GGUF at all — their quantized formats are AWQ/GPTQ.
- **vLLM/SGLang are CUDA-first.** No Apple-Silicon support, and a Docker
  container on macOS gets no GPU passthrough — `docker run` would not
  work end-to-end on the machine this is developed and graded on.
- **FastAPI over llama.cpp's built-in server** because the spec wants *my*
  REST API: typed request validation (pydantic), an SSE contract I
  control, per-request queue/TTFT metrics, and a `services/` layer that
  can swap llama.cpp for an SGLang client without touching routes.

This is the same format-to-runtime matching argument as my Section 3
write-up: GGUF for CPU/Mac/edge serving, AWQ/GPTQ on an inference engine
for GPU concurrency. Section 4 is the GGUF case; [NOTES.md](NOTES.md) is
where the engine swap happens.

## Two backends, one API

| Backend | What it is | When |
|---|---|---|
| `local` (default) | llama-cpp-python in-process; one request at a time behind a measured queue | simplest single-container deploy |
| `server` | proxies **llama-server**: ONE weight copy, 4 KV-cache slots, continuous batching — the inference-engine pattern | concurrency; the compose stack |

Routes never change — `services/` swaps engines via `LLM_BACKEND`. That
boundary is the whole point of the layout: swapping llama.cpp for an
SGLang client in production touches one file.

Design notes: the model loads once (FastAPI lifespan) and every request
shares it; streaming is SSE with a final stats event so every client sees
the latency it actually got (`stream: false` gives plain JSON from the
same path); and thinking is pinned **off** via a ChatML prefill (same as
Section 3's `--reasoning off`) — latency numbers measure answering, not
variable-length hidden reasoning.

## API

| Endpoint | What |
|---|---|
| `POST /generate` | SSE stream of `{"token": ...}` events; final stats event: `ttft_ms`, `queue_ms`, `model_ttft_ms`, `total_ms`, `tokens`; `"stream": false` → one JSON payload |
| `GET /health` | backend, model, slots, load time, requests served |

```bash
curl -N -X POST localhost:8000/generate -H 'Content-Type: application/json' \
  -d '{"prompt": "Why is the sky blue?", "max_tokens": 128}'
```

## Run it

```bash
# Native (Metal GPU), single-stream backend:
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000

# Native, concurrent backend (one weight copy, 4 slots):
llama-server -m ../section3_quantization/models/Qwen3.5-4B-Q4_K_M.gguf \
             -c 16384 -np 4 -ngl 99 --port 8080 &
LLM_BACKEND=server .venv/bin/uvicorn app.main:app --port 8000

# Docker, single container (spec's docker build && docker run):
docker build -t qwen-serve .
docker run --rm -p 8000:8000 \
  -v "$(pwd)/../section3_quantization/models:/models:ro" qwen-serve

# Docker, concurrent stack (llama-server + API gateway):
docker compose up --build

# GPU hosts (Linux + NVIDIA + nvidia-container-toolkit) — swaps only the
# engine container for a CUDA build (untested here: no NVIDIA hardware):
COMPOSE_PROFILES=gpu LLAMA_URL=http://llama-gpu:8080 docker compose up --build

# Hybrid (Mac): engine native on Metal, API in the container —
# containers on macOS get no GPU, so the engine stays outside:
llama-server -m ../section3_quantization/models/Qwen3.5-4B-Q4_K_M.gguf \
             -c 16384 -np 4 -ngl 99 --host 0.0.0.0 --port 8080 &
docker run --rm -p 8000:8000 -e LLM_BACKEND=server \
  -e LLAMA_SERVER_URL=http://host.docker.internal:8080 qwen-serve
```

**Which way should you run it?** Linux+GPU: compose gpu profile. Linux
CPU: compose (no VM tax). Mac: native is fastest; hybrid if you want the
API containerized; all-Docker only to verify the container works.

The 2.5 GB model is volume-mounted, never baked into an image (api image:
434 MB, multi-stage, non-root, healthcheck, env-var config).

## Load test — 10 concurrent streaming requests, four configurations

`load_test.py` measures client TTFT, per-request queue time, model TTFT,
and total latency (raw JSON in `results/`, five configurations):

| configuration | wall clock | aggregate | TTFT median | TTFT max |
|---|---|---|---|---|
| Native · serialized | 17.5 s | 23.7 tok/s | 8.3 s | 15.0 s |
| **Native · 4 slots, batched** | **7.7 s** | **52.5 tok/s** | **1.3 s** | **3.7 s** |
| Docker (CPU) · serialized | 22.8 s | 17.9 tok/s | 10.5 s | 20.3 s |
| Docker (CPU) · 4 slots, batched | 13.4 s | 28.9 tok/s | 2.9 s | 10.2 s |
| Hybrid: container API + native Metal engine | 12.0 s | 32.8 tok/s | 1.8 s | 5.4 s |

**Finding 1 — the queue is the whole latency story.** Serialized, the
model's own TTFT stayed ~210 ms flat for all 10 requests while `queue_ms`
grew linearly to ~15 s: latency was concurrency, not the model.

**Finding 2 — continuous batching fixes it, when hardware has headroom.**
Same laptop, same GGUF, llama-server with 4 slots: TTFT max **15 s → 3.7 s**
and aggregate throughput **more than doubled** (23.7 → 52.5 tok/s), because
batched decode steps share each pass's weight streaming — the memory-
bandwidth argument from Section 3, now measured at the serving layer.

### Slot tuning — swept, not guessed (native, 10 concurrent)

| slots | 2 | 3 | **4** | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|
| total tok/s | 35.5 | 36.9 | **52.5** | 41.3 | 34.6 | 32.1 | 33.6 |
| TTFT max | 8.2 s | 8.9 s | **3.7 s** | 5.1 s | 4.6 s | 3.5 s | 2.6 s |

**4 slots is the optimum on this chip.** Fewer -> requests still queue.
More -> decodes fight over memory bandwidth and total speed falls (the
only gain past 4 is fairness: 8 slots starts everyone within 2.6 s, at a
36% speed cost). Raw runs: `results/load_test_native_np*.json`.

**Finding 3 — a benchmark is only as good as its build.** The first two
batched-Docker runs *lost* to serialized (10–15 tok/s) and nearly led to
the wrong architectural conclusion. The real culprits: the official
arm64 llama.cpp image is a portable build (no dotprod/i8mm kernels, ~5×
slow), and `GGML_NATIVE` CPU probing is flaky inside VMs (the same
Dockerfile compiled fast once and slow the next time). With features
pinned explicitly ([Dockerfile.llama](Dockerfile.llama)), batching wins
on CPU too — 1.6× throughput, TTFT median 10.5 s → 2.9 s. The gain is
smaller than Metal's 2.2× because CPU decode has less spare parallel
headroom — the scaling gradient (CPU < Metal < GPU) that drives the
50-user write-up in [NOTES.md](NOTES.md).

## Container lessons that cost real debugging time

- An unbounded `make -j` OOM-killed the Docker VM (both engine builds now
  pin `-j 4`), and the CUDA image's configure stage is smoke-tested in its
  exact base image, but it has never been compiled or run on a GPU — no
  NVIDIA hardware here, stated on the image itself.
- **If I published the image** (instead of shipping a Dockerfile), I
  would build multi-arch with buildx
  (`--platform linux/amd64,linux/arm64`) so every puller gets a native
  image — `--platform` is a compatibility tool, not a speed tool, and
  forcing amd64 on ARM means slow emulation.

## Honest caveats

- **Docker on macOS is CPU-only** (Linux VM, no Metal passthrough). I
  expected a several-fold slowdown; the measured gap is smaller — ~1.3×
  on aggregate throughput, ~2× on model TTFT — because llama.cpp's ARM
  NEON CPU path is strong on this chip. Both environments are load-tested
  side by side, and both show the identical queueing signature (flat
  model TTFT, linear queue growth).
- The `local` backend is capped at one request's generation speed
  regardless of concurrency — measured, owned, and answered by the
  `server` backend and the 50-user write-up.
- The client-side load test runs on the same machine as the server, so it
  slightly competes for CPU; TTFT/queue numbers dwarf that noise.
- `n_ctx=4096` per request keeps memory modest; long-context serving
  would need explicit KV budgeting per concurrent slot.

## Layout

```
app/
  main.py                app factory — backend chosen at startup
  config.py              all knobs via env vars (12-factor style)
  routes/generate.py     POST /generate (SSE) + GET /health
  services/llm.py        local backend: in-process + measured queue
  services/llm_server.py server backend: llama-server client (slots)
Dockerfile               api image (multi-stage, non-root, healthcheck)
Dockerfile.llama         llama-server, CPU build w/ pinned SIMD features
Dockerfile.llama.cuda    llama-server, CUDA build (GPU hosts; untested here)
docker-compose.yml       concurrent stack; cpu/gpu engine via profiles
load_test.py             concurrency test harness -> results/
results/                 12 load-test JSONs (5 configs + slot sweep 2-8)
NOTES.md                 the spec write-up (50 concurrent users)
```
