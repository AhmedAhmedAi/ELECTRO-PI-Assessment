# Section 3 — Quantization

Take one model, quantize it **myself**, and measure what 4-bit actually
costs and buys: RAM, speed, and answer quality on 5 fixed prompts.

- **Model**: [Qwen3.5-4B](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF)
  (Apache 2.0, dense 4.21B — the strongest small dense model at time of
  writing, and small enough that the BF16 *baseline* still fits in 16 GB,
  which is the constraint that rules out bigger/E-series models here).
- **Method**: download the BF16 GGUF once, produce the Q4_K_M **locally**
  with `llama-quantize` (23 s on an M5), then benchmark both with the same
  llama.cpp build (Metal). Q4_K_M is a *k-quant*: most weights go to 4-bit
  blocks with per-block scales, while the attention/output tensors that
  hurt the most stay at higher precision — 5.13 bits/weight effective,
  which is why it's the community default: nearly Q5 quality at nearly
  Q4 size.
- **Hardware**: MacBook Pro M5, 16 GB unified memory, macOS.

## Run it

Everything runs from the notebook — download, quantization, all
measurements — and every step writes its raw artifact to `results/`.

```bash
brew install llama.cpp                      # llama-cli, llama-bench, llama-quantize
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/jupyter execute analysis.ipynb    # or open it in Jupyter Lab
```

Notes: the results are readable in 0 minutes — the notebook ships
pre-executed with real outputs. A full reproduction takes ~15 min of
compute plus the one-time 8 GB model download (skipped when present).
Step 8 (blind judge) needs `GOOGLE_API_KEY` in the environment; it can
also run standalone: `GOOGLE_API_KEY=... .venv/bin/python judge.py`.

**[analysis.ipynb](analysis.ipynb)** is the pipeline itself — each step is
an executed cell with its measurement code visible (ships pre-executed with real outputs).

## Results

### Size & RAM (identical 8k context for both)

| | BF16 | Q4_K_M | change |
|---|---|---|---|
| Weights (= file on disk) | 7.84 GiB | 2.51 GiB | **−68%** |
| KV cache + state + compute (8k ctx) | 412 MiB | 412 MiB | unchanged |
| **Unique inference RAM, one session** | **~8.24 GiB** | **~2.92 GiB** | **−65%** |

Accounting note (a good catch during review): llama.cpp's per-backend buffer
log sums to *more* than the file (9236 / 3070 MiB) because the **tied
token-embedding tensor** (248320 × 2560 = 1212 MiB in BF16, 497 MiB in Q4)
is mapped by BOTH the CPU backend (input lookup) and Metal (output head) —
two views, one set of mmap-shared physical pages. Unique bytes = file size;
what inference genuinely *adds* on top is the KV cache and compute buffers.
What the −65% means in practice: BF16 does not fit next to anything else
on a 16 GB laptop; Q4 fits with room for a browser, an IDE, and Section
4's Docker VM.

### Speed (llama-bench, 5 reps, Metal)

| test | BF16 | Q4_K_M | change |
|---|---|---|---|
| Prompt processing (pp512) | 954.6 t/s | 1013.5 t/s | ≈ same |
| **Generation (tg128)** | **13.0 t/s** | **35.0 t/s** | **2.7×** |

Generation is memory-bandwidth-bound — every token streams all weights
through the memory bus, so weights ⅓ the size ⇒ ~3× fewer bytes ⇒ 2.7×
faster. Prompt processing is compute-bound (batch matmuls), so quantization
barely moves it. This asymmetry is the whole speed story of quantization.

**Measurement conditions matter on laptops** (measured, not guessed): on
battery, or with background load (video decoding), absolute throughput
drops roughly 2× (Q4 gen: 35 → 18–24 t/s) while the BF16-vs-Q4 *ratio*
stays at 2.5–2.9×. The table above is this notebook's executed run (AC
power); the ratio held in every condition measured.

### Time-to-first-token — added beyond the task requirements

(The metric users actually feel — the silence before the first word — and a Section 4 requirement baselined here early.)

| | BF16 | Q4_K_M |
|---|---|---|
| Prefill (732-token prompt) | 890.9 ms | 773.7 ms |
| Per-token decode | 67.3 ms | 26.0 ms |
| **TTFT** | **958 ms** | **800 ms** |

The finding worth knowing: **TTFT improves far less than generation** —
~17% here vs 2.7× on decode — because TTFT is dominated by prefill, which
is compute-bound and barely benefits from smaller weights. So quantize
for RAM and generation speed; the first-token gain is a modest side
benefit, not the headline.

### Blind LLM-judge — added beyond the task requirements

(Rubric-based pairwise judging is the standard approach in model evaluation and RLHF preference pipelines — MT-Bench/Arena style. gemini-3.5-flash, temp 0, blinded + randomized A/B.)

Beyond eyeballing: a blind judge scored both models' answers per prompt on
instruction following, truthfulness, conciseness, writing style, and
helpfulness (1–5), then picked a winner on a 7-point A/B scale — never
knowing which model was which (`judge.py`; raw verdicts with the blinding
map in `results/judge_results.json`, tables in `results/judge_summary.md`).

Verdicts: **4 ties + BF16 "better" on the Arabic prompt** — the judge
independently cited the Q4 answer's "broken, nonsensical Arabic" without
knowing it was the quantized model, confirming the human review. Rubric
averages (BF16 / Q4): truthfulness 4.4/4.0, writing style 4.4/4.0,
helpfulness 4.0/3.6, instruction following 4.0/4.2, conciseness 4.8/4.6.

### Quality (same 5 prompts, greedy/temp 0, raw outputs in `results/`)

| prompt | BF16 | Q4_K_M | verdict |
|---|---|---|---|
| Factual + length constraint | 2 sentences — missed the "exactly 3" constraint | ✓ 3 sentences, correct | Q4 edges it |
| Math (exact answer 5170) | ✓ 5170, clean steps | ✓ 5170, clean steps | tie |
| Code (O(n) merge) | ✓ correct algorithm | ✓ correct algorithm | tie* |
| JSON extraction | ✓ exact, valid JSON | ✓ exact, valid JSON | tie |
| **Egyptian-Arabic reply** | coherent, minor slips | **broken phrasing** | **BF16 wins** |

\* both put doctests in a syntactically odd place — a model limitation,
identical across precisions, so not a quantization effect.

**The finding:** on mainstream tasks (English, math, code, JSON) Q4_K_M is
indistinguishable from BF16 at greedy decoding — the one length-constraint
slip in the set was the *baseline's*, not the quantized model's. The degradation surfaces
first in the model's *weakest* capability — Egyptian dialect — where the
Q4 output drifts into ungrammatical Arabic while BF16 stays coherent.
Quantization doesn't shave quality uniformly; it erodes the tails first.
The practical rule that follows: **evaluate quantized models on your own
weakest-case workload, not on standard English benchmarks** — the
benchmarks are exactly where the loss hides.

## The trade-off in one table (spec deliverable)

| | BF16 | Q4_K_M | change |
|---|---|---|---|
| Precision | 16-bit | ~5.1 bit effective | — |
| Size (file = weights) | 7.84 GiB | 2.51 GiB | **−68%** |
| Total inference RAM (8k ctx) | ~8.24 GiB | ~2.92 GiB | **−65%** |
| Speed — generation | 13.0 t/s | 35.0 t/s | **2.7×** |
| Speed — prompt processing | ~955 t/s | ~1013 t/s | ≈ same |
| Speed — TTFT (732-tok prompt) | ~958 ms | ~800 ms | **−17%** |
| Quality — EN/math/code/JSON | reference | indistinguishable | tie |
| Quality — Egyptian Arabic | coherent | broken phrasing | degraded |

## Verdict

For this model on this machine, Q4_K_M is the obvious serving choice:
**65% less RAM, 2.7× faster generation, no measurable loss on core tasks** —
with the caveat that low-resource-language quality is the canary to test
before shipping. The Q4 file produced here is what Section 4 deploys.

## Honest caveats

- 5 prompts is a smoke test, not an eval; perplexity over a corpus or a
  task suite would quantify the gap. The point here is the methodology and
  the measured deltas, all reproducible by re-executing `analysis.ipynb`
  (raw artifacts in `results/`).
- macOS `time -l` RSS was misleading for RAM (Metal maps the working-set
  ceiling, ~11.4 GiB, for *both* models); the honest numbers are
  llama.cpp's own buffer allocations at a pinned context, which is what
  this README reports.
- Qwen3.5 is a hybrid-reasoning model; thinking was pinned **off** so
  outputs are deterministic and comparable — with reasoning on,
  variable thinking traces would differ across precisions for reasons
  unrelated to weight quality.
- Gemma 4 "4B" (E4B) was considered and rejected: it is 8B raw parameters
  (BF16 GGUF 15.1 GiB) — the fp16 baseline would not run honestly in
  16 GB, and a comparison where the baseline swaps is not a comparison.

Spec write-up (GPTQ/AWQ vs bitsandbytes vs GGUF for production):
[NOTES.md](NOTES.md). Raw outputs: `results/`.
