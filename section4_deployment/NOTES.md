# Scaling to 50 concurrent users — what I would change

The first thing I would change is the engine, not the API. My simple
in-process backend runs one request at a time, and the load test shows
what that costs: the model itself answers in ~210 ms, but the 10th user
waits 15 seconds in line. I already proved the fix on this laptop —
switching to llama-server with 4 batched slots cut worst TTFT from 15 s
to 3.7 s and doubled throughput. Production is the same idea with real
GPU headroom: an inference engine, and SGLang is my preference (vLLM
also works, but I have deployed on SGLang before). That choice also
changes the model format: engines don't take GGUF, so I would
re-quantize to AWQ, their native quantized format. The API layer survives
this swap — in this repo changing engines touched one service file.

Before touching any config, I would do the capacity math. VRAM minus the
weights gives me the KV cache budget, and dividing that by KV-per-token
times the expected context length tells me how many users one GPU really
holds. On a 24 GB card with ~3 GB of AWQ weights, that comes out to
roughly 33 users at a full 4k context, and more at realistic lengths.
That number drives everything else:

- Queueing by token budget, not by request count. The queue is bounded,
  and overflow gets a fast 429 with Retry-After instead of a silent wait.
- Agent requests that pause for a tool call free their compute for other
  users in the meantime — extra capacity for free.
- If long-context requests fill the budget and 50 users don't fit, a
  second GPU takes the overflow. Autoscaling watches queue depth and
  TTFT p95, not CPU — my numbers show the wait explodes in the queue
  long before compute runs out.
- Caching mostly comes with the engine: SGLang's RadixAttention caches
  the shared system prompt once and keeps each conversation's KV warm
  across turns until the session goes quiet. In front of it, a semantic
  cache (like the one I built in Section 2) answers repeated questions
  without a generation.

Finally, keep SSE streaming and watch the right metrics — TTFT p95, queue
depth, tokens/sec per replica. The service already reports `queue_ms` and
`model_ttft_ms` on every response, so that monitoring is free.
