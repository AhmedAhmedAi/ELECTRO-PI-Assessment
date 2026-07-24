"""Load test: N concurrent streaming requests -> TTFT + total latency.

  python load_test.py                          # 10 concurrent vs localhost:8000
  python load_test.py --n 10 --url http://localhost:8000 --label native

TTFT here is measured at the CLIENT (first SSE token on the wire), which is
what a user actually feels — it includes HTTP overhead and, crucially, the
time spent queued behind other requests on the single llama.cpp instance.
Writes results/load_test_<label>.json and prints a summary table.
"""
import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx

PROMPTS = [
    "Explain what quantization does to a neural network in two sentences.",
    "Write a Python one-liner to reverse the words in a sentence.",
    "What is the capital of Japan? Answer in one word.",
    "Give three bullet points on why streaming APIs improve UX.",
    "Summarize what an SSE (server-sent events) stream is in one sentence.",
    "What is 17 * 23? Show only the result.",
    "Name two trade-offs of 4-bit quantization, briefly.",
    "Write a haiku about GPUs.",
    "In one sentence, what does a KV cache store?",
    "List two differences between REST and gRPC, briefly.",
]


async def one_request(client: httpx.AsyncClient, url: str, prompt: str, rid: int) -> dict:
    t0 = time.perf_counter()
    ttft_ms, tokens, server_stats = None, 0, {}
    async with client.stream(
        "POST", f"{url}/generate",
        json={"prompt": prompt, "max_tokens": 96, "temperature": 0.2},
    ) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            if "token" in event:
                if ttft_ms is None:
                    ttft_ms = round((time.perf_counter() - t0) * 1000)
                tokens += 1
            elif event.get("done"):
                server_stats = event
    total_ms = round((time.perf_counter() - t0) * 1000)
    gen_ms = total_ms - server_stats.get("queue_ms", 0)
    return {
        "request": rid,
        "client_ttft_ms": ttft_ms,          # includes queue time — what users feel
        "client_total_ms": total_ms,
        "queue_ms": server_stats.get("queue_ms"),        # waiting behind others
        "model_ttft_ms": server_stats.get("model_ttft_ms"),  # once at the model
        "tokens": tokens,
        "gen_tokens_per_s": round(tokens / (gen_ms / 1000), 1) if gen_ms > 0 else 0,
    }


async def main(n: int, url: str, label: str):
    async with httpx.AsyncClient(timeout=600) as client:
        # sanity + model warm-up (first request pays one-off graph init)
        health = (await client.get(f"{url}/health")).json()
        print(f"Target: {url} — {health['model']} (gpu_layers={health['gpu_layers']})")
        await one_request(client, url, "Say OK.", -1)

        t0 = time.perf_counter()
        results = await asyncio.gather(*[
            one_request(client, url, PROMPTS[i % len(PROMPTS)], i) for i in range(n)
        ])
        wall_s = round(time.perf_counter() - t0, 1)

    ttfts = [r["client_ttft_ms"] for r in results]
    totals = [r["client_total_ms"] for r in results]
    throughput = round(sum(r["tokens"] for r in results) / wall_s, 1)
    summary = {
        "label": label, "concurrency": n, "wall_seconds": wall_s,
        "ttft_ms": {"min": min(ttfts), "median": int(statistics.median(ttfts)),
                    "p90": int(sorted(ttfts)[int(0.9 * (n - 1))]), "max": max(ttfts)},
        "total_ms": {"min": min(totals), "median": int(statistics.median(totals)),
                     "max": max(totals)},
        "model_ttft_ms": {"median": int(statistics.median(r["model_ttft_ms"] for r in results))},
        "aggregate_tokens_per_s": throughput,
        "health": health,
        "requests": results,
    }

    print(f"\n{n} concurrent requests finished in {wall_s}s "
          f"(aggregate {throughput} tok/s)")
    print(f"{'req':>3} {'client TTFT':>12} {'queued':>9} {'model TTFT':>11} {'total':>9} {'gen tok/s':>9}")
    for r in sorted(results, key=lambda r: r["client_ttft_ms"]):
        print(f"{r['request']:>3} {r['client_ttft_ms']:>10}ms {r['queue_ms']:>7}ms "
              f"{r['model_ttft_ms']:>9}ms {r['client_total_ms']:>7}ms {r['gen_tokens_per_s']:>9}")
    print(f"\nTTFT   min {summary['ttft_ms']['min']}ms · median {summary['ttft_ms']['median']}ms "
          f"· p90 {summary['ttft_ms']['p90']}ms · max {summary['ttft_ms']['max']}ms")
    print(f"total  min {summary['total_ms']['min']}ms · median {summary['total_ms']['median']}ms "
          f"· max {summary['total_ms']['max']}ms")

    out = Path(__file__).parent / "results" / f"load_test_{label}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--label", default="native")
    args = ap.parse_args()
    asyncio.run(main(args.n, args.url, args.label))
