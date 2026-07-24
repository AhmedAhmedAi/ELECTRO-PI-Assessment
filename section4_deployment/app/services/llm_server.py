"""llama-server backend: the inference-engine pattern with a GGUF.

One llama-server process holds ONE copy of the weights and N parallel KV
slots (`-np N`, continuous batching on). Decode steps from all active
requests share each forward pass — the weight streaming that dominates
generation cost is paid once per pass, not once per user. This service is
a thin async streaming client; no lock, no queue: concurrency is the
backend's job now.

Same public contract as the local backend (`stream()` yielding tokens,
`stats` dict), so routes don't change — swapping engines is a config flag.
"""
import json

import httpx

from app import config
from app.services.llm import PROMPT_TEMPLATE, STOP, SYSTEM_PROMPT


class LLMServerService:
    def __init__(self):
        self.base_url = config.LLAMA_SERVER_URL.rstrip("/")
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(300, connect=10))
        self.requests_served = 0
        self.load_seconds = 0.0            # server process owns model loading

    async def ready(self) -> dict:
        r = await self.client.get(f"{self.base_url}/health")
        r.raise_for_status()
        return r.json()

    async def stream(self, prompt: str, max_tokens: int, temperature: float,
                     stats: dict | None = None):
        """Async token generator backed by llama-server's SSE /completion."""
        self.requests_served += 1
        payload = {
            "prompt": PROMPT_TEMPLATE.format(system=SYSTEM_PROMPT, prompt=prompt),
            "n_predict": max_tokens,
            "temperature": temperature,
            "stop": STOP,
            "stream": True,
            "cache_prompt": True,          # reuse the shared system-prefix KV
        }
        async with self.client.stream(
            "POST", f"{self.base_url}/completion", json=payload
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                event = json.loads(line[6:])
                if event.get("content"):
                    yield event["content"]
                if event.get("stop"):
                    if stats is not None:
                        # With slots there is no app-side queue to report.
                        stats["queue_ms"] = 0
                    break
