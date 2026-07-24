"""llama.cpp wrapper: one loaded model, one request at a time, async queue.

llama.cpp has no continuous batching — a single Llama instance must run one
generation at a time (concurrent calls corrupt the KV cache). So the service
serializes inference with an asyncio lock: concurrent HTTP requests queue
here, and the load test measures exactly what that queueing costs. The
honest scaling answer (batching engines, more replicas) is in
NOTES.md.
"""
import asyncio
import time
from queue import SimpleQueue

from llama_cpp import Llama

from app import config

# Qwen3.5 ChatML with an empty think-block prefill — pins the hybrid
# reasoning mode OFF (the llama-cli equivalent is `--reasoning off`), so
# latency numbers measure answering, not variable-length hidden thinking.
PROMPT_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{prompt}<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)
SYSTEM_PROMPT = "You are a helpful, concise assistant."
STOP = ["<|im_end|>"]
_SENTINEL = object()


class LLMService:
    def __init__(self):
        t0 = time.perf_counter()
        self.llm = Llama(
            model_path=config.MODEL_PATH,
            n_ctx=config.N_CTX,
            n_threads=config.N_THREADS,
            n_gpu_layers=config.N_GPU_LAYERS,
            verbose=False,
        )
        self.load_seconds = round(time.perf_counter() - t0, 1)
        self.lock = asyncio.Lock()          # the request queue
        self.requests_served = 0

    def _generate_sync(self, prompt: str, max_tokens: int, temperature: float, out: SimpleQueue):
        """Runs in a worker thread; pushes token strings, then the sentinel."""
        try:
            stream = self.llm(
                PROMPT_TEMPLATE.format(system=SYSTEM_PROMPT, prompt=prompt),
                max_tokens=max_tokens,
                temperature=temperature,
                stop=STOP,
                stream=True,
            )
            for part in stream:
                out.put(part["choices"][0]["text"])
            out.put(_SENTINEL)
        except Exception as e:               # surfaced to the SSE stream
            out.put(e)

    async def stream(self, prompt: str, max_tokens: int, temperature: float,
                     stats: dict | None = None):
        """Async token generator. Holds the model lock for the whole request.
        If given, `stats` receives queue_ms — time spent waiting for the model
        behind other requests (the number that explains all tail latency)."""
        t_enqueue = time.perf_counter()
        async with self.lock:
            if stats is not None:
                stats["queue_ms"] = round((time.perf_counter() - t_enqueue) * 1000)
            self.requests_served += 1
            out: SimpleQueue = SimpleQueue()
            loop = asyncio.get_running_loop()
            task = loop.run_in_executor(
                None, self._generate_sync, prompt, max_tokens, temperature, out
            )
            while True:
                item = await loop.run_in_executor(None, out.get)
                if item is _SENTINEL:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
            await task
