"""Every knob of the serving stack, overridable via environment variables
(the Docker way: same image, config injected at run time)."""
import os
from pathlib import Path

# Default points at the Section 3 artifact for native runs; in Docker the
# model is volume-mounted and MODEL_PATH is set to /models/... instead.
_DEFAULT_MODEL = (
    Path(__file__).resolve().parents[2]
    / "section3_quantization" / "models" / "Qwen3.5-4B-Q4_K_M.gguf"
)

# Backend: "local"  = in-process llama-cpp-python, one request at a time
#          "server" = proxy to llama-server (one weight copy, N KV slots,
#                     continuous batching — the inference-engine pattern)
LLM_BACKEND = os.getenv("LLM_BACKEND", "local")
LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:8080")
N_PARALLEL = int(os.getenv("N_PARALLEL", "4"))     # slots (server backend)

MODEL_PATH = os.getenv("MODEL_PATH", str(_DEFAULT_MODEL))
N_CTX = int(os.getenv("N_CTX", "4096"))          # context window per request
N_THREADS = int(os.getenv("N_THREADS", "0")) or None   # 0 -> llama.cpp auto
N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", "-1"))    # -1 = all (Metal); 0 in Docker
MAX_TOKENS_LIMIT = int(os.getenv("MAX_TOKENS_LIMIT", "1024"))
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "256"))
