"""App factory: load the model once at startup, serve until shutdown.

Run:  uvicorn app.main:app --host 0.0.0.0 --port 8000
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import config
from app.routes.generate import router
from app.services.llm import LLMService


@asynccontextmanager
async def lifespan(app: FastAPI):
    if config.LLM_BACKEND == "server":
        from app.services.llm_server import LLMServerService
        app.state.llm = LLMServerService()
        print(f"Backend: llama-server at {config.LLAMA_SERVER_URL} "
              f"({config.N_PARALLEL} slots, continuous batching)")
    else:
        print(f"Loading {config.MODEL_PATH.rsplit('/', 1)[-1]} "
              f"(ctx={config.N_CTX}, gpu_layers={config.N_GPU_LAYERS})...")
        app.state.llm = LLMService()
        print(f"Model ready in {app.state.llm.load_seconds}s")
    yield


app = FastAPI(
    title="Qwen3.5-4B Q4_K_M serving API (Section 4)",
    description="Serves the model quantized in Section 3 with SSE streaming.",
    lifespan=lifespan,
)
app.include_router(router)
