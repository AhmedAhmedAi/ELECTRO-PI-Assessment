"""Endpoints: streaming generation (SSE) and health/info."""
import json
import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app import config

router = APIRouter()


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=8000)
    max_tokens: int = Field(default=config.DEFAULT_MAX_TOKENS, ge=1, le=config.MAX_TOKENS_LIMIT)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = True


def sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/generate")
async def generate(body: GenerateRequest, request: Request):
    """Token-by-token SSE stream (default), or one JSON response with
    stream=false. Each stream ends with a stats event: TTFT, total time,
    token count — so every client sees the latency it actually got."""
    llm = request.app.state.llm

    async def event_stream():
        t_start = time.perf_counter()
        ttft_ms, n_tokens, stats = None, 0, {}
        try:
            async for token in llm.stream(body.prompt, body.max_tokens,
                                          body.temperature, stats):
                if ttft_ms is None:
                    ttft_ms = round((time.perf_counter() - t_start) * 1000)
                n_tokens += 1
                yield sse({"token": token})
        except Exception as e:
            yield sse({"error": str(e)[:200]})
            return
        total_ms = round((time.perf_counter() - t_start) * 1000)
        queue_ms = stats.get("queue_ms", 0)
        yield sse({"done": True, "ttft_ms": ttft_ms, "queue_ms": queue_ms,
                   "model_ttft_ms": (ttft_ms or 0) - queue_ms,
                   "total_ms": total_ms, "tokens": n_tokens})
        yield "data: [DONE]\n\n"

    if body.stream:
        return StreamingResponse(event_stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache"})

    # Non-streaming: collect the same generator into one JSON payload.
    t_start = time.perf_counter()
    ttft_ms, chunks = None, []
    try:
        async for token in llm.stream(body.prompt, body.max_tokens, body.temperature):
            if ttft_ms is None:
                ttft_ms = round((time.perf_counter() - t_start) * 1000)
            chunks.append(token)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:200])
    return {"text": "".join(chunks), "tokens": len(chunks), "ttft_ms": ttft_ms,
            "total_ms": round((time.perf_counter() - t_start) * 1000)}


@router.get("/health")
async def health(request: Request):
    llm = request.app.state.llm
    info = {
        "status": "ok",
        "backend": config.LLM_BACKEND,
        "model": config.MODEL_PATH.rsplit("/", 1)[-1],
        "n_ctx": config.N_CTX,
        "gpu_layers": config.N_GPU_LAYERS,
        "load_seconds": llm.load_seconds,
        "requests_served": llm.requests_served,
    }
    if config.LLM_BACKEND == "server":
        try:
            await llm.ready()
            info["slots"] = config.N_PARALLEL
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"llama-server down: {str(e)[:100]}")
    return info
