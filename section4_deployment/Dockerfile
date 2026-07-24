# Multi-stage: llama-cpp-python compiles C++ in the builder; the runtime
# image ships only the installed packages (no compilers, ~400 MB smaller).
FROM python:3.11-slim AS builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11-slim
# libgomp: OpenMP runtime llama.cpp links against; not in slim by default.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home appuser
COPY --from=builder /install /usr/local
WORKDIR /srv
COPY app ./app

# The 2.5 GB model is NOT baked into the image — mount it at run time:
#   docker run -v /path/to/models:/models:ro -p 8000:8000 qwen-serve
ENV MODEL_PATH=/models/Qwen3.5-4B-Q4_K_M.gguf \
    N_GPU_LAYERS=0
# N_GPU_LAYERS=0: containers on macOS/most hosts have no GPU passthrough —
# CPU inference. On a CUDA host, build llama-cpp-python with CUBLAS instead.

USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
