# Write-up: GPTQ/AWQ vs bitsandbytes vs GGUF for production

I will always choose AWQ/GPTQ over bitsandbytes or GGUF for inference-engine
deployments — by "production" here I mean serving a model over an inference
engine (vLLM, SGLang) to many concurrent users. Especially AWQ, because it
preserves quality closest to the original at 4-bit — it scales up the
weights that activations rely on most before quantizing, so the model's
core reasoning survives compression.

In my previous production deployments, I have already used AWQ with
inference engines like SGLang (for example, running AWQ on models like
Qwen). Because AWQ and GPTQ are natively built for inference engines like
SGLang and vLLM, they allow me to handle every detail of the serving
pipeline — from managing concurrent GPUs and batch sizes to tuning queue
handling for maximum throughput.

On the other hand, I personally use **GGUF** a lot, but I keep it on my
personal machine rather than using it for server production. Since my main
setup is a MacBook with an Apple M-series chip, GGUF is the most convenient
format for unified memory because it runs easily across CPU, GPU, and Apple
Metal. It is single-file, highly portable, and allows easy layer offloading,
but because it isn't natively built for high-concurrency CUDA inference
engines like SGLang or vLLM, I don't use it for production APIs.

Finally, regarding **bitsandbytes**, I haven't used it much myself in my
day-to-day work since I don't do heavy QLoRA fine-tuning often. But its
design makes it the right pick for experimentation and fine-tuning on a
budget: it quantizes the model at load time into 4-bit NF4 (no
pre-quantized artifact to manage), freezes the compressed base, and trains
only small 16-bit LoRA adapters on top. Gradients touch only those 16-bit
adapters, so updates don't get rounded away — which is what lets a large
model fine-tune on limited GPU hardware instead of an expensive cluster.
