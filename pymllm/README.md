# pymllm

![pymllm-arch](../assets/pymllm-arch.png)

## Overview

`pymllm` is mllm's Python / CUDA inference and serving runtime, running mainly
on NVIDIA Jetson Orin edge GPUs (Orin NX / AGX Orin). It is adapted for the INT8
throughput of the Orin Ampere Tensor Cores, supports BF16 native inference plus
two quantization schemes (W4A16 and W8A8_INT8), and currently covers Qwen3,
Qwen3-VL, and Qwen3.5, exposing an OpenAI-compatible HTTP API.

## Environment

A known-good set of versions:

| Component | Version / notes |
| --- | --- |
| JetPack / Jetson Linux | JetPack `6.2.1` / Jetson Linux `36.4.4` (L4T `R36.4.4`) |
| Python | `3.10.12` |
| PyTorch | `2.4.0` |
| torchvision | `0.19.0a0+48b1edf` |
| transformers | `5.3.0` |
| safetensors | `0.7.0` |
| flashinfer | `0.6.7` |
| Triton Language | `triton==3.6.0` aarch64 wheel |
| CUDA | `12.6` |
| GPU | Jetson Orin NX, SM87 |

## Install

Clone the repo, then install `pymllm` and `mllm-kernel` from the repo root:

```bash
cd <repo-root>
SKBUILD_WHEEL_CMAKE=false python3 -m pip install -e .
python3 -m pip install -e <repo-root>/mllm-kernel --no-deps --no-build-isolation
```

`triton` and `flashinfer` have two sources; pick either one:

```bash
# Option 1: Jetson wheels from Jetson AI Lab.
python3 -m pip install --extra-index-url https://pypi.jetson-ai-lab.io/ triton flashinfer

# Option 2: pin Triton from official PyPI, still get FlashInfer from Jetson AI Lab.
python3 -m pip install --index-url https://pypi.org/simple triton==3.6.0
python3 -m pip install --extra-index-url https://pypi.jetson-ai-lab.io/ flashinfer
```

On aarch64, whether the Triton wheel works out of the box mostly depends on the
wheel source and the `ptxas` / `cuda.h` lookup paths. In the validated
environment above, the official PyPI `triton==3.6.0` manylinux aarch64 wheel is
closest to working out of the box; if a Jetson AI Lab wheel hits `ptxas` or CUDA
header lookup issues, setting `TRITON_PTXAS_PATH` and `CPATH` explicitly usually
fixes it. After installing, run a smoke test with a minimal kernel such as
`per_token_quant_int8` to confirm Triton actually compiles.

## W8A8 first-run JIT compilation

The W8A8 INT8 GEMM goes through CUTLASS and needs CUTLASS headers. No extra
setup is required by default — `flashinfer` ships a bundled CUTLASS; set
`CUTLASS_HOME` if you want to point at your own copy.

The first W8A8 kernel call triggers a one-time JIT compile, cached at:

```text
~/.cache/mllm_kernel/cutlass_int8_scaled_mm/
```

Later runs reuse the cache. To re-check the first-compile behavior, delete this
directory and run again.

## Launch the server

The entry point is `pymllm.server.launch`. Once up, it serves `/health`,
`/v1/models`, `/v1/completions`, `/v1/chat/completions`, and `/generate`.

W4A16 / W8A8 quantized models and BF16 base models share the same entry point;
the runtime reads the quantization config in `config.json` and picks the W4A16
or W8A8 path automatically. A typical quantized-model launch:

```bash
cd <repo-root>

python3 -m pymllm.server.launch \
  --server.model_path <quantized-model-path> \
  --server.dtype float16 \
  --quantization.method compressed-tensors \
  --server.host 0.0.0.0 \
  --server.port 30000 \
  --server.mem_fraction_static 0.8 \
  --server.max_running_requests 1 \
  --server.max_total_tokens 4096 \
  --server.disable_radix_cache \
  --server.log_level debug
```

For BF16 / FP16 base models, use the same command and drop
`--quantization.method`.

## Common parameters

| Parameter | Description |
| --- | --- |
| `--server.model_path` | Model weight directory, usually HuggingFace or ModelScope format. |
| `--server.tokenizer_path` | Tokenizer directory; defaults to `model_path` when unset, so you rarely pass it. |
| `--server.dtype` | Runtime dtype: `auto`, `float16`, or `bfloat16`. |
| `--quantization.method compressed-tensors` | Enables `compressed-tensors` weight loading and the quantized linear path. |
| `--server.mem_fraction_static` | Static budget for `model weights + KV cache pool` as a fraction of total GPU memory. Too small fails to allocate the KV pool at startup; too large leaves no room for activations and CUDA Graph. On Jetson, Qwen3-VL-2B usually starts around `0.5`–`0.8`. |
| `--server.max_running_requests` | Concurrent requests. On small-VRAM Jetson, start from `1`. |
| `--server.max_total_tokens` | Upper bound on the KV cache token pool, shared globally across the worker (not a per-request limit). Actual capacity is `min(profiled tokens, max_total_tokens)` and does not bypass the memory profile. |
| `--server.disable_radix_cache` | Disables Radix Cache, uses `ChunkCache` instead. |

## OpenAI-compatible requests

Health check:

```bash
curl -s --noproxy '*' http://127.0.0.1:30000/v1/models ; echo
```

Text request:

```bash
curl -s --noproxy '*' http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Reply with: ok"}],
    "max_tokens": 8,
    "temperature": 0.0,
    "stream": false
  }' ; echo
```

For image requests, use an absolute path the server process can read; do not use
the `file://` prefix:

```bash
cat > /tmp/mm_req_path.json <<'JSON'
{
  "model": "default",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Please describe this image."},
        {"type": "image_url", "image_url": {"url": "/workspace/test.png"}}
      ]
    }
  ],
  "max_tokens": 128,
  "temperature": 0.0,
  "stream": false
}
JSON

curl -s --noproxy '*' http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  --data @/tmp/mm_req_path.json ; echo
```

## Benchmark

`bench_one_batch` is a low-level offline benchmark. It initializes
`pymllm.executor.model_runner.ModelRunner` directly, bypassing the HTTP server,
tokenizer, scheduler, and detokenizer, and only measures one static prefill plus
per-token decode of the model itself. It is good for analyzing model forward, KV
cache, attention, CUDA Graph, and quantized kernels at the model level, and for
checking graph optimizations such as fused projection and residual-carry. It
does not measure online-serving TTFT / ITL / E2E — don't mix the two.

`bench_one_batch` supports three measurement modes:

- **Text-only**: prefill / decode with synthetic token ids.
- **Vision encoding (`vit_prefill`)**: a synchronized wall clock around the
  vision encoder (`self.visual(...)`) only, reflecting pure vision-encode speed.
- **Multimodal prefill (`multimodal_prefill`)**: covers "vision encoding + LLM
  prefill over image/text tokens", reflecting full multimodal prefill speed.

Text-only:

```bash
PYTHONPATH="$PWD:$PWD/mllm-kernel" python3 -m pymllm.bench_one_batch \
  --server.model_path <model-or-quantized-model-path> \
  --server.dtype float16 \
  --quantization.method compressed-tensors \
  --server.mem_fraction_static 0.8 \
  --server.max_running_requests 1 \
  --server.max_total_tokens 2048 \
  --batch-size 1 \
  --input-len 256 512 1024 \
  --output-len 128 \
  --result-filename <result-jsonl-path>
```

`--batch-size`, `--input-len`, and `--output-len` all accept multiple values;
the script sweeps every combination and appends results to the JSONL file.
`output_len` uses total-output-token semantics: the first next token is already
produced after prefill, so the decode loop runs `output_len - 1` more steps.

Multimodal prefill: pass a real image to `--image`, and when you also pass
`--input-len` explicitly, the length means the target total of
`image placeholder tokens + text prompt tokens` — the script only pads or
truncates text tokens, never image tokens. So you can sweep different totals such
as `314/512/1024/2048` on the same image to measure full multimodal prefill
including vision encoding:

```bash
PYTHONPATH="$PWD:$PWD/mllm-kernel" python3 -m pymllm.bench_one_batch \
  --server.model_path <qwen3-vl-quantized-model-path> \
  --server.trust_remote_code true \
  --server.dtype float16 \
  --quantization.method compressed-tensors \
  --server.mem_fraction_static 0.8 \
  --server.max_running_requests 1 \
  --server.disable_cuda_graph \
  --batch-size 1 \
  --input-len 314 512 1024 2048 \
  --output-len 1 \
  --image <image-path> \
  --prompt "Describe this image." \
  --result-filename <result-jsonl-path>
```

In the JSONL, `vit_prefill_ms` wraps only `self.visual(...)`, while
`multimodal_prefill_*` are alias fields for the full VIT + LLM prefill — the two
have different scopes. In measurements on AGX Orin 32GB, W8A8 clearly leads FP16
/ W4A16 on long prefill.

### Profile

`bench_one_batch` has a built-in profile entry for inspecting kernel timelines
locally. There are two paths:

- **torch.profiler (supported)**: `--profile-activities CPU GPU` (default),
  emits a `.trace.json.gz` timeline you can open in Perfetto / chrome://tracing.
  The output directory is set by `PYMLLM_TORCH_PROFILER_DIR`, defaulting to
  `/tmp`.
- **Nsight Systems / nsys (experimental)**: `--profile-activities CUDA_PROFILER`
  drives nsys via `cudaProfilerStart/Stop`, and needs an outer wrapper like
  `nsys --capture-range=cudaProfilerApi`. This path is still being polished and
  may be rough in places; treat it as an optional deep-dive tool.
