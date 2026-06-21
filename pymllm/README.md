# pymllm

![pymllm-arch](../assets/pymllm-arch.png)

`pymllm` is the Python inference and serving entry point for `mllm`. This
directory currently focuses on Qwen3 / Qwen3-VL serving on Jetson Orin,
OpenAI-compatible APIs, `compressed-tensors` quantized loading, and the W8A8
INT8 kernel path.

This README reflects the development state as of 2026-04-27 for the integration
branch:

```text
feature/jetson-qwen3-family-bf16-w4a16-w8a8
```

## Current status

Validated paths:

- `Qwen3-VL-2B-Instruct`: BF16 base-model serving.
- `Qwen3-VL-2B-Instruct-AWQ-4bit`: `compressed-tensors` W4A16 / AWQ Marlin
  serving.
- `Qwen3-VL-2B-Instruct-quantized.w8a8`: `compressed-tensors` W8A8
  `int-quantized` end-to-end serving.

Implemented and unit-tested models/components:

- `Qwen3VLForConditionalGeneration`: the main multimodal serving path.
- `Qwen3ForCausalLM`: text-only model skeleton, weight loading, and timing
  tests.
- `compressed-tensors`:
  - `pack-quantized` 4-bit weight path via GPTQ Marlin.
  - `int-quantized` W8A8 path via Triton activation quantization and CUTLASS
    `int8_scaled_mm`.

The current W8A8 forward path is:

```text
x(fp16/bf16)
  -> per_token_quant_int8        [Triton, dynamic per-token activation quant]
  -> int8_scaled_mm              [CUTLASS, INT8 Tensor Core, fused scales]
  -> output(fp16/bf16)
```

## Validated environment

The commands below were validated on Jetson Orin with:

- JetPack / L4T: `R36.4.4` (`/etc/nv_tegra_release`)
- Python: `3.10.12`
- PyTorch: `2.4.0`
- torchvision: `0.19.0a0+48b1edf`
- transformers: `5.3.0`
- safetensors: `0.7.0`
- flashinfer: `0.6.7`
- Triton Language: official PyPI `triton==3.6.0` manylinux aarch64 wheel
- CUDA: `12.6`
- GPU: Jetson Orin NX, SM87

Triton here means the GPU kernel DSL, not Triton Inference Server. The
Jetson-AI-Lab index also provides `3.4.0`, `3.5.1`, and `3.6.0`, but the tested
environment may require extra `TRITON_PTXAS_PATH` and `CPATH` settings with
those wheels. For this project, prefer the official PyPI `triton==3.6.0` wheel
and verify it with a minimal CUDA kernel or `per_token_quant_int8` smoke test.

The W8A8 CUTLASS JIT path requires CUTLASS headers. The lookup order is:

1. `CUTLASS_HOME/include`
2. `flashinfer` bundled `data/cutlass/include`
3. `/usr/local/include`, `/usr/include`, `/usr/local/cuda/include`

The first CUTLASS kernel call triggers JIT compilation and may take about
100 seconds. Later runs reuse:

```text
~/.cache/mllm_kernel/cutlass_int8_scaled_mm/
```

## Install the development environment

Run from the repository root:

```bash
cd <repo-root>
SKBUILD_WHEEL_CMAKE=false python3 -m pip install -e .
python3 -m pip install -e <repo-root>/mllm-kernel --no-deps --no-build-isolation
```

Run a minimal import check:

```bash
python3 - <<'PY'
import pymllm
import mllm_kernel

print("pymllm import ok")
print("mllm_kernel import ok")
PY
```

## Launch the server

### Quantized models (W4A16 / W8A8)

```bash
cd <repo-root>

python3 -m pymllm.server.launch \
  --server.model_path <quantized-model-path> \
  --server.tokenizer_path <quantized-model-path> \
  --server.load_format safetensors \
  --server.dtype float16 \
  --quantization.method compressed-tensors \
  --server.host 0.0.0.0 \
  --server.port 30000 \
  --server.attention_backend auto \
  --server.gdn_decode_backend pytorch \
  --server.mem_fraction_static 0.05 \
  --server.max_running_requests 1 \
  --server.max_total_tokens 256 \
  --server.max_prefill_tokens 128 \
  --server.chunked_prefill_size 128 \
  --server.disable_radix_cache \
  --server.disable_cuda_graph \
  --server.log_level debug
```

Notes:

- `--quantization.method compressed-tensors` reads the model `config.json` and
  selects the W4A16 or W8A8 signature automatically.
- W8A8 requires SM80 or newer GPUs.
- `--server.disable_radix_cache` uses `ChunkCache`; the KV slot leak in this
  mode has been fixed.
- If port `30000` is already in use, switch to another free port.

### BF16 base models

```bash
cd <repo-root>

python3 -m pymllm.server.launch \
  --server.model_path <model-path> \
  --server.tokenizer_path <model-path> \
  --server.load_format safetensors \
  --server.dtype float16 \
  --server.host 0.0.0.0 \
  --server.port 30000 \
  --server.attention_backend auto \
  --server.gdn_decode_backend pytorch \
  --server.mem_fraction_static 0.05 \
  --server.max_running_requests 1 \
  --server.max_total_tokens 256 \
  --server.max_prefill_tokens 128 \
  --server.chunked_prefill_size 128 \
  --server.disable_radix_cache \
  --server.disable_cuda_graph \
  --server.log_level debug
```

## Request examples

### Health check

```bash
curl -s --noproxy '*' http://127.0.0.1:30000/v1/models ; echo
```

Expected response contains:

```text
"owned_by":"pymllm"
```

### Text request

```bash
curl -s --noproxy '*' http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "None",
    "messages": [{"role": "user", "content": "Reply with: ok"}],
    "max_tokens": 8,
    "temperature": 0.0,
    "stream": false
  }' ; echo
```

### Image request

Use a container-visible absolute image path. Do not use the `file://...`
prefix.

```bash
python3 - <<'PY'
import json

payload = {
    "model": "None",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Please describe this image in detail."},
                {"type": "image_url", "image_url": {"url": "/workspace/xcd_mllm/test.png"}},
            ],
        }
    ],
    "max_tokens": 128,
    "temperature": 0.0,
    "stream": False,
}

with open("/tmp/mm_req_path.json", "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False)

print("saved /tmp/mm_req_path.json")
PY

curl -s --noproxy '*' http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  --data @/tmp/mm_req_path.json ; echo
```

## Development and tests

Common unit tests:

```bash
pytest pymllm/tests/test_compressed_tensors_config.py -q
pytest pymllm/tests/test_compressed_tensors_runtime.py -q
pytest pymllm/tests/test_qwen3_model_registry.py -q
pytest pymllm/tests/test_qwen3_weight_loading.py -q
pytest pymllm/tests/test_qwen3_forward_timing.py -q
pytest mllm-kernel/tests/test_int8_scaled_mm_cutlass.py -q
```

Common microbenchmarks:

```bash
python3 pymllm/tests/bench_w8a8_activation_quant.py
python3 mllm-kernel/benchmarks/bench_int8_scaled_mm.py
python3 mllm-kernel/benchmarks/bench_w4a16_vs_w8a8.py
```

To measure first-use CUTLASS compilation again, clear the JIT cache:

```bash
rm -rf ~/.cache/mllm_kernel/cutlass_int8_scaled_mm/
```

## Known limitations

- The W8A8 CUTLASS path is JIT-compiled, so first startup includes about
  100 seconds of compilation overhead.
- W8A8 activation quantization uses a Triton kernel; its fixed decode-time
  cost remains a future optimization target.
- Qwen3-VL ViT, `lm_head`, embeddings, and LayerNorm are outside the current
  W8A8 quantized scope.
- Other GPUs need separate validation for tile dispatch, JIT compilation, and
  performance.
- OpenAI-compatible responses hide debug timing by default for SGLang/OpenAI
  compatibility. Use `--server.enable_debug_timing` only for local diagnostics;
  strict model-level timing should use dedicated benchmarks.
