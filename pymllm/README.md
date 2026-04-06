# pymllm

![pymllm-arch](../assets/pymllm-arch.png)

## Validated environment

The commands in this document were validated on Jetson Orin with the following
environment baseline:

- JetPack / L4T: `R36.4.4` (`/etc/nv_tegra_release`)
- Python: `3.10.12`
- pip: `26.0.1`
- PyTorch: `2.4.0`
- torchvision: `0.19.0a0+48b1edf`
- transformers: `5.3.0`
- safetensors: `0.7.0`
- flashinfer: `0.6.7`
- CUDA: `12.6`
- `torch.cuda.is_available()`: `True`

## Scope

This document covers `pymllm` usage on Jetson Orin based on the workflows
validated in this repository.

The current validated paths are:

- Base model: `Qwen3-VL-2B-Instruct`
- Quantized model: `Qwen3-VL-2B-Instruct-AWQ-4bit` with `compressed-tensors`

## Install the editable development environment

Run the following from the repository root:

```bash
cd <repo-root>
SKBUILD_WHEEL_CMAKE=false python3 -m pip install -e .
python3 -m pip install -e <repo-root>/mllm-kernel --no-deps --no-build-isolation
```

After installation, run a minimal import check:

```bash
python3 - <<'PY'
import pymllm
import mllm_kernel

print("pymllm import ok")
print("mllm_kernel import ok")
PY
```

## Launch the pymllm server

### Launch the quantized model

The following `compressed-tensors` command has been validated on Jetson Orin:

```bash
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
  --server.log_level debug \
  2>&1 | tee /tmp/pymllm_qwen3_vl_awq_ct.log
```

Notes:

- If port `30000` is already in use, switch to another free port such as
  `30001`.
- This validated quantized path uses `float16`.

### Launch the base model

To run the base `Qwen3-VL-2B-Instruct` model:

```bash
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
  --server.log_level debug \
  2>&1 | tee /tmp/pymllm_server.log
```

## Request examples

The examples below use the OpenAI-compatible API and work with `curl` or any
SGLang/OpenAI-compatible client:

```text
/v1/chat/completions
```

### Text inference

Use the following minimal text request as a smoke test:

```bash
curl -s --noproxy '*' http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<served-model-name-or-path>",
    "messages": [{"role": "user", "content": "Reply with: ok"}],
    "max_tokens": 8,
    "temperature": 0.0,
    "stream": false
  }' ; echo
```

### Image inference

First, prepare a request payload that references a local image path:

```bash
python3 - <<'PY'
import json

payload = {
    "model": "<served-model-name-or-path>",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Please describe this image in detail."},
                {
                    "type": "image_url",
                    "image_url": {"url": "<image-path>"},
                },
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
```

Then send the request:

```bash
curl -s --noproxy '*' \
  http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  --data @/tmp/mm_req_path.json ; echo
```

## Validated configuration

The validated quantized setup described in this document uses:

- Model family: `Qwen3-VL-2B-Instruct-AWQ-4bit`
- Quantization method: `compressed-tensors`
- Load format: `safetensors`
- Dtype: `float16`

If this repository later adds validated instructions for other models,
precisions, or quantization variants, extend this README with the new commands
and notes.
