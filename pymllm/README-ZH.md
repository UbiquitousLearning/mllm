# pymllm

![pymllm-arch](../assets/pymllm-arch.png)

## 已验证环境

本文档中的命令基于 Jetson Orin 上已验证通过的如下环境整理：

- JetPack / L4T：`R36.4.4`（来自 `/etc/nv_tegra_release`）
- Python：`3.10.12`
- pip：`26.0.1`
- PyTorch：`2.4.0`
- torchvision：`0.19.0a0+48b1edf`
- transformers：`5.3.0`
- safetensors：`0.7.0`
- flashinfer：`0.6.7`
- CUDA：`12.6`
- `torch.cuda.is_available()`：`True`

## 适用范围

本文档面向 Jetson Orin 上的 `pymllm` 使用，内容基于当前仓库内已验证流程整理。

当前只覆盖两条已验证路径：

- 原生模型：`Qwen3-VL-2B-Instruct`
- 量化模型：`Qwen3-VL-2B-Instruct-AWQ-4bit` + `compressed-tensors`

## 安装 editable 开发环境

在仓库根目录执行：

```bash
cd <repo-root>
SKBUILD_WHEEL_CMAKE=false python3 -m pip install -e .
python3 -m pip install -e <repo-root>/mllm-kernel --no-deps --no-build-isolation
```

安装完成后，可以用下面的命令做最小检查：

```bash
python3 - <<'PY'
import pymllm
import mllm_kernel

print("pymllm import ok")
print("mllm_kernel import ok")
PY
```

## 启动 pymllm server

### 启动量化模型服务

当前 Jetson Orin 上已验证的 `compressed-tensors` 启动命令如下：

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

说明：

- 若 `30000` 已被占用，可改成其他空闲端口，例如 `30001`。
- 当前这条量化路径按已验证配置使用 `float16`。

### 启动原生模型服务

如果要运行原生 `Qwen3-VL-2B-Instruct`，可使用：

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

## 调用示例

以下示例使用 OpenAI-compatible 接口，适合直接用 `curl` 或兼容 SGLang/OpenAI API 的客户端访问：

```text
/v1/chat/completions
```

### 文本推理示例

服务启动后，可以用下面的最小文本请求做 smoke test：

```bash
curl -s --noproxy '*' http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<served-model-name-or-path>",
    "messages": [{"role": "user", "content": "你好，只回复：ok"}],
    "max_tokens": 8,
    "temperature": 0.0,
    "stream": false
  }' ; echo
```

### 图片推理示例

先构造一个包含本地图片路径的请求：

```bash
python3 - <<'PY'
import json

payload = {
    "model": "<served-model-name-or-path>",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "请详细描述这张图片。"},
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

然后发送请求：

```bash
curl -s --noproxy '*' \
  http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  --data @/tmp/mm_req_path.json ; echo
```

## 当前已验证配置

当前文档对应的量化路径，已验证的是下面这组模型与配置：

- 模型类型：`Qwen3-VL-2B-Instruct-AWQ-4bit`
- quantization method：`compressed-tensors`
- load format：`safetensors`
- dtype：`float16`

如果后续扩展到其他模型、精度或量化变体，建议继续补充新的实测命令与说明。
