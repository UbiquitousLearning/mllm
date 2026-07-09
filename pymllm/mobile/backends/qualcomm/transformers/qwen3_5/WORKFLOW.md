# Qwen3.5 QNN Deployment Workflow

## Overview

Qwen3.5 has a hybrid architecture: 18 GDN (recurrent) + 6 full attention layers.
Only the 6 full attention layers are compiled to QNN. GDN layers stay on CPU.

```
[Step 1] Python: Quantize (train.py)              → model.safetensors
[Step 2] Python: Convert (mllm-convertor)          → qwen3_5-0.8B.mllm
[Step 3] C++ x86: AOT Compile (mllm-qwen3_5-aot-c) → qwen3_5-0.8B-hybrid.bin
[Step 4] C++ cross: Build runner for device
[Step 5] Device: Run hybrid CPU+QNN inference
```

## Step 1 — Quantize (Python, GPU host)

```bash
python pymllm/mobile/backends/qualcomm/transformers/qwen3_5/train.py \
  --model_path /ssd/mllm/models/Qwen3.5-0.8B \
  --max_length 2048 \
  --num_samples 128 \
  --output_dir /ssd/mllm/models/Qwen3.5-0.8B/quantized
# Output: quantized/model.safetensors
```

This calibrates activation quantization on wikitext, then converts all weights
(QLinearLPBQ → Conv2D HWIO, QRMSNorm, QEmbedding) and saves the full state dict
including QDQ scale/zp parameters.

## Step 2 — Convert safetensors → `.mllm` (Python, host)

```bash
python -m pymllm.mobile.utils.mllm_convertor \
  --input_path /ssd/mllm/models/Qwen3.5-0.8B/quantized/model.safetensors \
  --output_path /ssd/mllm/models/Qwen3.5-0.8B/quantized/qwen3_5-0.8B.mllm \
  --model_name qwen3_5_0.8b \
  --verbose
```

## Step 3 — AOT Compile (x86 host, requires QNN SDK)

```bash
# Set QNN SDK path
source /opt/qcom/aistack/qairt/2.41.0.251128/bin/envsetup.sh

# Build x86 AOT compiler
python task.py tasks/build_x86_qnn_aot.yaml

# Compile 6 attention layers → QNN context binary
./build-qnn-aot/bin/mllm-qwen3_5-aot-c \
  -m /ssd/mllm/models/Qwen3.5-0.8B/quantized/qwen3_5-0.8B.mllm \
  -c examples/qwen3_5_qnn_aot/qnn_aot_cfg_0.8B.json \
  -aot_cfg examples/qwen3_5_qnn_aot/qnn_aot_cfg_0.8B.json \
  --context_len 1024 \
  --prefill_len 32

# Output: qwen3_5-0.8B-hybrid.bin (12 QNN graphs: 6 layers × 2 seq lengths)
```

## Step 4 — Build device runner (Android cross-compile)

```bash
export ANDROID_NDK_PATH=/path/to/android-ndk
python task.py tasks/build_android_qnn.yaml

# Produces: build-android-arm64-v8a-qnn/bin/mllm-qwen3_5-aot-runner
```

## Step 5 — Push to device and run

```bash
adb push build-android-arm64-v8a-qnn/bin/mllm-qwen3_5-aot-runner /data/local/tmp/
adb push qwen3_5-0.8B-hybrid.bin /data/local/tmp/
adb push /ssd/mllm/models/Qwen3.5-0.8B/config_mllm.json /data/local/tmp/
adb push $QNN_SDK_ROOT/lib/aarch64-android/libQnnHtp*.so /data/local/tmp/
adb push $QNN_SDK_ROOT/lib/aarch64-android/libQnnSystem.so /data/local/tmp/
adb push build-android-arm64-v8a-qnn/lib/libQnnLLaMAPackage.so /data/local/tmp/

adb shell "cd /data/local/tmp && LD_LIBRARY_PATH=. \
  ./mllm-qwen3_5-aot-runner \
    -m qwen3_5-0.8B-hybrid.bin \
    -t /path/to/tokenizer \
    -c config_mllm.json \
    --qnn_context qwen3_5-0.8B-hybrid.bin"
```

## Architecture Notes

- 24 total layers: indices 0-23
- Full attention at indices: 3, 7, 11, 15, 19, 23 (6 layers)
- GDN (linear attention) at all other indices (18 layers)
- Runtime alternates: CPU(GDN) → QNN(Attn) → CPU(GDN) → QNN(Attn) → ...
- Per-layer QNN graphs (not monolithic) due to interleaved GDN execution
