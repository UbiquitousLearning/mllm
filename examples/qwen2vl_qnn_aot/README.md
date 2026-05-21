# Qwen2-VL QNN AOT Example

This example demonstrates full QNN AOT inference for Qwen2-VL style models:

- visual encoder graphs
- LLM prefill graph
- LLM decode graph
- interactive image-text inference on Android

Model weights and compiled QNN context binaries are not stored in this repository.

The recommended Qwen2-VL-2B baseline tested on 2026-05-21 uses:

- LPBQ / W4A16 LLM weights
- FP16 visual encoder weights
- one combined QNN context with `model.0.s32`, `model.0.s1`, and five visual bucket graphs
- `--visual_bundle_layout single`
- `--visual_io_dtype fp16`

## Build

Build the x86 AOT compiler:

```bash
cmake --build build-qnn-aot --target mllm-qwen2vl-aot-c -j2
```

Build the Android runner:

```bash
cmake --build build-android-arm64-v8a-qnn --target mllm-qwen2vl-aot-runner -j2
```

## Compile QNN Context

Set `QNN_SDK_ROOT` or pass `--qnn_env_path` explicitly. The compiler uses the
QNN x86 backend to generate a context binary for Android.

```bash
./build-qnn-aot/bin/mllm-qwen2vl-aot-c \
  --model_path path/to/qwen2vl-2b-sm8650-qnn234-lpbq-visualfp16-vprojg16.mllm \
  --config examples/qwen2vl/config_2B_qnn_lpbq.json \
  --aot_config examples/qwen2vl/qnn_aot_cfg_2B_lpbq_vprojg16_unsignedpd.json \
  --output_context_name qwen2vl-2b-sm8650-qnn234-fullqnn-5bucket-visualfp16-v1.bin \
  --context_len 1024 \
  --prefill_len 32 \
  --include_visual_bundle \
  --visual_model path/to/qwen2vl-2b-sm8650-qnn234-lpbq-visualfp16-vprojg16.mllm \
  --visual_config examples/qwen2vl/config_2B_qnn_lpbq.json \
  --visual_aot_config examples/qwen2vl/qnn_aot_cfg_2B_visual_fp16.json \
  --visual_bundle_layout single \
  --visual_io_dtype fp16 \
  --visual_bucket_grids 12x16,16x24,24x24,24x32,18x52,52x18,26x36,36x26
```

The default input embedding quantization parameters are widened for multimodal
embeddings. Pass both `--input_embedding_scale -1` and
`--input_embedding_zero_point -1` to use the model embedding quantization
parameters instead.

## Android Interactive Runner

The helper script pushes the runner binary and launches an interactive shell
session. Set paths through environment variables as needed:

```bash
REMOTE_QNN_DIR=/data/local/tmp/mllm-qwen2vl-qnn \
REMOTE_CPU_DIR=/data/local/tmp/mllm-qwen2vl \
CONTEXT=models/qwen2vl-2b-sm8650-qnn234-fullqnn-5bucket-visualfp16-v1.bin \
QNN_PARAMS=models/qwen2vl-2b-sm8650-qnn234-lpbq-visualfp16-vprojg16.mllm \
VISUAL_IO_DTYPE=fp16 \
scripts/qwen2vl_qnn/run_qnn_interactive.sh
```

At the prompt, enter an image path on device and then the text prompt.

## Visual Buckets

The runner chooses the smallest visual bucket that covers the preprocessed image
grid. Padded visual tokens are masked in visual attention, and only valid visual
embeddings are passed to the LLM side.

The default bucket list is:

```text
12x16,16x24,24x24,24x32,18x52,52x18,26x36,36x26
```
