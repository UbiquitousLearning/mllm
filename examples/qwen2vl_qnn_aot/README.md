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
- one combined QNN context with `model.0.s32`, `model.0.s1`, and five visual size buckets with aspect-ratio variants
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

## Run Qwen2-VL VLM on Android

Large runtime artifacts are not included in this repository. Download the
prebuilt Qwen2-VL-2B QNN artifacts from ModelScope, or rebuild them with the
compile command below:

```text
https://www.modelscope.cn/models/twlddd/Qwen2-VL-2B-Instruct-Full-QNN-AOT-for-mllm/summary
```

The expected local artifact layout is flat:

```text
QWEN2VL_ARTIFACT_DIR/
|-- qwen2vl-2b-sm8650-qnn234-lpbq-visualfp16-vprojg16.mllm
|-- qwen2vl-2b-sm8650-qnn234-fullqnn-5bucket-visualfp16-v1.bin
|-- config_2B_qnn_lpbq.json
`-- tokenizer.json
```

Prepare the device directories and push the runner plus model artifacts:

```bash
REMOTE_QNN_DIR=/data/local/tmp/mllm-qwen2vl-qnn
REMOTE_CPU_DIR=/data/local/tmp/mllm-qwen2vl
ARTIFACT_DIR=/path/to/QWEN2VL_ARTIFACT_DIR

adb shell "mkdir -p \
  ${REMOTE_QNN_DIR}/bin \
  ${REMOTE_QNN_DIR}/lib \
  ${REMOTE_QNN_DIR}/models \
  ${REMOTE_QNN_DIR}/config \
  ${REMOTE_CPU_DIR}/tokenizer \
  ${REMOTE_CPU_DIR}/images/eval"

adb push build-android-arm64-v8a-qnn/bin/mllm-qwen2vl-aot-runner \
  ${REMOTE_QNN_DIR}/bin/
adb shell "chmod 755 ${REMOTE_QNN_DIR}/bin/mllm-qwen2vl-aot-runner"

adb push "${ARTIFACT_DIR}/qwen2vl-2b-sm8650-qnn234-fullqnn-5bucket-visualfp16-v1.bin" \
  ${REMOTE_QNN_DIR}/models/
adb push "${ARTIFACT_DIR}/qwen2vl-2b-sm8650-qnn234-lpbq-visualfp16-vprojg16.mllm" \
  ${REMOTE_QNN_DIR}/models/
adb push "${ARTIFACT_DIR}/config_2B_qnn_lpbq.json" \
  ${REMOTE_QNN_DIR}/config/
adb push "${ARTIFACT_DIR}/tokenizer.json" \
  ${REMOTE_CPU_DIR}/tokenizer/
```

The device also needs the mllm runtime libraries, the mllm QNN backend library,
the Qualcomm QNN runtime libraries, and the QNN op-package libraries under
`${REMOTE_QNN_DIR}/lib`. Those libraries are built or obtained from the local
mllm/QNN SDK environment and are not shipped in this example.

Push one test image:

```bash
adb push /path/to/test.jpg ${REMOTE_CPU_DIR}/images/eval/test.jpg
```

Run the interactive VLM session from the repository root:

```bash
REMOTE_QNN_DIR=/data/local/tmp/mllm-qwen2vl-qnn \
REMOTE_CPU_DIR=/data/local/tmp/mllm-qwen2vl \
CONTEXT=models/qwen2vl-2b-sm8650-qnn234-fullqnn-5bucket-visualfp16-v1.bin \
QNN_PARAMS=models/qwen2vl-2b-sm8650-qnn234-lpbq-visualfp16-vprojg16.mllm \
QNN_CONFIG=config/config_2B_qnn_lpbq.json \
TOKENIZER=/data/local/tmp/mllm-qwen2vl/tokenizer/tokenizer.json \
VISUAL_IO_DTYPE=fp16 \
scripts/qwen2vl_qnn/run_qnn_interactive.sh
```

Then enter the on-device image path and the text prompt:

```text
Image path> /data/local/tmp/mllm-qwen2vl/images/eval/test.jpg
Prompt> describe this picture
```

For fixed-set evaluation, put images under `ASSET_SRC` and use a TSV case file
with `case_id|image_file|prompt` rows:

```bash
ASSET_SRC=/path/to/eval/images \
CASES_FILE=scripts/qwen2vl_qnn/qwen2vl_eval_cases_5bucket.tsv \
REMOTE_QNN_DIR=/data/local/tmp/mllm-qwen2vl-qnn \
REMOTE_CPU_DIR=/data/local/tmp/mllm-qwen2vl \
VISUAL_IO_DTYPE=fp16 \
scripts/qwen2vl_qnn/run_qnn_eval_fixed_set.sh
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
