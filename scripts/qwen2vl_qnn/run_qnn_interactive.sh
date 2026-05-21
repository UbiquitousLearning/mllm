#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

ADB_BIN="$(pick_adb)"
REMOTE_QNN_DIR="${REMOTE_QNN_DIR:-/data/local/tmp/mllm-qwen2vl-qnn}"
REMOTE_CPU_DIR="${REMOTE_CPU_DIR:-/data/local/tmp/mllm-qwen2vl}"
REMOTE_IMAGE_DIR="${REMOTE_IMAGE_DIR:-${REMOTE_CPU_DIR}/images/eval}"

LOCAL_RUNNER="${LOCAL_RUNNER:-${QWEN2VL_QNN_REPO_ROOT}/build-android-arm64-v8a-qnn/bin/mllm-qwen2vl-aot-runner}"
PUSH_RUNNER="${PUSH_RUNNER:-1}"

CONTEXT="${CONTEXT:-models/qwen2vl-2b-sm8650-qnn234-fullqnn-5bucket-visualfp16-v1.bin}"
QNN_PARAMS="${QNN_PARAMS:-models/qwen2vl-2b-sm8650-qnn234-lpbq-visualfp16-vprojg16.mllm}"
TOKENIZER="${TOKENIZER:-${REMOTE_CPU_DIR}/tokenizer/tokenizer.json}"
QNN_CONFIG="${QNN_CONFIG:-config/config_2B_qnn_lpbq.json}"
VISUAL_MODEL="${VISUAL_MODEL:-${REMOTE_CPU_DIR}/models/qwen2vl-2b-w4a32-kai.mllm}"
VISUAL_CONFIG="${VISUAL_CONFIG:-${REMOTE_CPU_DIR}/config/config_2B_w32a32.json}"
VISUAL_MODEL_VERSION="${VISUAL_MODEL_VERSION:-v2}"
VISUAL_QNN="${VISUAL_QNN:-1}"
VISUAL_BUNDLE_LAYOUT="${VISUAL_BUNDLE_LAYOUT:-single}"
VISUAL_IO_DTYPE="${VISUAL_IO_DTYPE:-fp16}"
VISUAL_BUCKET_GRIDS="${VISUAL_BUCKET_GRIDS:-12x16,16x24,24x24,24x32,18x52,52x18,26x36,36x26}"

AR_LEN="${AR_LEN:-32}"
CONTEXT_LEN="${CONTEXT_LEN:-1024}"
GEN_LEN="${GEN_LEN:-1000}"
PROMPT="${PROMPT:-describe this picture}"
INPUT_EMBEDDING_SCALE="${INPUT_EMBEDDING_SCALE:-0.002563515}"
INPUT_EMBEDDING_ZERO_POINT="${INPUT_EMBEDDING_ZERO_POINT:-15604}"
VISUAL_OUTPUT_SCALE="${VISUAL_OUTPUT_SCALE:--1}"
VISUAL_OUTPUT_ZERO_POINT="${VISUAL_OUTPUT_ZERO_POINT:--1}"
KEY_CACHE_DTYPE="${KEY_CACHE_DTYPE:-uint8}"
DUMP_STATS="${DUMP_STATS:-0}"
ADB_SHELL_TTY="${ADB_SHELL_TTY:-1}"

if [[ "${PUSH_RUNNER}" != "0" ]]; then
  if [[ ! -f "${LOCAL_RUNNER}" ]]; then
    echo "Runner not found: ${LOCAL_RUNNER}" >&2
    echo "Build it first, or set PUSH_RUNNER=0 if the phone already has the correct binary." >&2
    exit 1
  fi
  "${ADB_BIN}" shell "mkdir -p $(remote_quote "${REMOTE_QNN_DIR}/bin")" </dev/null
  "${ADB_BIN}" push "${LOCAL_RUNNER}" "${REMOTE_QNN_DIR}/bin/mllm-qwen2vl-aot-runner" </dev/null
  "${ADB_BIN}" shell "chmod 755 $(remote_quote "${REMOTE_QNN_DIR}/bin/mllm-qwen2vl-aot-runner")" </dev/null
fi

visual_args=""
if [[ "${VISUAL_QNN}" == "1" ]]; then
  visual_args="--visual_qnn --visual_bundle_layout $(remote_quote "${VISUAL_BUNDLE_LAYOUT}") --visual_io_dtype $(remote_quote "${VISUAL_IO_DTYPE}")"
  if [[ -n "${VISUAL_BUCKET_GRIDS}" ]]; then
    visual_args="${visual_args} --visual_bucket_grids $(remote_quote "${VISUAL_BUCKET_GRIDS}")"
  fi
  if [[ "${VISUAL_BUNDLE_LAYOUT}" == "hybrid_single" ]]; then
    visual_args="${visual_args} --visual_model $(remote_quote "${VISUAL_MODEL}") --visual_model_version $(remote_quote "${VISUAL_MODEL_VERSION}")"
  fi
else
  visual_args="--visual_model $(remote_quote "${VISUAL_MODEL}") --visual_model_version $(remote_quote "${VISUAL_MODEL_VERSION}")"
fi

echo "Starting Qwen2-VL QNN interactive runner on device."
echo "Remote image examples are under: ${REMOTE_IMAGE_DIR}"
echo "Type /exit at Image path> to quit."

remote_cmd="
cd $(remote_quote "${REMOTE_QNN_DIR}") &&
export MLLM_QWEN2VL_AOT_DUMP_STATS=$(remote_quote "${DUMP_STATS}") &&
export LD_LIBRARY_PATH=\$PWD/lib:${REMOTE_CPU_DIR}/lib:/vendor/lib64:/system/lib64:\$LD_LIBRARY_PATH &&
export ADSP_LIBRARY_PATH=\"\$PWD/lib;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp\" &&
exec ./bin/mllm-qwen2vl-aot-runner \
  -m $(remote_quote "${CONTEXT}") \
  --qnn_params $(remote_quote "${QNN_PARAMS}") \
  ${visual_args} \
  -t $(remote_quote "${TOKENIZER}") \
  -c $(remote_quote "${QNN_CONFIG}") \
  --visual_config $(remote_quote "${VISUAL_CONFIG}") \
  -p $(remote_quote "${PROMPT}") \
  --interactive \
  --ar_len ${AR_LEN} \
  --context_len ${CONTEXT_LEN} \
  --gen_len ${GEN_LEN} \
  --input_embedding_scale ${INPUT_EMBEDDING_SCALE} \
  --input_embedding_zero_point ${INPUT_EMBEDDING_ZERO_POINT} \
  --visual_output_scale ${VISUAL_OUTPUT_SCALE} \
  --visual_output_zero_point ${VISUAL_OUTPUT_ZERO_POINT} \
  --key_cache_dtype $(remote_quote "${KEY_CACHE_DTYPE}")
"

if [[ "${ADB_SHELL_TTY}" == "1" ]]; then
  "${ADB_BIN}" shell -t "${remote_cmd}"
else
  "${ADB_BIN}" shell "${remote_cmd}"
fi
