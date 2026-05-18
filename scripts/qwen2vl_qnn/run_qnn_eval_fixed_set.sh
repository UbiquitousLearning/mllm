#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

ADB_BIN="$(pick_adb)"
CASES_FILE="${CASES_FILE:-${SCRIPT_DIR}/qwen2vl_eval_cases_5bucket.tsv}"
REMOTE_QNN_DIR="${REMOTE_QNN_DIR:-/data/local/tmp/mllm-qwen2vl-qnn}"
REMOTE_CPU_DIR="${REMOTE_CPU_DIR:-/data/local/tmp/mllm-qwen2vl}"
REMOTE_IMAGE_DIR="${REMOTE_IMAGE_DIR:-${REMOTE_CPU_DIR}/images/eval}"

CONTEXT="${CONTEXT:-models/qwen2vl-2b-sm8650-qnn234-fullqnn-5bucket-baseline-v1.bin}"
QNN_PARAMS="${QNN_PARAMS:-models/qwen2vl-2b-sm8650-qnn234-lpbq-baseline-v1.mllm}"
TOKENIZER="${TOKENIZER:-${REMOTE_CPU_DIR}/tokenizer/tokenizer.json}"
QNN_CONFIG="${QNN_CONFIG:-config/config_2B_qnn_lpbq.json}"
VISUAL_MODEL="${VISUAL_MODEL:-${REMOTE_CPU_DIR}/models/qwen2vl-2b-w4a32-kai.mllm}"
VISUAL_CONFIG="${VISUAL_CONFIG:-${REMOTE_CPU_DIR}/config/config_2B_w32a32.json}"
VISUAL_MODEL_VERSION="${VISUAL_MODEL_VERSION:-v2}"
VISUAL_QNN="${VISUAL_QNN:-1}"
VISUAL_BUNDLE_LAYOUT="${VISUAL_BUNDLE_LAYOUT:-single}"
VISUAL_BUCKET_GRIDS="${VISUAL_BUCKET_GRIDS:-12x16,16x24,24x24,24x32,18x52,52x18,26x36,36x26}"

AR_LEN="${AR_LEN:-32}"
CONTEXT_LEN="${CONTEXT_LEN:-1024}"
GEN_LEN="${GEN_LEN:-120}"
INPUT_EMBEDDING_SCALE="${INPUT_EMBEDDING_SCALE:-0.002563515}"
INPUT_EMBEDDING_ZERO_POINT="${INPUT_EMBEDDING_ZERO_POINT:-15604}"
KEY_CACHE_DTYPE="${KEY_CACHE_DTYPE:-uint8}"
DUMP_STATS="${DUMP_STATS:-1}"
PUSH_ASSETS="${PUSH_ASSETS:-1}"
OUT_ROOT="${OUT_ROOT:-${QWEN2VL_QNN_REPO_ROOT}/logs/qwen2vl_qnn_eval}"
RUN_ID="${RUN_ID:-$(timestamp)}"
OUT_DIR="${OUT_DIR:-${OUT_ROOT}/${RUN_ID}}"

if [[ "${PUSH_ASSETS}" != "0" ]]; then
  CASES_FILE="${CASES_FILE}" REMOTE_IMAGE_DIR="${REMOTE_IMAGE_DIR}" "${SCRIPT_DIR}/push_eval_assets.sh"
fi

mkdir -p "${OUT_DIR}"

cat >"${OUT_DIR}/run_config.txt" <<EOF
adb=${ADB_BIN}
cases_file=${CASES_FILE}
remote_qnn_dir=${REMOTE_QNN_DIR}
remote_cpu_dir=${REMOTE_CPU_DIR}
remote_image_dir=${REMOTE_IMAGE_DIR}
context=${CONTEXT}
qnn_params=${QNN_PARAMS}
visual_model=${VISUAL_MODEL}
visual_config=${VISUAL_CONFIG}
visual_qnn=${VISUAL_QNN}
visual_bundle_layout=${VISUAL_BUNDLE_LAYOUT}
visual_bucket_grids=${VISUAL_BUCKET_GRIDS}
ar_len=${AR_LEN}
context_len=${CONTEXT_LEN}
gen_len=${GEN_LEN}
input_embedding_scale=${INPUT_EMBEDDING_SCALE}
input_embedding_zero_point=${INPUT_EMBEDDING_ZERO_POINT}
key_cache_dtype=${KEY_CACHE_DTYPE}
EOF

status_file="${OUT_DIR}/case_status.tsv"
printf "case_id\timage_file\tstatus\tlog\n" >"${status_file}"

while IFS='|' read -r case_id image_file prompt; do
  remote_image="${REMOTE_IMAGE_DIR}/${image_file}"
  log_file="${OUT_DIR}/${case_id}.log"
  visual_args=""
  if [[ "${VISUAL_QNN}" == "1" ]]; then
    visual_args="--visual_qnn --visual_bundle_layout $(remote_quote "${VISUAL_BUNDLE_LAYOUT}")"
    if [[ -n "${VISUAL_BUCKET_GRIDS}" ]]; then
      visual_args="${visual_args} --visual_bucket_grids $(remote_quote "${VISUAL_BUCKET_GRIDS}")"
    fi
  else
    visual_args="--visual_model $(remote_quote "${VISUAL_MODEL}") --visual_model_version $(remote_quote "${VISUAL_MODEL_VERSION}")"
  fi

  echo
  echo "==> ${case_id}"
  echo "image : ${remote_image}"
  echo "prompt: ${prompt}"
  echo "log   : ${log_file}"

  remote_cmd="
cd $(remote_quote "${REMOTE_QNN_DIR}") &&
export MLLM_QWEN2VL_AOT_DUMP_STATS=$(remote_quote "${DUMP_STATS}") &&
export LD_LIBRARY_PATH=\$PWD/lib:${REMOTE_CPU_DIR}/lib:/vendor/lib64:/system/lib64:\$LD_LIBRARY_PATH &&
export ADSP_LIBRARY_PATH=\"\$PWD/lib;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp\" &&
./bin/mllm-qwen2vl-aot-runner \
  -m $(remote_quote "${CONTEXT}") \
  --qnn_params $(remote_quote "${QNN_PARAMS}") \
  ${visual_args} \
  -t $(remote_quote "${TOKENIZER}") \
  -c $(remote_quote "${QNN_CONFIG}") \
  --visual_config $(remote_quote "${VISUAL_CONFIG}") \
  -i $(remote_quote "${remote_image}") \
  -p $(remote_quote "${prompt}") \
  --ar_len ${AR_LEN} \
  --context_len ${CONTEXT_LEN} \
  --gen_len ${GEN_LEN} \
  --input_embedding_scale ${INPUT_EMBEDDING_SCALE} \
  --input_embedding_zero_point ${INPUT_EMBEDDING_ZERO_POINT} \
  --key_cache_dtype $(remote_quote "${KEY_CACHE_DTYPE}")
"

  if "${ADB_BIN}" shell "${remote_cmd}" </dev/null 2>&1 | tee "${log_file}"; then
    printf "%s\t%s\tOK\t%s\n" "${case_id}" "${image_file}" "${log_file}" >>"${status_file}"
  else
    printf "%s\t%s\tFAIL\t%s\n" "${case_id}" "${image_file}" "${log_file}" >>"${status_file}"
  fi
done < <(read_eval_cases "${CASES_FILE}")

summary_file="${OUT_DIR}/summary.tsv"
python3 "${QWEN2VL_QNN_REPO_ROOT}/tools/qnn_debug/summarize_qwen2vl_eval_logs.py" \
  --log_dir "${OUT_DIR}" \
  --cases "${CASES_FILE}" \
  --out "${summary_file}" || true

echo
echo "Logs    : ${OUT_DIR}"
echo "Status  : ${status_file}"
echo "Summary : ${summary_file}"
