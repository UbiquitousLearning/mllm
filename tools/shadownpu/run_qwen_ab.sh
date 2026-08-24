#!/usr/bin/env bash
set -euo pipefail

# Reproduce the matched dense/sparse Qwen2.5-1.5B benchmark on an already
# provisioned Android device. This script never reboots the device, changes
# DSP sessions, or deletes remote files.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
result_dir="${RESULT_DIR:-${repo_root}/results/shadownpu-qwen}"
device_root="${DEVICE_ROOT:-/data/local/tmp/mllm}"
run_dir="${RUN_DIR:-${device_root}/qwen-run}"
bank="${HMX_BANK:-${device_root}/hmx-qwen2-m512-h12-n4160}"
profile="${HEAD_PROFILE:-${device_root}/qwen2-1.5B-retain-0.2.txt}"
benchmark="${BENCHMARK:-${device_root}/benchmark_qwen_npu}"
redirect="${SESSION_REDIRECT:-${device_root}/libfastrpc_session_redirect.so}"
qnn_lib_dir="${QNN_LIB_DIR:-${device_root}/qnn-lib}"
vocab="${VOCAB:-${device_root}/vocab/qwen2.5_vocab.mllm}"
merge="${MERGE:-${device_root}/vocab/qwen2.5_merges.txt}"
qnn_model="${QNN_MODEL:-${device_root}/models/Qwen2.5-1.5B-Instruct_rotated-noshadow.mllm}"
decoding_model="${DECODING_MODEL:-${device_root}/models/Qwen2.5-1.5B-Instruct_rotated-Q40.mllm}"
prompts="${PROMPTS:-1024 2048 4096}"
repetition_count="${REPETITIONS:-3}"
adb_command="${ADB:-adb}"

if [[ ! "${repetition_count}" =~ ^[1-9][0-9]*$ ]]; then
    echo "REPETITIONS must be a positive integer" >&2
    exit 2
fi
read -r -a prompt_values <<<"${prompts}"
if (( ${#prompt_values[@]} == 0 )); then
    echo "PROMPTS must contain at least one positive integer" >&2
    exit 2
fi
for prompt in "${prompt_values[@]}"; do
    if [[ ! "${prompt}" =~ ^[1-9][0-9]*$ ]]; then
        echo "PROMPTS must contain positive integers" >&2
        exit 2
    fi
done

adb_prefix=("${adb_command}")
if [[ -n "${ADB_SERIAL:-}" ]]; then
    adb_prefix+=(-s "${ADB_SERIAL}")
fi

mkdir -p "${result_dir}"
"${adb_prefix[@]}" get-state >/dev/null
"${adb_prefix[@]}" shell "test -x '${benchmark}' && test -d '${qnn_lib_dir}' && test -f '${run_dir}/qnn_context.bin' && test -f '${bank}/manifest.tsv' && test -f '${profile}' && test -f '${redirect}' && test -f '${vocab}' && test -f '${merge}' && test -f '${qnn_model}' && test -f '${decoding_model}'"

manifest_row="$("${adb_prefix[@]}" shell "awk -F '\t' '\$1 == \"qwen2_1p5b\" && \$2 == 12 && \$5 == 512 && \$6 == 128 && \$7 == 4160 { operators[++count] = \$3; skels[count] = \$4 } END { if (count) { selected = int((count + 1) / 2); print operators[selected] \"\\t\" skels[selected] } }' '${bank}/manifest.tsv'" | tr -d '\r')"
if [[ -z "${manifest_row}" || "${manifest_row}" == *$'\n'* || "${manifest_row}" != *$'\t'* ]]; then
    echo "failed to select one qwen2_1p5b H12/M512/K128/N4160 operator/dispatcher pair from ${bank}/manifest.tsv" >&2
    exit 2
fi
manifest_operator="${manifest_row%%$'\t'*}"
dispatcher="${manifest_row#*$'\t'}"
operator="${HMX_OPERATOR:-${manifest_operator}}"
if [[ -z "${operator}" || "${operator}" == *$'\n'* ]]; then
    echo "failed to select one H12 operator from ${bank}/manifest.tsv" >&2
    exit 2
fi
if [[ "${operator}" == /* ]]; then
    operator_path="${operator}"
else
    operator_path="${bank}/${operator}"
fi
if [[ "${dispatcher}" == /* ]]; then
    dispatcher_path="${dispatcher}"
else
    dispatcher_path="${bank}/${dispatcher}"
fi
"${adb_prefix[@]}" shell "test -f '${operator_path}' && test -f '${dispatcher_path}'"

logs=()

run_one() {
    local prompt="$1"
    local mode="$2"
    local repetition="$3"
    local log="${result_dir}/${mode}-p${prompt}-run${repetition}.log"
    local attention_mode="dense"
    local attention_worker=0
    local main_cpu=7
    local profile_argument=""
    local profile_used="none"
    logs+=("${log}")
    if [[ "${mode}" == "sparse" ]]; then
        attention_mode="hmx-topk"
        attention_worker=1
        main_cpu=2
        profile_argument="--head-retain-profile '${profile}'"
        profile_used="${profile}"
    fi

    printf 'SHADOWNPU_RUN_CONFIG mode=%s prompt=%s operator=%s dispatcher=%s profile=%s\n' \
        "${mode}" "${prompt}" "${operator_path}" "${dispatcher_path}" \
        "${profile_used}" >"${log}"

    "${adb_prefix[@]}" shell "cd '${run_dir}' && env \
MLLM_SKIP_QNN_CONTEXT_SAVE=1 \
MLLM_QNN_SEQUENCE_TILE=256 \
LD_PRELOAD='${redirect}' \
LD_LIBRARY_PATH='${qnn_lib_dir}:${bank}' \
ADSP_LIBRARY_PATH='${qnn_lib_dir};${bank};/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp' \
MLLM_ATTENTION_PROFILE=1 \
MLLM_HMX_INT8_OPERATOR_LIBRARY='${operator_path}' \
MLLM_HMX_INT8_OPERATOR_MANIFEST='${bank}/manifest.tsv' \
MLLM_HMX_INT8_OPERATOR_MODEL=qwen2_1p5b \
MLLM_HMX_INT8_BUCKET_DIAGNOSTICS=0 \
MLLM_HMX_INT8_RECALL_DIAGNOSTICS=0 \
MLLM_HMX_INT8_CPU_SCALE_PROFILE=1 \
MLLM_HMX_INT8_CPU_PACK=1 \
MLLM_HMX_INT8_CPU_PACK_INTRA_HEAD=1 \
MLLM_HMX_CPU_PACK_CPU=5 \
MLLM_HMX_CPU_PACK_WORKERS=1 \
MLLM_HMX_CPU_PACK_OVERLAP=1 \
MLLM_HMX_CPU_PACK_SPIN_US=20000 \
MLLM_HMX_INT8_INCREMENTAL_K_SCALE=1 \
MLLM_HMX_INT8_LAYER_K_CACHE=1 \
MLLM_HMX_INT8_ASYNC_K_CACHE_STORE=1 \
MLLM_HMX_INT8_ATTENTION_EXECUTION_SCOPE=0 \
MLLM_HMX_INT8_FUSE_PER_HEAD_BUCKETS=1 \
MLLM_HMX_INT8_PER_HEAD_BUCKET_GROUP_HEADS=12 \
MLLM_HMX_INT8_DYNAMIC_QK_SCALE=0 \
MLLM_HMX_INT8_DYNAMIC_OUTPUT_SCALE=0 \
MLLM_HMX_INT8_DSP_INT32_TOPK=0 \
MLLM_HMX_INT8_TOPK_OVERSAMPLE=1 \
MLLM_HMX_INT8_OUTLIER_FALLBACK=0 \
MLLM_HMX_PIPELINE_GROUP_HEADS=12 \
MLLM_HMX_PIPELINE_READY_GROUP_HEADS=12 \
MLLM_HMX_PIPELINE_SCHEDULE=fifo \
MLLM_HMX_PIPELINE_TOPK_MODE=cooperative \
MLLM_HMX_PIPELINE_TOPK_CPU=4,6 \
MLLM_HMX_PIPELINE_TOPK_WORKERS=2 \
MLLM_HMX_PIPELINE_SPARSE_CPU=7 \
MLLM_HMX_PIPELINE_SPARSE_WORKERS=1 \
MLLM_HMX_PIPELINE_LONG_RPC=1 \
MLLM_HMX_PIPELINE_RPC_CPU=1 \
MLLM_HMX_PIPELINE_EXECUTOR=direct-three-stage \
MLLM_HMX_PIPELINE_DIRECT_SPARSE_ASSIST_PERCENT=20 \
MLLM_HMX_PIPELINE_DIRECT_SPARSE_WORKER_ASSIST=0 \
MLLM_CPU_ATTENTION_WORKER='${attention_worker}' \
MLLM_CPU_ATTENTION_WORKER_CPU=7 \
MLLM_HMX_TOPK_THREADS=1 \
MLLM_SPARSE_ATTENTION_THREADS=1 \
OMP_NUM_THREADS=4 \
KMP_AFFINITY='norespect,granularity=fine,proclist=[7,4,5,6],explicit' \
'${benchmark}' \
--vocab '${vocab}' \
--merge '${merge}' \
--qnn-model '${qnn_model}' \
--decoding-model '${decoding_model}' \
--model-size 1.5B-rotated --qnn-profile off --limits 4160 --thread 4 \
--prompt-tokens '${prompt}' --decode-tokens 8 --chunk-size 512 \
--main-cpu '${main_cpu}' --chunk-overlap off --attention-mode '${attention_mode}' \
${profile_argument} --quality-prompt records --needle NPU-7391" \
        2>&1 | tee -a "${log}"
}

for prompt in "${prompt_values[@]}"; do
    for ((repetition = 1; repetition <= repetition_count; ++repetition)); do
        if (( repetition % 2 == 1 )); then
            run_one "${prompt}" sparse "${repetition}"
            run_one "${prompt}" dense "${repetition}"
        else
            run_one "${prompt}" dense "${repetition}"
            run_one "${prompt}" sparse "${repetition}"
        fi
    done
done

python3 "${repo_root}/tools/shadownpu/summarize_results.py" \
    "${logs[@]}" --output "${result_dir}/summary.tsv"
