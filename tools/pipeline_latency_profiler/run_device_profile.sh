#!/system/bin/sh
set -eu

if [ "$#" -lt 3 ] || [ "$#" -gt 8 ]; then
    echo "usage: $0 PROFILER_BINARY MANIFEST HEAD_PROFILE [RAW_OUTPUT] [MAX_KEY_LEN] [WARMUP] [REPETITIONS] [QUERY_LEN]" >&2
    exit 2
fi

profiler_binary=$1
manifest=$2
head_profile=$3
raw_output=${4:-/data/local/tmp/hmx-pipeline-profile-raw.tsv}
max_key_len=${5:-4160}
warmup=${6:-3}
repetitions=${7:-20}
query_len=${8:-512}

: "${MLLM_HMX_PIPELINE_PROFILE_STAGES:=all}"
: "${MLLM_HMX_PIPELINE_MAIN_CPU:=5}"
: "${MLLM_HMX_PIPELINE_TOPK_CPU:=0,1}"
: "${MLLM_HMX_PIPELINE_TOPK_WORKERS:=2}"
: "${MLLM_HMX_PIPELINE_SPARSE_CPU:=5}"
: "${MLLM_HMX_PIPELINE_SPARSE_WORKERS:=1}"
: "${MLLM_HMX_PIPELINE_TOPK_MODE:=cooperative}"

export MLLM_HMX_PIPELINE_PROFILE_STAGES
export MLLM_HMX_PIPELINE_MAIN_CPU
export MLLM_HMX_PIPELINE_TOPK_CPU
export MLLM_HMX_PIPELINE_TOPK_WORKERS
export MLLM_HMX_PIPELINE_SPARSE_CPU
export MLLM_HMX_PIPELINE_SPARSE_WORKERS
export MLLM_HMX_PIPELINE_TOPK_MODE

exec "$profiler_binary" "$manifest" "$head_profile" "$raw_output" \
    "$max_key_len" "$warmup" "$repetitions" "$query_len"
