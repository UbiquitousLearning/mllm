#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST_MODELS_DIR="${HOST_MODELS_DIR:-$ROOT_DIR/../models}"
DEVICE_ROOT="${MLLM_DEVICE_ROOT:-/data/local/tmp/mllm}"
QNN_MODEL="${QNN_MODEL:-$HOST_MODELS_DIR/Qwen2.5-1.5B-Instruct_rotated-noshadow.mllm}"
DECODING_MODEL="${DECODING_MODEL:-$HOST_MODELS_DIR/Qwen2.5-1.5B-Instruct_rotated-Q40.mllm}"
VOCAB_FILE="${VOCAB_FILE:-$ROOT_DIR/vocab/qwen2.5_vocab.mllm}"
MERGES_FILE="${MERGES_FILE:-$ROOT_DIR/vocab/qwen2.5_merges.txt}"
DEMO_BINARY="${DEMO_BINARY:-$ROOT_DIR/bin-arm-qnn/demo_qwen_npu}"

if [[ -n "${ADB:-}" ]]; then
    if [[ ! -x "$ADB" ]] && ! command -v "$ADB" >/dev/null 2>&1; then
        echo "Configured ADB is not executable: $ADB" >&2
        exit 1
    fi
elif command -v adb >/dev/null 2>&1; then
    ADB="${ADB:-adb}"
elif command -v adb.exe >/dev/null 2>&1; then
    ADB="${ADB:-adb.exe}"
else
    echo "adb/adb.exe was not found in PATH" >&2
    exit 1
fi

adb_push() {
    local source="$1"
    local target="$2"
    if [[ "$ADB" == *.exe ]]; then
        source="$(wslpath -w "$source")"
    fi
    "$ADB" push "$source" "$target"
}

for file in "$QNN_MODEL" "$DECODING_MODEL" "$VOCAB_FILE" "$MERGES_FILE" "$DEMO_BINARY"; do
    if [[ ! -f "$file" ]]; then
        echo "Required deployment file is missing: $file" >&2
        exit 1
    fi
done

"$ROOT_DIR/scripts/push_qnn_lib.sh"
"$ADB" shell mkdir -p "$DEVICE_ROOT/bin" "$DEVICE_ROOT/models" "$DEVICE_ROOT/vocab"
adb_push "$VOCAB_FILE" "$DEVICE_ROOT/vocab/qwen2.5_vocab.mllm"
adb_push "$MERGES_FILE" "$DEVICE_ROOT/vocab/qwen2.5_merges.txt"
adb_push "$QNN_MODEL" "$DEVICE_ROOT/models/Qwen2.5-1.5B-Instruct_rotated-noshadow.mllm"
adb_push "$DECODING_MODEL" "$DEVICE_ROOT/models/Qwen2.5-1.5B-Instruct_rotated-Q40.mllm"
adb_push "$DEMO_BINARY" "$DEVICE_ROOT/bin/demo_qwen_npu"
"$ADB" shell chmod 755 "$DEVICE_ROOT/bin/demo_qwen_npu"

"$ADB" shell "cd '$DEVICE_ROOT/bin' && LD_LIBRARY_PATH='$DEVICE_ROOT/qnn-lib' ADSP_LIBRARY_PATH='$DEVICE_ROOT/qnn-lib' ./demo_qwen_npu --vocab '$DEVICE_ROOT/vocab/qwen2.5_vocab.mllm' --merge '$DEVICE_ROOT/vocab/qwen2.5_merges.txt' --qnn-model '$DEVICE_ROOT/models/Qwen2.5-1.5B-Instruct_rotated-noshadow.mllm' --decoding-model '$DEVICE_ROOT/models/Qwen2.5-1.5B-Instruct_rotated-Q40.mllm' --billion 1.5B-rotated"
