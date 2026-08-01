#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HTP_ARCH="${HTP_ARCH:-v79}"
HTP_VERSION="${HTP_ARCH^^}"
DEST="${MLLM_DEVICE_ROOT:-/data/local/tmp/mllm}/qnn-lib"
OP_BUILD_DIR="${OP_BUILD_DIR:-$ROOT_DIR/mllm/backends/qnn/LLaMAOpPackageHtp/LLaMAPackage/build}"

: "${QNN_SDK_ROOT:?Set QNN_SDK_ROOT to the extracted QAIRT/QNN SDK}"

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

push_required() {
    local source="$1"
    local target="${2:-$DEST/}"
    if [[ ! -f "$source" ]]; then
        echo "Required QNN file is missing: $source" >&2
        exit 1
    fi
    adb_push "$source" "$target"
}

push_optional() {
    local source="$1"
    if [[ -f "$source" ]]; then
        adb_push "$source" "$DEST/"
    fi
}

"$ADB" shell mkdir -p "$DEST"

ANDROID_LIB="$QNN_SDK_ROOT/lib/aarch64-android"
push_required "$ANDROID_LIB/libQnnHtp.so"
push_required "$ANDROID_LIB/libQnnHtp${HTP_VERSION}Stub.so"
push_required "$QNN_SDK_ROOT/lib/hexagon-$HTP_ARCH/unsigned/libQnnHtp${HTP_VERSION}Skel.so"
push_required "$ANDROID_LIB/libQnnSystem.so"
push_optional "$ANDROID_LIB/libQnnHtpPrepare.so"
push_optional "$ANDROID_LIB/libQnnHtpProfilingReader.so"
push_optional "$ANDROID_LIB/libQnnHtpOptraceProfilingReader.so"
push_optional "$ANDROID_LIB/libQnnHtp${HTP_VERSION}CalculatorStub.so"

push_required "$OP_BUILD_DIR/aarch64-android/libQnnLLaMAPackage.so" "$DEST/libQnnLLaMAPackage_CPU.so"
push_required "$OP_BUILD_DIR/hexagon-$HTP_ARCH/libQnnLLaMAPackage.so" "$DEST/libQnnLLaMAPackage_HTP.so"

echo "Pushed QNN runtime and custom op package for HTP $HTP_VERSION to $DEST"
