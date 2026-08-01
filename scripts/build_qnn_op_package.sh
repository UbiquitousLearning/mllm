#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PACKAGE_DIR="$ROOT_DIR/mllm/backends/qnn/LLaMAOpPackageHtp/LLaMAPackage"
MAKEFILE="$PACKAGE_DIR/../Makefile"
HTP_ARCH="${HTP_ARCH:-v79}"

: "${QNN_SDK_ROOT:?Set QNN_SDK_ROOT to the extracted QAIRT/QNN SDK}"
: "${HEXAGON_SDK_ROOT:?Set HEXAGON_SDK_ROOT to Hexagon SDK 6.x}"
ANDROID_NDK_ROOT="${ANDROID_NDK_ROOT:-${ANDROID_NDK:-}}"
: "${ANDROID_NDK_ROOT:?Set ANDROID_NDK_ROOT or ANDROID_NDK to Android NDK r26+}"
export ANDROID_NDK_ROOT

case "$HTP_ARCH" in
    v68|v69|v73|v75|v79) ;;
    *) echo "Unsupported HTP_ARCH: $HTP_ARCH" >&2; exit 1 ;;
esac

make -C "$PACKAGE_DIR" -f "$MAKEFILE" htp_aarch64 -j"$(nproc)"
make -C "$PACKAGE_DIR" -f "$MAKEFILE" "htp_$HTP_ARCH" -j"$(nproc)"

echo "Built custom QNN op package for aarch64 Android and HTP ${HTP_ARCH^^}"
