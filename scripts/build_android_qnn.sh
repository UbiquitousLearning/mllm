#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build-arm-qnn}"
ANDROID_NDK="${ANDROID_NDK:-${ANDROID_NDK_ROOT:-}}"

: "${ANDROID_NDK:?Set ANDROID_NDK or ANDROID_NDK_ROOT to Android NDK r26+}"
: "${QNN_SDK_ROOT:?Set QNN_SDK_ROOT to the extracted QAIRT/QNN SDK}"

mkdir -p "$BUILD_DIR"

cmake -S "$ROOT_DIR" -B "$BUILD_DIR" \
-DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
-DCMAKE_BUILD_TYPE=Release \
-DANDROID_ABI="arm64-v8a" \
-DANDROID_STL=c++_static \
-DANDROID_PLATFORM=android-28 \
-DCMAKE_CXX_FLAGS="-march=armv8.2-a+dotprod" \
-DNATIVE_LIBRARY_OUTPUT=. -DNATIVE_INCLUDE_OUTPUT=. "$@" \
-DQNN_SDK_ROOT="$QNN_SDK_ROOT" \
-DQNN=ON \
-DARM=ON \
-DOPENCL=OFF \
-DDEBUG=OFF \
-DTEST=OFF \
-DQUANT=OFF \
-DQNN_VALIDATE_NODE=ON \
-DMLLM_BUILD_XNNPACK_BACKEND=OFF

cmake --build "$BUILD_DIR" --parallel "$(nproc)"
