#!/usr/bin/env bash
set -euo pipefail

android_ndk="${ANDROID_NDK:?set ANDROID_NDK to an Android NDK root}"
hexagon_sdk="${HEXAGON_SDK_ROOT:?set HEXAGON_SDK_ROOT to a Hexagon SDK root}"
output="${1:-libfastrpc_session_redirect.so}"
toolchain_root="${android_ndk}/toolchains/llvm/prebuilt"
host_toolchain="$(find "${toolchain_root}" -mindepth 1 -maxdepth 1 -type d -print -quit)"
compiler="${host_toolchain}/bin/aarch64-linux-android28-clang"

[[ -x "${compiler}" ]] || { echo "Android compiler not found: ${compiler}" >&2; exit 1; }
[[ -f "${hexagon_sdk}/incs/remote.h" ]] || { echo "Hexagon SDK headers not found" >&2; exit 1; }

"${compiler}" -std=c11 -O2 -fPIC -shared -Wall -Wextra -Werror \
    -I"${hexagon_sdk}/incs" \
    -I"${hexagon_sdk}/incs/stddef" \
    "$(dirname "${BASH_SOURCE[0]}")/fastrpc_session_redirect.c" \
    -ldl -o "${output}"
