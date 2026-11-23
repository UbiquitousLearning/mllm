#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if (( $# < 1 || $# > 2 )); then
    echo "usage: $0 operator.so [matching_dsp_skel.so]" >&2
    exit 2
fi

operator_path="$1"
operator_dir="$(cd "$(dirname "${operator_path}")" && pwd)"
operator_name="$(basename "${operator_path}")"
operator_path="${operator_dir}/${operator_name}"
[[ -f "${operator_path}" ]] || {
    echo "operator not found: ${operator_path}" >&2
    exit 1
}

if (( $# == 2 )); then
    skel_path="$2"
else
    if [[ "${operator_name}" != libhmx_qk_i8_*.so ]]; then
        echo "cannot infer skel name from ${operator_name}" >&2
        exit 1
    fi
    variant_suffix="${operator_name#libhmx_qk_i8_}"
    skel_path="${operator_dir}/libhmx_int8_rpc_skel_${variant_suffix}"
fi
skel_dir="$(cd "$(dirname "${skel_path}")" && pwd)"
skel_name="$(basename "${skel_path}")"
skel_path="${skel_dir}/${skel_name}"
[[ -f "${skel_path}" ]] || {
    echo "DSP skel not found: ${skel_path}" >&2
    exit 1
}

loader_path="${HMX_OPERATOR_TEST_LOADER:-${operator_dir}/hmx_qk_i8_operator_dlopen_example}"
if [[ ! -f "${loader_path}" ]]; then
    loader_path="${project_root}/android_ReleaseG_aarch64/ship/hmx_qk_i8_operator_dlopen_example"
fi
[[ -f "${loader_path}" ]] || {
    echo "operator loader not found; build the bank or ARM targets first" >&2
    exit 1
}

device_dir="${HMX_OPERATOR_DEVICE_DIR:-/data/local/tmp/hmx_qk_operator_test}"
adb_args=()
if [[ -n "${ANDROID_SERIAL:-}" ]]; then
    adb_args=(-s "${ANDROID_SERIAL}")
fi

adb "${adb_args[@]}" shell mkdir -p "${device_dir}"
adb "${adb_args[@]}" push \
    "${operator_path}" \
    "${skel_path}" \
    "${loader_path}" \
    "${device_dir}/"
adb "${adb_args[@]}" shell chmod 755 \
    "${device_dir}/hmx_qk_i8_operator_dlopen_example"
adb "${adb_args[@]}" shell \
    "cd ${device_dir} && LD_LIBRARY_PATH=${device_dir} ADSP_LIBRARY_PATH='${device_dir};/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp' ./hmx_qk_i8_operator_dlopen_example ./${operator_name}"
