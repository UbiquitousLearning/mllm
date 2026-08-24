#!/usr/bin/env bash
set -euo pipefail

# Build the fixed-shape/fixed-scale QK estimator catalog used for NPU/HMX
# estimates. Defaults build the nine operators used by the reproducible
# Qwen2.5-1.5B path: one H12/M512/K128/N4160 shape and a 3x3 Q/K scale grid.
#
# Positional arguments:
#   1. DSP architecture (default: v75)
#   2. output directory (default: operator_bank/<DSP architecture>)
#
# Common overrides (lists may use whitespace, commas, or semicolons):
#   HMX_OPERATOR_MODELS="qwen2_1p5b"
#   HMX_FIXED_HEADS="1 2 4 8"
#   HMX_OPERATOR_Q_MULTIPLIERS="0.5 1.0 2.0"
#   HMX_OPERATOR_K_MULTIPLIERS="0.5 1.0 2.0"
#   HMX_OPERATOR_M_VALUES="128"
#   HMX_OPERATOR_N_VALUES="2048"
#
# HMX_OPERATOR_Q_SCALES, HMX_OPERATOR_K_SCALES, and
# HMX_OPERATOR_OUTPUT_SCALES replace model-relative values with absolute scale
# lists. HMX_OPERATOR_TARGET_REQUANT_SCALE instead derives one output scale for
# each Q/K pair as q_scale * k_scale / target_requant_scale. The two output
# scale modes are mutually exclusive. HMX_OPERATOR_SHAPES="MxKxN ..."
# replaces the catalog-derived shapes.
# HMX_BUILD_JOBS controls parallelism inside each individual build.
#
# A custom HMX_OPERATOR_CATALOG_FILE may replace the built-in models. It is a
# whitespace-, comma-, or tab-separated text file with five fields per row:
#
#   model  head_dim  q_base_scale  k_base_scale  output_scale
#
# Blank lines, a header row, and comments beginning with '#' are accepted. If
# HMX_OPERATOR_MODELS is also set, it filters rows from the custom catalog.

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sdk_root="${HEXAGON_SDK_ROOT:?set HEXAGON_SDK_ROOT to a Hexagon SDK root}"
dsp_arch="${1:-${HMX_DSP_ARCH:-v75}}"
output_dir="${2:-${HMX_OPERATOR_BANK_DIR:-${project_root}/operator_bank/${dsp_arch}}}"
build_jobs="${HMX_BUILD_JOBS:-1}"
catalog_file="${HMX_OPERATOR_CATALOG_FILE:-}"
shared_dispatch_skel="${HMX_SHARED_DISPATCH_SKEL:-1}"
hvx_lanes="${HMX_HVX_LANES:-4}"
hvx_epilogue_lanes="${HMX_HVX_EPILOGUE_LANES:-1}"
hvx_epilogue_batch_tiles="${HMX_HVX_EPILOGUE_BATCH_TILES:-8}"
long_rpc_ready_group_heads="${HMX_LONG_RPC_READY_GROUP_HEADS:-12}"

default_models="qwen2_1p5b"
model_spec="${HMX_OPERATOR_MODELS:-${default_models}}"
heads_spec="${HMX_OPERATOR_HEADS:-${HMX_FIXED_HEADS:-12}}"
m_spec="${HMX_OPERATOR_M_VALUES:-${HMX_FIXED_M:-512}}"
n_spec="${HMX_OPERATOR_N_VALUES:-${HMX_FIXED_N:-4160}}"
shape_spec="${HMX_OPERATOR_SHAPES:-}"
fixed_k_override="${HMX_FIXED_K:-}"
q_multiplier_spec="${HMX_OPERATOR_Q_MULTIPLIERS:-${HMX_OPERATOR_SCALE_MULTIPLIERS:-0.5 1.0 2.0}}"
k_multiplier_spec="${HMX_OPERATOR_K_MULTIPLIERS:-${HMX_OPERATOR_SCALE_MULTIPLIERS:-0.5 1.0 2.0}}"
absolute_q_spec="${HMX_OPERATOR_Q_SCALES:-}"
absolute_k_spec="${HMX_OPERATOR_K_SCALES:-}"
absolute_output_spec="${HMX_OPERATOR_OUTPUT_SCALES:-${HMX_OPERATOR_OUTPUT_SCALE:-}}"
target_requant_spec="${HMX_OPERATOR_TARGET_REQUANT_SCALE:-}"

die() {
    echo "error: $*" >&2
    exit 1
}

normalize_list() {
    local value="$1"
    value="${value//,/ }"
    value="${value//;/ }"
    value="${value//$'\n'/ }"
    value="${value//$'\t'/ }"
    printf '%s\n' "${value}"
}

is_positive_integer() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

# Python's Decimal keeps the official artifact's decimal constants stable in
# both manifest entries and filenames. The SDK build itself already requires a
# host Python installation.
decimal_product() {
    local lhs="$1"
    local rhs="$2"
    local result
    [[ "${lhs}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]] ||
        die "invalid scale '${lhs}'; expected a positive finite decimal"
    [[ "${rhs}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]] ||
        die "invalid scale or multiplier '${rhs}'; expected a positive finite decimal"
    result="$(python3 -c '
from decimal import Decimal
import sys
value = Decimal(sys.argv[1]) * Decimal(sys.argv[2])
if not value.is_finite() or value <= 0:
    raise SystemExit(1)
text = format(value, "f")
if "." in text:
    text = text.rstrip("0").rstrip(".")
if "." not in text:
    text += ".0"
print(text)
' "${lhs}" "${rhs}")" || die "scale product '${lhs} * ${rhs}' must be finite and positive"
    [[ -n "${result}" ]] || die "scale product '${lhs} * ${rhs}' is empty"
    printf '%s\n' "${result}"
}

canonical_scale() {
    decimal_product "$1" "1"
}

scale_tag() {
    local tag="$1"
    tag="${tag//E/e}"
    tag="${tag//./p}"
    tag="${tag//+/p}"
    tag="${tag//-/m}"
    printf '%s\n' "${tag}"
}

requant_scale() {
    local q_scale="$1"
    local k_scale="$2"
    local output_scale="$3"
    python3 -c '
from decimal import Decimal, getcontext
import sys
getcontext().prec = 17
value = Decimal(sys.argv[1]) * Decimal(sys.argv[2]) / Decimal(sys.argv[3])
print(value)
' "${q_scale}" "${k_scale}" "${output_scale}"
}

output_scale_for_requant() {
    local q_scale="$1"
    local k_scale="$2"
    local target_requant_scale="$3"
    python3 -c '
from decimal import Decimal, getcontext
import sys
getcontext().prec = 25
value = Decimal(sys.argv[1]) * Decimal(sys.argv[2]) / Decimal(sys.argv[3])
if not value.is_finite() or value <= 0:
    raise SystemExit(1)
text = format(value, "f").rstrip("0").rstrip(".")
if "." not in text:
    text += ".0"
print(text)
' "${q_scale}" "${k_scale}" "${target_requant_scale}" ||
        die "derived output scale must be finite and positive"
}

copy_artifact() {
    local source_file="$1"
    local output_file="$2"
    [[ -f "${source_file}" ]] || die "expected build artifact not found: ${source_file}"
    install -m 0755 "${source_file}" "${output_file}"
}

catalog_models=()
catalog_dims=()
catalog_q_bases=()
catalog_k_bases=()
catalog_output_scales=()

add_catalog_entry() {
    local model="$1"
    local head_dim="$2"
    local q_base="$3"
    local k_base="$4"
    local output_scale="$5"
    [[ "${model}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] ||
        die "catalog model '${model}' is not filename-safe"
    is_positive_integer "${head_dim}" ||
        die "catalog head_dim for '${model}' must be a positive integer"
    catalog_models+=("${model}")
    catalog_dims+=("${head_dim}")
    catalog_q_bases+=("$(canonical_scale "${q_base}")")
    catalog_k_bases+=("$(canonical_scale "${k_base}")")
    catalog_output_scales+=("$(canonical_scale "${output_scale}")")
}

add_builtin_model() {
    local requested="$1"
    local key="${requested,,}"
    key="${key//-/_}"
    key="${key//./p}"
    case "${key}" in
        qwen2_1p5b)
            add_catalog_entry qwen2_1p5b 128 \
                0.086935067916610501 0.071037953121309075 1.9938009425636014
            ;;
        *)
            die "unknown built-in model '${requested}'"
            ;;
    esac
}

read -r -a requested_models <<< "$(normalize_list "${model_spec}")"
(( ${#requested_models[@]} > 0 )) || die "HMX_OPERATOR_MODELS is empty"

if [[ -n "${catalog_file}" ]]; then
    [[ -f "${catalog_file}" ]] || die "custom catalog not found: ${catalog_file}"
    filter_custom_catalog=0
    if [[ -v HMX_OPERATOR_MODELS ]]; then
        filter_custom_catalog=1
    fi
    catalog_line=0
    while IFS= read -r line || [[ -n "${line}" ]]; do
        ((catalog_line += 1))
        line="${line%%#*}"
        line="${line//,/ }"
        read -r -a fields <<< "$(normalize_list "${line}")"
        (( ${#fields[@]} > 0 )) || continue
        if [[ "${fields[0],,}" == "model" ]]; then
            continue
        fi
        (( ${#fields[@]} == 5 )) ||
            die "${catalog_file}:${catalog_line}: expected five fields"

        include_model=1
        if (( filter_custom_catalog )); then
            include_model=0
            for requested in "${requested_models[@]}"; do
                if [[ "${fields[0],,}" == "${requested,,}" ]]; then
                    include_model=1
                    break
                fi
            done
        fi
        if (( include_model )); then
            add_catalog_entry \
                "${fields[0]}" "${fields[1]}" "${fields[2]}" "${fields[3]}" "${fields[4]}"
        fi
    done < "${catalog_file}"
else
    for requested in "${requested_models[@]}"; do
        add_builtin_model "${requested}"
    done
fi
(( ${#catalog_models[@]} > 0 )) || die "the selected model catalog is empty"

read -r -a heads <<< "$(normalize_list "${heads_spec}")"
read -r -a m_values <<< "$(normalize_list "${m_spec}")"
read -r -a n_values <<< "$(normalize_list "${n_spec}")"
read -r -a q_multipliers <<< "$(normalize_list "${q_multiplier_spec}")"
read -r -a k_multipliers <<< "$(normalize_list "${k_multiplier_spec}")"

(( ${#heads[@]} > 0 )) || die "HMX_FIXED_HEADS is empty"
(( ${#m_values[@]} > 0 )) || die "HMX_OPERATOR_M_VALUES is empty"
(( ${#n_values[@]} > 0 )) || die "HMX_OPERATOR_N_VALUES is empty"
for value in "${heads[@]}"; do
    is_positive_integer "${value}" || die "head count '${value}' must be a positive integer"
done
for value in "${m_values[@]}" "${n_values[@]}"; do
    is_positive_integer "${value}" || die "shape dimension '${value}' must be a positive integer"
done
if [[ -n "${fixed_k_override}" ]]; then
    is_positive_integer "${fixed_k_override}" || die "HMX_FIXED_K must be a positive integer"
fi

explicit_shapes=()
if [[ -n "${shape_spec}" ]]; then
    read -r -a explicit_shapes <<< "$(normalize_list "${shape_spec}")"
    for shape in "${explicit_shapes[@]}"; do
        [[ "${shape}" =~ ^([1-9][0-9]*)[xX]([1-9][0-9]*)[xX]([1-9][0-9]*)$ ]] ||
            die "invalid shape '${shape}'; expected MxKxN with positive integers"
    done
fi

absolute_q_scales=()
if [[ -n "${absolute_q_spec}" ]]; then
    read -r -a raw_scales <<< "$(normalize_list "${absolute_q_spec}")"
    for scale in "${raw_scales[@]}"; do
        absolute_q_scales+=("$(canonical_scale "${scale}")")
    done
else
    normalized_multipliers=()
    for multiplier in "${q_multipliers[@]}"; do
        normalized_multipliers+=("$(canonical_scale "${multiplier}")")
    done
    q_multipliers=("${normalized_multipliers[@]}")
fi

absolute_k_scales=()
if [[ -n "${absolute_k_spec}" ]]; then
    read -r -a raw_scales <<< "$(normalize_list "${absolute_k_spec}")"
    for scale in "${raw_scales[@]}"; do
        absolute_k_scales+=("$(canonical_scale "${scale}")")
    done
else
    normalized_multipliers=()
    for multiplier in "${k_multipliers[@]}"; do
        normalized_multipliers+=("$(canonical_scale "${multiplier}")")
    done
    k_multipliers=("${normalized_multipliers[@]}")
fi

absolute_output_scales=()
if [[ -n "${absolute_output_spec}" && -n "${target_requant_spec}" ]]; then
    die "HMX_OPERATOR_OUTPUT_SCALES and HMX_OPERATOR_TARGET_REQUANT_SCALE are mutually exclusive"
fi
if [[ -n "${absolute_output_spec}" ]]; then
    read -r -a raw_scales <<< "$(normalize_list "${absolute_output_spec}")"
    for scale in "${raw_scales[@]}"; do
        absolute_output_scales+=("$(canonical_scale "${scale}")")
    done
fi
target_requant_scale=""
if [[ -n "${target_requant_spec}" ]]; then
    target_requant_scale="$(canonical_scale "${target_requant_spec}")"
fi

q_count=${#q_multipliers[@]}
if (( ${#absolute_q_scales[@]} > 0 )); then
    q_count=${#absolute_q_scales[@]}
fi
k_count=${#k_multipliers[@]}
if (( ${#absolute_k_scales[@]} > 0 )); then
    k_count=${#absolute_k_scales[@]}
fi
output_count=1
if (( ${#absolute_output_scales[@]} > 0 )); then
    output_count=${#absolute_output_scales[@]}
fi
shape_count=$((${#m_values[@]} * ${#n_values[@]}))
if (( ${#explicit_shapes[@]} > 0 )); then
    shape_count=${#explicit_shapes[@]}
fi
variant_count=$((${#catalog_models[@]} * ${#heads[@]} * shape_count *
    q_count * k_count * output_count))

[[ -f "${sdk_root}/setup_sdk_env.source" ]] ||
    die "Hexagon SDK not found: ${sdk_root}"
command -v python3 >/dev/null 2>&1 || die "python3 is required for exact decimal scales"
[[ "${dsp_arch}" =~ ^v[0-9]+$ ]] || die "invalid DSP architecture: ${dsp_arch}"
[[ "${build_jobs}" =~ ^[1-9][0-9]*$ ]] || die "HMX_BUILD_JOBS must be a positive integer"
[[ "${shared_dispatch_skel}" =~ ^[01]$ ]] ||
    die "HMX_SHARED_DISPATCH_SKEL must be 0 or 1"

if [[ "${output_dir}" != /* ]]; then
    output_dir="${project_root}/${output_dir}"
fi
mkdir -p "${output_dir}"

echo "Building ${variant_count} ARM operators and their DSP dispatcher skels"
echo "Models: ${catalog_models[*]}"
echo "Heads: ${heads[*]}"
echo "DSP architecture: ${dsp_arch}"
echo "Output: ${output_dir}"

# Keep this bank's iterative CMake configuration separate from build.sh.
# These names intentionally match the module's android_* / hexagon_* ignores.
host_build_dir="android_operator_bank_aarch64"
dsp_build_dir="hexagon_operator_bank_${dsp_arch}"

# setup_sdk_env.source is not nounset-clean and expects HEXAGON_SDK_ROOT to be
# unset before it establishes the selected SDK environment.
# shellcheck disable=SC1090
unset HEXAGON_SDK_ROOT
set +u
source "${sdk_root}/setup_sdk_env.source"
set -u
export LD_LIBRARY_PATH="${DEFAULT_HEXAGON_TOOLS_ROOT}/Tools/bin:${LD_LIBRARY_PATH:-}"

host_ship="${project_root}/${host_build_dir}/ship"
dsp_ship="${project_root}/${dsp_build_dir}/ship"
manifest="${output_dir}/manifest.tsv"
manifest_tmp="${manifest}.tmp.$$"
trap 'unlink "${manifest_tmp}" 2>/dev/null || true' EXIT
printf 'model\theads\toperator_so\tdsp_skel\tm\tk\tn\tq_scale\tk_scale\toutput_scale\trequant_scale\n' \
    > "${manifest_tmp}"

variant_index=0
declare -A built_dispatch_skels=()
max_dispatch_heads=0
for heads_value in "${heads[@]}"; do
    if (( heads_value > max_dispatch_heads )); then
        max_dispatch_heads="${heads_value}"
    fi
done

build_shape_variants() {
    local model="$1"
    local heads_value="$2"
    local fixed_m="$3"
    local fixed_k="$4"
    local fixed_n="$5"
    local q_scale k_scale output_scale q_tag k_tag output_tag
    local variant_suffix operator_output_name operator_file
    local skel_output_name skel_file rq_scale
    local -a pair_output_scales

    for q_scale in "${current_q_scales[@]}"; do
        q_tag="$(scale_tag "${q_scale}")"
        for k_scale in "${current_k_scales[@]}"; do
            k_tag="$(scale_tag "${k_scale}")"
            pair_output_scales=("${current_output_scales[@]}")
            if [[ -n "${target_requant_scale}" ]]; then
                pair_output_scales=("$(output_scale_for_requant \
                    "${q_scale}" "${k_scale}" "${target_requant_scale}")")
            fi
            for output_scale in "${pair_output_scales[@]}"; do
                output_tag="$(scale_tag "${output_scale}")"
                ((variant_index += 1))

                variant_suffix="${model}_h${heads_value}_m${fixed_m}_k${fixed_k}_n${fixed_n}_sq${q_tag}_sk${k_tag}_sy${output_tag}"
                operator_output_name="hmx_qk_i8_${variant_suffix}"
                operator_file="lib${operator_output_name}.so"
                if [[ "${shared_dispatch_skel}" == "1" ]]; then
                    skel_output_name="hmx_int8_rpc_skel_${model}_m${fixed_m}_k${fixed_k}_n${fixed_n}_dispatcher"
                else
                    skel_output_name="hmx_int8_rpc_skel_${variant_suffix}"
                fi
                skel_file="lib${skel_output_name}.so"
                rq_scale="$(requant_scale "${q_scale}" "${k_scale}" "${output_scale}")"

                if [[ -z "${built_dispatch_skels[${skel_output_name}]:-}" ]]; then
                    echo "[DSP dispatcher] ${skel_file}"
                    build_cmake hexagon "-j${build_jobs}" \
                        BUILD_OUTPUT_DIR="${dsp_build_dir}" DSP_ARCH="${dsp_arch}" \
                        HMX_FIXED_HEADS="${max_dispatch_heads}" \
                        HMX_FIXED_M="${fixed_m}" HMX_FIXED_K="${fixed_k}" HMX_FIXED_N="${fixed_n}" \
                        HMX_OPERATOR_OUTPUT_NAME= \
                        HMX_OPERATOR_Q_SCALE= HMX_OPERATOR_K_SCALE= \
                        HMX_OPERATOR_OUTPUT_SCALE= \
                        HMX_HVX_LANES="${hvx_lanes}" \
                        HMX_HVX_EPILOGUE_LANES="${hvx_epilogue_lanes}" \
                        HMX_HVX_EPILOGUE_BATCH_TILES="${hvx_epilogue_batch_tiles}" \
                        HMX_LONG_RPC_READY_GROUP_HEADS="${long_rpc_ready_group_heads}" \
                        HMX_RPC_SKEL_OUTPUT_NAME="${skel_output_name}"
                    copy_artifact "${dsp_ship}/${skel_file}" "${output_dir}/${skel_file}"
                    built_dispatch_skels[${skel_output_name}]=1
                fi

                echo "[ARM ${variant_index}/${variant_count}] ${operator_file}"
                build_cmake android "-j${build_jobs}" \
                    BUILD_OUTPUT_DIR="${host_build_dir}" \
                    HMX_FIXED_HEADS="${heads_value}" \
                    HMX_FIXED_M="${fixed_m}" HMX_FIXED_K="${fixed_k}" HMX_FIXED_N="${fixed_n}" \
                    HMX_OPERATOR_OUTPUT_NAME="${operator_output_name}" \
                    HMX_OPERATOR_Q_SCALE="${q_scale}" \
                    HMX_OPERATOR_K_SCALE="${k_scale}" \
                    HMX_OPERATOR_OUTPUT_SCALE="${output_scale}" \
                    HMX_HVX_LANES="${hvx_lanes}" \
                    HMX_RPC_SKEL_OUTPUT_NAME="${skel_output_name}"
                copy_artifact "${host_ship}/${operator_file}" "${output_dir}/${operator_file}"

                printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                    "${model}" "${heads_value}" "${operator_file}" "${skel_file}" \
                    "${fixed_m}" "${fixed_k}" "${fixed_n}" \
                    "${q_scale}" "${k_scale}" "${output_scale}" "${rq_scale}" \
                    >> "${manifest_tmp}"
            done
        done
    done
}

cd "${project_root}"
for model_index in "${!catalog_models[@]}"; do
    model="${catalog_models[model_index]}"
    model_k="${catalog_dims[model_index]}"
    model_q_base="${catalog_q_bases[model_index]}"
    model_k_base="${catalog_k_bases[model_index]}"
    model_output_scale="${catalog_output_scales[model_index]}"

    current_q_scales=()
    if (( ${#absolute_q_scales[@]} > 0 )); then
        current_q_scales=("${absolute_q_scales[@]}")
    else
        for multiplier in "${q_multipliers[@]}"; do
            current_q_scales+=("$(decimal_product "${model_q_base}" "${multiplier}")")
        done
    fi
    current_k_scales=()
    if (( ${#absolute_k_scales[@]} > 0 )); then
        current_k_scales=("${absolute_k_scales[@]}")
    else
        for multiplier in "${k_multipliers[@]}"; do
            current_k_scales+=("$(decimal_product "${model_k_base}" "${multiplier}")")
        done
    fi
    current_output_scales=()
    if (( ${#absolute_output_scales[@]} > 0 )); then
        current_output_scales=("${absolute_output_scales[@]}")
    else
        current_output_scales=("${model_output_scale}")
    fi

    for heads_value in "${heads[@]}"; do
        if (( ${#explicit_shapes[@]} > 0 )); then
            for shape in "${explicit_shapes[@]}"; do
                [[ "${shape}" =~ ^([1-9][0-9]*)[xX]([1-9][0-9]*)[xX]([1-9][0-9]*)$ ]]
                build_shape_variants "${model}" "${heads_value}" \
                    "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}" "${BASH_REMATCH[3]}"
            done
        else
            current_k="${model_k}"
            if [[ -n "${fixed_k_override}" ]]; then
                current_k="${fixed_k_override}"
            fi
            for fixed_m in "${m_values[@]}"; do
                for fixed_n in "${n_values[@]}"; do
                    build_shape_variants "${model}" "${heads_value}" \
                        "${fixed_m}" "${current_k}" "${fixed_n}"
                done
            done
        fi
    done
done

copy_artifact \
    "${host_ship}/hmx_qk_i8_operator_dlopen_example" \
    "${output_dir}/hmx_qk_i8_operator_dlopen_example"
mv -f "${manifest_tmp}" "${manifest}"
trap - EXIT
echo "Completed: ${variant_count} ARM operators; ${#built_dispatch_skels[@]} DSP dispatcher(s)"
echo "Manifest: ${manifest}"
