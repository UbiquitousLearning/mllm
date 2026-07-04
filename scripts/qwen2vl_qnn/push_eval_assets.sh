#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

ADB_BIN="$(pick_adb)"
CASES_FILE="${CASES_FILE:-${SCRIPT_DIR}/qwen2vl_eval_cases.tsv}"
ASSET_SRC="${ASSET_SRC:-}"
REMOTE_IMAGE_DIR="${REMOTE_IMAGE_DIR:-/data/local/tmp/mllm-qwen2vl/images/eval}"

if [[ -z "${ASSET_SRC}" ]]; then
  echo "Set ASSET_SRC to the local directory that contains the eval images." >&2
  echo "Example: ASSET_SRC=/path/to/images $0" >&2
  exit 1
fi

echo "ADB              : ${ADB_BIN}"
echo "Cases file       : ${CASES_FILE}"
echo "Local assets     : ${ASSET_SRC}"
echo "Remote image dir : ${REMOTE_IMAGE_DIR}"

"${ADB_BIN}" shell "mkdir -p $(remote_quote "${REMOTE_IMAGE_DIR}")" </dev/null

declare -A pushed=()
while IFS='|' read -r case_id image_file prompt; do
  [[ -n "${case_id}" ]] || continue
  [[ -n "${image_file}" ]] || continue
  if [[ -n "${pushed[${image_file}]:-}" ]]; then
    continue
  fi
  local_image="${ASSET_SRC}/${image_file}"
  if [[ ! -f "${local_image}" ]]; then
    echo "Missing local image for case ${case_id}: ${local_image}" >&2
    exit 1
  fi
  echo "Pushing ${image_file}"
  "${ADB_BIN}" push "${local_image}" "${REMOTE_IMAGE_DIR}/${image_file}" </dev/null
  pushed["${image_file}"]=1
done < <(read_eval_cases "${CASES_FILE}")

echo "Done. Pushed ${#pushed[@]} image(s)."
