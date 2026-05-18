#!/usr/bin/env bash

set -euo pipefail

QWEN2VL_QNN_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QWEN2VL_QNN_REPO_ROOT="$(cd "${QWEN2VL_QNN_SCRIPT_DIR}/../.." && pwd)"

pick_adb() {
  if [[ -n "${ADB:-}" ]]; then
    printf '%s\n' "${ADB}"
    return
  fi
  if command -v adb.exe >/dev/null 2>&1; then
    command -v adb.exe
    return
  fi
  if command -v adb >/dev/null 2>&1; then
    command -v adb
    return
  fi
  echo "Cannot find adb.exe or adb. Set ADB=/path/to/adb first." >&2
  return 1
}

remote_quote() {
  local value="$1"
  printf "'"
  printf "%s" "${value}" | sed "s/'/'\\\\''/g"
  printf "'"
}

timestamp() {
  date +%Y%m%d_%H%M%S
}

read_eval_cases() {
  local cases_file="$1"
  if [[ ! -f "${cases_file}" ]]; then
    echo "Cases file not found: ${cases_file}" >&2
    return 1
  fi
  sed -e 's/\r$//' "${cases_file}" | awk 'NF && $0 !~ /^#/'
}
