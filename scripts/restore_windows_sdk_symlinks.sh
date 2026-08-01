#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 <sdk-root> [--apply]" >&2
    exit 2
fi

ROOT="$(realpath "$1")"
MODE="${2:-}"

if [[ ! -d "$ROOT" ]]; then
    echo "SDK root does not exist: $ROOT" >&2
    exit 1
fi

if [[ -n "$MODE" && "$MODE" != "--apply" ]]; then
    echo "Unknown mode: $MODE" >&2
    exit 2
fi

candidate_count=0
converted_count=0

while IFS= read -r -d '' file; do
    if ! grep -Iq . "$file"; then
        continue
    fi
    target="$(<"$file")"

    # Windows archive tools often materialize a Linux symlink as a tiny file
    # containing only its relative target. Ignore ordinary small data files.
    if [[ -z "$target" || "$target" == /* || "$target" == *$'\n'* ]]; then
        continue
    fi
    if [[ ! "$target" =~ ^[A-Za-z0-9_@+.,:/-]+$ ]]; then
        continue
    fi

    target_path="$(realpath -m "$(dirname "$file")/$target")"
    case "$target_path" in
        "$ROOT"/*) ;;
        *) continue ;;
    esac
    if [[ ! -e "$target_path" ]]; then
        continue
    fi

    candidate_count=$((candidate_count + 1))
    printf '%s -> %s\n' "$file" "$target"

    if [[ "$MODE" == "--apply" ]]; then
        rm -- "$file"
        ln -s -- "$target" "$file"
        converted_count=$((converted_count + 1))
    fi
done < <(find "$ROOT" -type f -size -256c -print0)

echo "Candidates: $candidate_count"
echo "Converted: $converted_count"
