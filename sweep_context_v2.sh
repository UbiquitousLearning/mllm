#!/usr/bin/env bash
set -euo pipefail

# Usage:
# BIN=... MODEL=... CFG=... ./sweep_context_v2.sh
# Optional Parameters:
# THREADS=8 RUNS=1 COOLDOWN=0
# TG_DH=256 TG_TTFT=2
# CLS="256 512 1024 2048 4096"
# decode_heavy: pp = cl - 256, tg = 256  (long prefill, then decode)
# prefill_ttft: pp = cl - 2,   tg = 2    (measure prefill/TTFT)

BIN="${BIN:-$HOME/mllm-runok/build/bin/mllm-llm-benchmark}"
MODEL="${MODEL:-/home/huangzhenhua/models/mllm_tinyllama/tinyllama-fp32.mllm}"
CFG="${CFG:-$HOME/mllm-runok/examples/llama/config_tiny_llama.json}"

THREADS="${THREADS:-8}"
RUNS="${RUNS:-1}"
COOLDOWN="${COOLDOWN:-0}"

TG_DH="${TG_DH:-256}"
TG_TTFT="${TG_TTFT:-2}"

CLS="${CLS:-256 512 1024 2048 4096}"

OUTDIR="${OUTDIR:-bench_context}"
mkdir -p "$OUTDIR"

OUTCSV="$OUTDIR/context_sweep_v2.csv"
echo "ts,git,arch,model,mode,cl,pp,tg,threads,ttft_ms,prefill_ms,decode_ms,decode_ms_per_tok,peak_rss_kb,kv_est_kb" > "$OUTCSV"

GIT="$(git rev-parse --short=12 HEAD 2>/dev/null || echo NA)"
ARCH="$(uname -m)"
TS="$(date -Iseconds)"

kv_est_kb() {
  python3 - <<'PY'
import json, os, math
cfg = os.environ["CFG"]
cl = int(os.environ["CL"])
bpe = int(os.environ.get("KV_BYTES_PER_ELEM","2"))  # 2=fp16, 4=fp32

j = json.load(open(cfg,"r"))
L = int(j["num_hidden_layers"])
H = int(j["num_attention_heads"])
KVH = int(j.get("num_key_value_heads", H))
hidden = int(j["hidden_size"])
head_dim = hidden // H

kv_bytes = 2 * L * KVH * head_dim * cl * bpe
print(int(math.ceil(kv_bytes / 1024)))
PY
}

run_one () {
  local mode="$1"
  local cl="$2"
  local tg="$3"

  if (( cl <= tg )); then
    echo "skip: cl=$cl <= tg=$tg (mode=$mode)"
    return 0
  fi
  local pp=$((cl - tg))
  if (( pp < 1 )); then pp=1; fi

  echo "==== mode=$mode cl=$cl pp=$pp tg=$tg ===="

  local ALLLOG="$OUTDIR/run_${mode}_cl${cl}.all"
  local TIMELOG="$OUTDIR/run_${mode}_cl${cl}.time"

  set +e  
  MLLM_MATMUL_SHAPE_LOG=1 /usr/bin/time -v \
    "$BIN" -n tiny_llama -m "$MODEL" -c "$CFG" \
      -pp "$pp" -tg "$tg" -t "$THREADS" -cl "$cl" -r "$RUNS" -cs "$COOLDOWN" \
    >"$ALLLOG" 2>"$TIMELOG"
  
  local EXIT_CODE=$?
  set -e  
  
  if [ -s "$TIMELOG" ]; then
      if [ $EXIT_CODE -ne 0 ]; then
          echo -e "\n\033[1;31m>>> [RUN FAILED] Error Log ($TIMELOG): >>>\033[0m"
          cat "$TIMELOG"
          echo -e "\033[1;31m<<< [END OF ERROR LOG] <<<\033[0m\n"
          return 1
      else
          echo -e "\n\033[1;33m>>> [SHAPE LOG CAPTURED] >>>\033[0m"
          grep -A 30 "\[MLLM_MATMUL_SHAPE_LOG\]" "$TIMELOG" || true
          echo -e "\033[1;33m<<< [END OF SHAPE LOG] <<<\033[0m\n"
      fi
  fi

  if [ $EXIT_CODE -ne 0 ]; then
      echo "run failed with exit code $EXIT_CODE: mode=$mode cl=$cl"
      return 1
  fi

  local TTFT_MS PREFILL_MS DECODE_MS PEAK_RSS_KB KV_EST_KB
  TTFT_MS="$(rg -o 'TTFT\s*: *[0-9.]+ ms' "$ALLLOG" | rg -o '[0-9.]+' | head -n 1 || echo 0)"
  PREFILL_MS="$(rg -o 'Prefill Latency\s*: *[0-9.]+ ms' "$ALLLOG" | rg -o '[0-9.]+' | head -n 1 || echo 0)"
  DECODE_MS="$(rg -o 'Decode Latency\s*: *[0-9.]+ ms' "$ALLLOG" | rg -o '[0-9.]+' | head -n 1 || echo 0)"

  local DECODE_PER_TOK
  DECODE_PER_TOK="$(python3 - <<PY
tg=float("$tg")
d=float("$DECODE_MS")
print(d/tg if tg>0 else 0.0)
PY
)"

  PEAK_RSS_KB="$(rg -o 'Maximum resident set size \(kbytes\): *[0-9]+' "$TIMELOG" | rg -o '[0-9]+' | head -n 1 || echo 0)"
  KV_EST_KB="$(CFG="$CFG" CL="$cl" KV_BYTES_PER_ELEM="${KV_BYTES_PER_ELEM:-2}" kv_est_kb || echo 0)"
  
  echo "TTFT=$TTFT_MS ms  Prefill=$PREFILL_MS ms  Decode=$DECODE_MS ms  Decode/tok=$DECODE_PER_TOK ms  peakRSS=$PEAK_RSS_KB KB  KV_est=$KV_EST_KB KB"

  echo "$TS,$GIT,$ARCH,tiny_llama,$mode,$cl,$pp,$tg,$THREADS,$TTFT_MS,$PREFILL_MS,$DECODE_MS,$DECODE_PER_TOK,$PEAK_RSS_KB,$KV_EST_KB" >> "$OUTCSV"
}

for CL in $CLS; do
  run_one "decode_heavy" "$CL" "$TG_DH"
  run_one "prefill_ttft" "$CL" "$TG_TTFT"
done

echo
echo "DONE -> $OUTCSV"
