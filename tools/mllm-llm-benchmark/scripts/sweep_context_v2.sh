#!/usr/bin/env bash
set -euo pipefail

BIN="${BIN:-./build/bin/mllm-llm-benchmark}"
MODEL="${MODEL:-}"
CFG="${CFG:-./examples/llama/config_tiny_llama.json}"
THREADS="${THREADS:-8}"
RUNS="${RUNS:-1}"
COOLDOWN="${COOLDOWN:-0}"
TG_DH="${TG_DH:-256}"
TG_TTFT="${TG_TTFT:-2}"
CTX_LENS="${CTX_LENS:-256 512 1024 2048 4096}"
OUTDIR="${OUTDIR:-bench_context}"

mkdir -p "$OUTDIR"
OUTCSV="$OUTDIR/context_sweep_v2.csv"

echo "ts,git,arch,model,mode,ctx_len,pp,tg,threads,ttft_ms,prefill_ms,decode_ms,decode_ms_per_tok,peak_rss_kb,kv_est_kb" > "$OUTCSV"

GIT="$(git rev-parse --short=12 HEAD 2>/dev/null || echo NA)"
ARCH="$(uname -m)"
TS="$(date -Iseconds)"

kv_est_kb() {
  python3 - <<'PY'
import json, os, math
cfg = os.environ["CFG"]
ctx_len = int(os.environ["CTX_LEN"])
bpe = int(os.environ.get("KV_BYTES_PER_ELEM","2"))
j = json.load(open(cfg,"r"))
L = int(j["num_hidden_layers"])
H = int(j["num_attention_heads"])
KVH = int(j.get("num_key_value_heads", H))
hidden = int(j["hidden_size"])
head_dim = hidden // H
kv_bytes = 2 * L * KVH * head_dim * ctx_len * bpe
print(int(math.ceil(kv_bytes / 1024)))
PY
}

run_one () {
  local mode="$1"
  local ctx_len="$2"
  local tg="$3"

  if (( ctx_len <= tg )); then
    echo "skip: ctx_len=$ctx_len <= tg=$tg (mode=$mode)"
    return 0
  fi

  local pp=$((ctx_len - tg))
  if (( pp < 1 )); then pp=1; fi

  echo "==== mode=$mode ctx_len=$ctx_len pp=$pp tg=$tg ===="

  local ALLLOG="$OUTDIR/run_${mode}_ctx${ctx_len}.all"
  local TIMELOG="$OUTDIR/run_${mode}_ctx${ctx_len}.time"

  set +e
  # pass ctx as cache limit
  /usr/bin/time -v \
    "$BIN" -n tiny_llama -m "$MODEL" -c "$CFG" \
      -pp "$pp" -tg "$tg" -t "$THREADS" -cl "$ctx_len" -r "$RUNS" -cs "$COOLDOWN" \
    >"$ALLLOG" 2>"$TIMELOG"
  local EXIT_CODE=$?
  set -e

  if [ $EXIT_CODE -ne 0 ]; then
    echo "run failed with exit code $EXIT_CODE: mode=$mode ctx_len=$ctx_len"
    return 1
  fi

  local TTFT_MS PREFILL_MS DECODE_MS PEAK_RSS_KB KV_EST_KB
  TTFT_MS="$(grep -oP 'TTFT\s*:\s*\K[0-9.]+' "$ALLLOG" | head -n 1 || echo 0)"
  PREFILL_MS="$(grep -oP 'Prefill Latency\s*:\s*\K[0-9.]+' "$ALLLOG" | head -n 1 || echo 0)"
  DECODE_MS="$(grep -oP 'Decode Latency\s*:\s*\K[0-9.]+' "$ALLLOG" | head -n 1 || echo 0)"

  local DECODE_PER_TOK
  DECODE_PER_TOK="$(python3 -c "tg=float('$tg'); d=float('$DECODE_MS'); print(d/tg if tg>0 else 0.0)")"

  PEAK_RSS_KB="$(grep -oP 'Maximum resident set size \(kbytes\):\s*\K[0-9]+' "$TIMELOG" | head -n 1 || echo 0)"
  KV_EST_KB="$(CFG="$CFG" CTX_LEN="$ctx_len" KV_BYTES_PER_ELEM="${KV_BYTES_PER_ELEM:-2}" kv_est_kb || echo 0)"

  echo "TTFT=$TTFT_MS ms  Prefill=$PREFILL_MS ms  Decode=$DECODE_MS ms  Decode/tok=$DECODE_PER_TOK ms  peakRSS=$PEAK_RSS_KB KB  KV_est=$KV_EST_KB KB"

  echo "$TS,$GIT,$ARCH,tiny_llama,$mode,$ctx_len,$pp,$tg,$THREADS,$TTFT_MS,$PREFILL_MS,$DECODE_MS,$DECODE_PER_TOK,$PEAK_RSS_KB,$KV_EST_KB" >> "$OUTCSV"
}

for CTX in $CTX_LENS; do
  run_one "decode_heavy" "$CTX" "$TG_DH"
  run_one "prefill_ttft" "$CTX" "$TG_TTFT"
done

echo
echo "DONE -> $OUTCSV"
