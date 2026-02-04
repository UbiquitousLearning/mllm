# Bench Artifacts: Context Sweep + Snapshot

This folder contains:
- `data/` : raw CSVs (context sweep + summary)
- `plots/`: generated figures (png)
- `snapshot.md`: 1-page conclusion (figures + key takeaways)

## Quick Repro (x86_64 example)

### 1) Build (Release-ish with debug symbols + frame pointer for perf)
```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CXX_FLAGS="-fno-omit-frame-pointer" \
  -DCMAKE_C_FLAGS="-fno-omit-frame-pointer"
ninja -C build mllm-llm-benchmark
````

### 2) Run context sweep (generates CSV)

```bash
cd ~/mllm-runok
BIN=~/mllm-runok/build/bin/mllm-llm-benchmark \
MODEL=/home/huangzhenhua/models/mllm_tinyllama/tinyllama-fp32.mllm \
CFG=~/mllm-runok/examples/llama/config_tiny_llama.json \
THREADS=8 RUNS=1 COOLDOWN=0 \
TG_DH=256 TG_TTFT=2 \
KV_BYTES_PER_ELEM=2 \
./sweep_context_v2.sh
```

Outputs:

* `bench_context/context_sweep_v2.csv`

### 3) Make snapshot plots (from the CSV)

If you already have `scripts/make_snapshot.py`, run:

```bash
python3 scripts/make_snapshot.py bench_context/context_sweep_v2.csv snapshots
```

Then copy artifacts into `bench_artifacts/` (optional):

```bash
mkdir -p bench_artifacts/data bench_artifacts/plots
cp -f bench_context/context_sweep_v2.csv bench_artifacts/data/
cp -f snapshots/context_sweep_v2.summary.csv bench_artifacts/data/
cp -f snapshots/*.png bench_artifacts/plots/
```

### 4) Perf (decode-heavy example)

```bash
cd ~/mllm-runok
rm -f perf.data perf.data.old
perf record -F 99 -g -- \
  "$BIN" -n tiny_llama -m "$MODEL" -c "$CFG" \
  -pp 32 -tg 256 -t 8 -cl 2048 -r 1 -cs 0

perf report --stdio --no-children --sort=overhead,symbol \
| rg '^\s*[0-9]+\.[0-9]+%\s' | head -n 20
```

## Snapshot

See `bench_artifacts/snapshot.md`.
