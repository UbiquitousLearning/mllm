# [Bench] CPU Context Sweep Benchmark

CPU-only benchmark for measuring prefill/decode latency and memory across different context lengths on x86_64.

## What is Context Sweep?

`sweep_context_v2.sh` runs the benchmark binary at different context lengths
(256, 512, 1024, 2048, 4096) and records:

- Prefill / TTFT latency
- Decode per-token latency
- Peak RSS memory
- KV cache size estimate

It also captures GEMM shapes via `MLLM_MATMUL_SHAPE_LOG=1` (M, N, K per matmul call),
which is useful for understanding compute patterns at each context length.

This folder contains:
- `data/` : raw CSVs (context sweep + summary)
- `plots/`: generated figures (png)
- `snapshot.md`: 1-page conclusion (figures + key takeaways)

## Quick Repro (x86_64 example)

```bash
# 1. Build 
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DBUILD_TESTING=OFF \
  -DMLLM_ENABLE_TOOLS=ON \
  -DCMAKE_CXX_FLAGS="-march=native"

cmake --build build -j"$(nproc)"

# 2. Run Context Sweep & View Shape Log
chmod +x sweep_context_v2.sh
export BIN=./build/bin/mllm-llm-benchmark
export MLLM_MATMUL_SHAPE_LOG=1

# Run a quick test
./sweep_context_v2.sh
```

Outputs:

* `bench_context/context_sweep_v2.csv`
* <img width="2879" height="1366" alt="image" src="https://github.com/user-attachments/assets/46a4cb2b-6a22-4fba-81e2-dccf41bf4357" />
* <img width="2837" height="1437" alt="image" src="https://github.com/user-attachments/assets/3468b710-c7c8-4bab-bfe6-afaf6826d5ed" />

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
