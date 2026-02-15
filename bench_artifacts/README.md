# [Bench] x86 CPU Baseline Profiler & Context Sweep Tool

**Disclaimer:** This toolchain currently runs strictly on the **CPU (x86_64)**. Its purpose is to establish a rigorous ground-truth CPU baseline, capturing explicit matrix shapes and memory usage boundaries. This data is intended to guide AOT Static Graph bucketing and memory pre-allocation for future NPU integration.

## What is Context Sweep?
`sweep_context_v2.sh` is an automated benchmarking workflow that tests the framework across scaling context lengths (e.g., 256 to 4096). It isolates prefill and decode stages to extract two critical metrics for AOT preparation:
1. **Shape Dominance:** Captures explicit GEMM `(M, N, K)` shapes to pinpoint exactly which bucket sizes the AOT compiler needs to nearest-pad (e.g., finding M=254 is dominant in prefill).
2. **KV Cache Memory Bounds:** Calculates theoretical and empirical (Peak RSS) VRAM requirements prior to execution, establishing strict lower bounds for AOT memory pre-allocation.

This folder contains:
- `data/` : raw CSVs (context sweep + summary)
- `plots/`: generated figures (png)
- `snapshot.md`: 1-page conclusion (figures + key takeaways)

## Quick Repro (x86_64 example)

```bash
# 1. Build (with toolchain enabled & SIMD fixes)
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
