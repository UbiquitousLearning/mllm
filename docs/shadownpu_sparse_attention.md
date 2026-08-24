# ShadowNPU sparse attention for Qwen models

The prefill path is:

```text
AE head profile -> model scale calibration -> static HMX operator bank
                -> HMX INT8 QK -> CPU INT8 Top-k -> sparse QK/softmax/PV
```

Runtime Q/K/output scales are static. Each model uses one calibrated target
requant value and a compiled output scale for every Q/K bucket pair.
`TOPK_OVERSAMPLE=1`; rerank, fallback, dynamic scale, and DSP INT32 Top-k are
disabled. Decode attention remains dense.

## Scope and device

This commit packages one-command reproduction for Qwen2.5-1.5B and includes
the source compatibility paths needed by Qwen1.5-1.8B and Qwen1.5-0.5B. The
two Qwen1.5 models still require separately provisioned contexts, profiles,
banks, and model files.

All runs used a Xiaomi 14 (23127PN0CC, `houji`) with a Snapdragon 8 Gen 3
(SM8650). At final artifact verification the phone ran Android 16 (API 36).

| Model | Layers × heads × head dim | QNN chunks / logical M | HMX bank used | Result scope |
|---|---:|---:|---|---|
| Qwen2.5-1.5B | 28 × 12 × 128 | M256 × 2 chunks (= logical M512) | H12 per-head requant, M512/K128/N4160 | Packaged reference |
| Qwen1.5-1.8B | 24 × 16 × 128 | M256 | H16 per-head requant, M256/K128/N4160 | External artifacts required |
| Qwen1.5-0.5B | 24 × 16 × 64 | M128 | H16 per-head requant, M128/K64/N2048 or N4160 | External artifacts required |

## 1. Profile head retention

Run one profile per checkpoint:

```bash
python3 tools/head_sparsity_profiler/head_sparsity_profiler.py collect \
  --model /path/to/model --model-label MODEL_NAME \
  --calibration-text /path/to/wiki.valid.txt \
  --output-dir results/MODEL_NAME-profile \
  --average-retention 0.2 --local-files-only --resume
```

The tool ablates every head and layer on all complete 128-token WikiText
validation windows, allocates a 20% average retention budget, and writes
`head-retention.txt`. `--resume` continues the per-ablation checkpoint.

| Model | Profiling source | Profile SHA-256 | In this commit |
|---|---|---|---|
| Qwen2.5-1.5B | Published AE W8A8 results, converted with `convert` | `0425ffef895df428ff70cd013ac108f194831f9d0d77860f4a9ab44f21d29b32` | Yes |
| Qwen1.5-1.8B | BF16 AE-style collection | `403e37321961f2f9f90ee84d87525ba8c7fc53b2ebe653c9578210874b6600ff` | No |
| Qwen1.5-0.5B | BF16 AE-style collection | `4a6155ca6eb0dc2b7a464b087ac46a399aff897129807f73b6fe45c30b392591` | No |

Use `--activation-scales` for a new W8A8 collection. Omitting it profiles the
loaded checkpoint directly. Never reuse a profile across checkpoints.

## 2. Calibrate scales and build the bank

Collect Q/K absmax and dynamic output observations with a compatible probe
bank. Use several representative prompts:

```bash
python3 tools/hmx_scale_calibrator/hmx_scale_calibrator.py collect-adb \
  --serial <serial> --model MODEL_ID \
  --output results/MODEL-scale/calibration.log \
  --env LD_PRELOAD=/data/local/tmp/mllm/libfastrpc_session_redirect.so \
  --env 'ADSP_LIBRARY_PATH=/data/local/tmp/hmx-probe;/vendor/dsp/cdsp' \
  --env MLLM_HMX_INT8_OPERATOR_MANIFEST=/data/local/tmp/hmx-probe/manifest.tsv \
  -- /data/local/tmp/mllm/benchmark_qwen_npu [model arguments]

python3 tools/hmx_scale_calibrator/hmx_scale_calibrator.py calibrate \
  --log results/MODEL-scale/calibration.log \
  --model MODEL_ID --head-dim HEAD_DIM --require-model-sidecar \
  --bank-mode BANK_MODE \
  --output results/MODEL-scale/scale-profile.json
```

The calibrator emits a 3×3 Q/K grid, `scale-profile.catalog.txt`, and
`scale-profile.build.env`. Build the model-specific shape:

```bash
set -a
. results/MODEL-scale/scale-profile.build.env
set +a
HEXAGON_SDK_ROOT=/path/to/Hexagon_SDK \
HMX_OPERATOR_HEADS='HEAD_GROUPS' \
HMX_OPERATOR_M_VALUES=M HMX_OPERATOR_N_VALUES=N \
tools/hmx_int8_matmul/scripts/build_operator_bank.sh v75 /path/to/MODEL-bank
```

| Model | `MODEL_ID` | `HEAD_DIM` | `BANK_MODE` | `HEAD_GROUPS`, M, N |
|---|---|---:|---|---|
| Qwen2.5-1.5B | `qwen2_1p5b` | 128 | `target-requant` | `12`, 512, 4160 |
| Qwen1.5-1.8B | `qwen15_1p8b` | 128 | `target-requant` | `16`, 256, 4160 |
| Qwen1.5-0.5B | `qwen2_0p5b` | 64 | `target-requant` | `16`, 128, 2048 or 4160 |

`target-requant` gives each Q/K pair a compiled static output scale; it is not
runtime dynamic quantization. The heterogeneous ABI passes one Q scale, K
scale, and requant multiplier per head, so all three models execute one
full-head group per attention call. Set the group size to 12 for Qwen2.5 and
16 for either Qwen1.5 model, for example:

```bash
MLLM_HMX_INT8_FUSE_PER_HEAD_BUCKETS=1 \
MLLM_HMX_INT8_PER_HEAD_BUCKET_GROUP_HEADS=16
```

Validate the built bank with model identity, strict quality, recall, and
DSP/CPU score checks as documented in
`tools/hmx_scale_calibrator/README.md`.

## 3. Build, provision, and optionally profile the pipeline

Build the benchmark and native latency profiler:

```bash
cd scripts
ANDROID_NDK=/path/to/android-ndk ./build_android_qnn.sh
cmake --build ../build-arm-qnn \
  --target benchmark_qwen_npu profile_hmx_pipeline_latency -j
```

Provision the benchmark, vocabulary/merges, NPU model, Q40 decode model,
matching QNN context, model-specific bank/profile, QNN libraries, and
`libfastrpc_session_redirect.so`. Qwen2.5 uses an M256 QNN context with
`MLLM_QNN_SEQUENCE_TILE=256`: every logical M512 attention chunk is executed
as **M256 × 2 chunks**, not as one native M512 QNN chunk. Context generation
is separate from timed A/B runs.

The legacy Qwen1.5 exports need these compatibility settings in addition to
their model-specific paths and HMX configuration:

```bash
# Qwen1.5-1.8B
MLLM_QWEN_ACTIVATION_SCALE_FACTOR=2 \
MLLM_QWEN_CPU_LAST_TOKEN_REFINE=1 \
./benchmark_qwen_npu --model-size 1.8B-rotated --chunk-size 256 ...

# Qwen1.5-0.5B (also place the qwen05 package first in both library paths)
MLLM_QNN_HTP_OP_PACKAGE=libQnnLLaMAPackage_HTP_qwen05.so \
./benchmark_qwen_npu --model-size 0.5B-rotated --chunk-size 128 ...
```

The latency profile is optional and only drives the two-level greedy
scheduler. The reported reference runner uses FIFO and does not load this
profile. Schema v1 covers Qwen2.5-1.5B with M512/K128/N≤4160.

After deploying the profiler, bank, and head profile, collect three warmups
and 20 samples per point:

```bash
python3 tools/pipeline_latency_profiler/pipeline_latency_profiler.py collect-adb \
  --serial <serial> \
  --device-binary /data/local/tmp/profile_hmx_pipeline_latency \
  --device-manifest /data/local/tmp/hmx-bank/manifest.tsv \
  --device-head-profile /data/local/tmp/head-retention.txt \
  --device-raw /data/local/tmp/hmx-pipeline-raw.tsv \
  --output results/pipeline-profile/raw.tsv \
  --query-len 512 --max-key-len 4160 --warmup 3 --repetitions 20 \
  --main-cpu 2 --topk-cpus 4,6 --sparse-cpus 7 \
  --env 'LD_PRELOAD=/data/local/tmp/libfastrpc_session_redirect.so' \
  --env 'ADSP_LIBRARY_PATH=/data/local/tmp/hmx-bank;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp'

python3 tools/pipeline_latency_profiler/pipeline_latency_profiler.py aggregate \
  --raw results/pipeline-profile/raw.tsv \
  --manifest /path/to/manifest.tsv \
  --head-profile /path/to/head-retention.txt \
  --binary /path/to/profile_hmx_pipeline_latency \
  --model qwen2_1p5b --device houji \
  --main-cpu 2 --topk-cpus 4,6 --sparse-cpus 7 \
  --query-len 512 --max-key-len 4160 --minimum-samples 20 \
  --output results/pipeline-profile/runtime-profile.tsv
```

The profiler measures intrinsic NPU, Top-k, and sparse-stage costs on the
production resources; it does not replay the complete direct-three-stage
timeline. Load the validated profile with:

```bash
MLLM_HMX_PIPELINE_SCHEDULE=greedy \
MLLM_HMX_PIPELINE_LATENCY_PROFILE=results/pipeline-profile/runtime-profile.tsv \
./benchmark_qwen_npu ...
```

## 4. Reproduce the packaged Qwen2.5 A/B

The runner validates its required device paths, alternates dense/sparse order, rejects
RPC/QNN and strict retrieval failures, and reports medians:

```bash
ADB_SERIAL=<serial> \
DEVICE_ROOT=/data/local/tmp/mllm \
RUN_DIR=/data/local/tmp/mllm/qwen-run \
HMX_BANK=/data/local/tmp/mllm/hmx-qwen2-m512-h12-n4160 \
REPETITIONS=3 \
tools/shadownpu/run_qwen_ab.sh
```

`PROMPTS` is a space-separated list; `REPETITIONS` is a positive run count.
The runner never reboots the phone, resets CDSP, deletes remote data, or
changes unrelated processes.

Dense attention runs directly on the CPU7 main thread with the CPU7/4/5/6
OpenMP team. Sparse execution requests CPU2 for the main thread, CPU1 for
dispatcher/RPC, CPU4/6 for cooperative Top-k, CPU5 for packing, and CPU7 for
sparse attention.

## 5. Preserved device results

Speedup is dense TTFT divided by sparse TTFT within the same row. These rows
use different model-specific graphs and are not a controlled cross-model
comparison. Every row contains one device run. Failed and thermally throttled
runs are retained and marked.

| Model | Tokens | Dense TTFT | Sparse TTFT | Speedup | Actual retention | Runs | Quality/status |
|---|---:|---:|---:|---:|---:|---:|---|
| Qwen2.5-1.5B | 1024 | 5431.974 ms | 2421.385 ms | 2.243× | 0.139438 | 1 | PASS |
| Qwen2.5-1.5B | 2048 | 15934.188 ms | 5830.462 ms | 2.733× | 0.138950 | 1 | PASS |
| Qwen2.5-1.5B | 4096 | 102427.645 ms | 18479.409 ms | 5.543× | 0.138706 | 1 | PASS; dense thermally throttled |
| Qwen1.5-1.8B | 1024 | 6367.342 ms | 2927.293 ms | 2.175× | 0.166194 | 1 | PASS; H16 per-head requant; matched one-token CPU refinement |
| Qwen1.5-1.8B | 2048 | 18397.042 ms | 6854.959 ms | 2.684× | 0.165706 | 1 | PASS; H16 per-head requant; matched one-token CPU refinement |
| Qwen1.5-1.8B | 4096 | 64306.241 ms | 18375.520 ms | 3.500× | 0.165462 | 1 | PASS; H16 per-head requant; matched one-token CPU refinement |
| Qwen1.5-0.5B | 1024 | 3340.596 ms | 1461.830 ms | 2.285× | 0.166193 | 1 | PASS; N2048 bank |
| Qwen1.5-0.5B | 2048 | 13812.382 ms | 4335.775 ms | 3.186× | 0.165706 | 1 | FAIL in both modes; timing only; N4160 bank |
| Qwen1.5-0.5B | 4096 | 45178.520 ms | 13285.944 ms | 3.400× | 0.165461 | 1 | FAIL in both modes; timing only; N4160 bank |

The Qwen2.5 4096 speedup is inflated by dense thermal throttling and is not a
hardware-independent result. The two failed Qwen1.5-0.5B rows are diagnostic
execution timings, not quality-preserving speedups.

The preserved Qwen2.5 and Qwen1.5-0.5B timing rows predate the switch to the
recommended target-requant banks and used legacy fixed-output banks. A
single-run target-requant smoke test at 1024 tokens passed strict retrieval
for all three models; use a matched multi-run A/B before replacing the
preserved performance table.

## Host verification

```bash
cmake -S . -B build-test -DTEST=ON -DQNN=OFF \
  -DMLLM_BUILD_XNNPACK_BACKEND=OFF -DOPENCL=OFF
cmake --build build-test --target MLLM_TEST -j
./bin/MLLM_TEST --gtest_filter='INT8TopKTest.*:HMXPipelineSchedulerTest.*:CPUTest.CPUSparseSoftmaxValue*:CPUTest.CPUPatternSparseAttention*:ChunkPipelineTest.*'
python3 -m unittest discover -s tools/head_sparsity_profiler -p 'test_*.py'
python3 -m unittest discover -s tools/hmx_scale_calibrator -p 'test_*.py'
python3 -m unittest discover -s tools/pipeline_latency_profiler -p 'test_*.py'
```
