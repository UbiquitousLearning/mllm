# Three-stage pipeline latency profiler

This tool measures the real runtime paths for:

1. NPU/HMX QK estimation;
2. CPU INT8 Top-k;
3. CPU sparse QK/softmax/PV.

It aggregates p50/p95 latency and binds the profile to the device, binary,
operator manifest, head profile, and CPU layout. Schema v1 supports
Qwen2/2.5-1.5B: 28 layers, 12 heads, head dimension 128, L0/L1 dense,
query length 512, and key length up to 4160.

The native profiler measures intrinsic NPU, cooperative Top-k, and sparse
stage costs on the production resources. It profiles head consumers with the
two-stage executor and sparse assistance disabled, then lets the scheduler
combine those costs; it does not replay a complete direct-three-stage run.

## Collect

Build the native profiler and check the inputs:

```bash
cmake --build build-arm-qnn --target profile_hmx_pipeline_latency -j

python3 tools/pipeline_latency_profiler/pipeline_latency_profiler.py inspect \
  --manifest /path/to/manifest.tsv \
  --head-profile /path/to/head-retention.txt \
  --model qwen2_1p5b \
  --query-len 512 --max-key-len 4160 --repetitions 20
```

Deploy the binary, operator bank, manifest, and head profile. Then collect the
raw TSV from the host:

```bash
python3 tools/pipeline_latency_profiler/pipeline_latency_profiler.py collect-adb \
  --device-binary /data/local/tmp/profile_hmx_pipeline_latency \
  --device-manifest /data/local/tmp/hmx-bank/manifest.tsv \
  --device-head-profile /data/local/tmp/head-retention.txt \
  --device-raw /data/local/tmp/hmx-pipeline-raw.tsv \
  --output results/pipeline-profile/raw.tsv \
  --main-cpu 2 --topk-cpus 4,6 --sparse-cpus 7 \
  --env 'LD_PRELOAD=/data/local/tmp/libfastrpc_session_redirect.so' \
  --env 'ADSP_LIBRARY_PATH=/data/local/tmp/hmx-bank;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp'
```

Formal collection uses three warmups and 20 measured runs. For a smoke test,
add `--max-key-len 512 --warmup 1 --repetitions 3`. `collect-adb` does not
push, delete, reboot, or change DSP sessions.

## Aggregate and use

```bash
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

The packaged reference runner uses FIFO and does not need a latency profile.
Enable the two-level greedy scheduler with:

```bash
MLLM_HMX_PIPELINE_SCHEDULE=greedy \
MLLM_HMX_PIPELINE_LATENCY_PROFILE=results/pipeline-profile/runtime-profile.tsv \
./benchmark_qwen_npu ...
```

If NPU and CPU stages were collected separately, pass the CPU file with
`--head-raw`. The validator rejects missing or duplicate stages, invalid
latencies, shape or retention mismatches, and stale hashes or CPU bindings.

```bash
python3 tools/pipeline_latency_profiler/pipeline_latency_profiler.py validate \
  --profile results/pipeline-profile/runtime-profile.tsv \
  --manifest /path/to/manifest.tsv \
  --head-profile /path/to/head-retention.txt \
  --binary /path/to/profile_hmx_pipeline_latency \
  --model qwen2_1p5b --device houji \
  --main-cpu 2 --topk-cpus 4,6 --sparse-cpus 7

python3 -m unittest discover -s tools/pipeline_latency_profiler -p 'test_*.py'
```
