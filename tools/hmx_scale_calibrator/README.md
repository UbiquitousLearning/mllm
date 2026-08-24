# Per-model HMX scale calibration

This tool collects runtime Q/K absmax and output-requant observations, then
emits a model-bound 3×3 Q/K bucket grid and operator-bank inputs. Never reuse
the output across checkpoints.

## 1. Collect on device

Use a shape-compatible probe bank and an explicit benchmark command:

```bash
python3 tools/hmx_scale_calibrator/hmx_scale_calibrator.py collect-adb \
  --serial <serial> --model MODEL_ID \
  --output results/MODEL-scale/calibration.log \
  --env LD_PRELOAD=/data/local/tmp/mllm/libfastrpc_session_redirect.so \
  --env 'ADSP_LIBRARY_PATH=/data/local/tmp/hmx-probe;/vendor/dsp/cdsp' \
  --env MLLM_HMX_INT8_OPERATOR_MANIFEST=/data/local/tmp/hmx-probe/manifest.tsv \
  -- /data/local/tmp/mllm/benchmark_qwen_npu [model arguments]
```

The collector forces bucket diagnostics, dynamic output observation, and
`TOPK_OVERSAMPLE=1`, and writes a model-ID sidecar. Use several representative
prompts. Add `--recall` only for final diagnostics because it recomputes FP32
QK. The command does not push, delete, reboot, or change DSP sessions.

## 2. Generate bank inputs

```bash
python3 tools/hmx_scale_calibrator/hmx_scale_calibrator.py calibrate \
  --log results/MODEL-scale/calibration-1.log \
  --log results/MODEL-scale/calibration-2.log \
  --model MODEL_ID --head-dim HEAD_DIM --require-model-sidecar \
  --bank-mode BANK_MODE \
  --output results/MODEL-scale/scale-profile.json
```

The default `target-requant` mode derives a compiled output scale for each Q/K
pair. `fixed-output` compiles one output scale for the whole bank. Both modes
are static at runtime.

| Model | `MODEL_ID` | `HEAD_DIM` | `BANK_MODE` | Heads, M, N |
|---|---|---:|---|---|
| Qwen2.5-1.5B | `qwen2_1p5b` | 128 | `target-requant` | `12`, 512, 4160 |
| Qwen1.5-1.8B | `qwen15_1p8b` | 128 | `target-requant` | `16`, 256, 4160 |
| Qwen1.5-0.5B | `qwen2_0p5b` | 64 | `target-requant` | `16`, 128, 2048 or 4160 |

Build the emitted configuration:

```bash
set -a
. results/MODEL-scale/scale-profile.build.env
set +a
HMX_OPERATOR_HEADS='HEAD_GROUPS' \
HMX_OPERATOR_M_VALUES=M HMX_OPERATOR_N_VALUES=N \
tools/hmx_int8_matmul/scripts/build_operator_bank.sh v75 /path/to/new-bank
```

For an operator exporting the heterogeneous per-head requant ABI,
`target-requant` can fuse heads that selected different Q/K/output buckets.
Pass the selected Q scale, K scale, and requant multiplier for every head and
set the group capacity to the model's head count. Use 12 for Qwen2.5-1.5B and
16 for both Qwen1.5 models, for example:

```bash
MLLM_HMX_INT8_FUSE_PER_HEAD_BUCKETS=1 \
MLLM_HMX_INT8_PER_HEAD_BUCKET_GROUP_HEADS=16
```

Older operators without `prepare_execute_raw_per_head_requant_hn` must keep
heterogeneous fusion disabled.

## 3. Validate

Run the built bank with bucket and recall diagnostics, then check model
identity, strict quality, recall, and DSP/CPU score agreement:

```bash
python3 tools/hmx_scale_calibrator/hmx_scale_calibrator.py validate \
  --profile results/MODEL-scale/scale-profile.json \
  --log results/MODEL-scale/validation.log \
  --require-model-sidecar --require-quality --require-recall \
  --minimum-mean-recall 0.90

python3 -m unittest discover -s tools/hmx_scale_calibrator -p 'test_*.py'
```
