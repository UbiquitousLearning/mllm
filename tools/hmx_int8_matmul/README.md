# HMX INT8 QK operator bank

This directory builds the fixed-shape operators used by the ShadowNPU sparse
attention path. Each ARM operator exposes immutable shape and Q/K/output scale
metadata, opens the shared DSP dispatcher through FastRPC, and executes signed
INT8 QK. The DSP implementation uses raw HMX instructions; it does not call a
Hexagon matrix-multiplication library.

## Standard bank

The recommended command builds the reproducible Qwen2.5-1.5B
target-requant bank:

- H=12, M=512, K=128, N=4160;
- Q scales `0.0434675, 0.0869351, 0.1738701`;
- K scales `0.0355190, 0.0710380, 0.142076`;
- target requant scale `0.0030974452601659333`, with one derived static output
  scale per Q/K pair;
- nine ARM operators and one shared DSP dispatcher skel;
- four HVX lanes, with results published after all 12 heads.

```bash
HEXAGON_SDK_ROOT=/path/to/Hexagon_SDK \
HMX_OPERATOR_TARGET_REQUANT_SCALE=0.0030974452601659333 \
scripts/build_operator_bank.sh v75 /path/to/output-bank
```

Omitting `HMX_OPERATOR_TARGET_REQUANT_SCALE` retains the legacy fixed-output
bank for controlled A/B tests.

The output directory contains `manifest.tsv`, `libhmx_qk_i8_*.so`, the shared
`libhmx_int8_rpc_skel_*_dispatcher.so`, and an operator smoke-test binary.
Runtime bucket dispatch uses the manifest and picks the nearest Q/K pair for
each head.

## Calibrated bank

`tools/hmx_scale_calibrator` writes a catalog and `scale-profile.build.env`.
Source that environment before the same build command to use model-specific
absolute scales. The build script also accepts explicit overrides such as
`HMX_OPERATOR_HEADS`, `HMX_OPERATOR_M_VALUES`, `HMX_OPERATOR_N_VALUES`,
`HMX_OPERATOR_Q_SCALES`, `HMX_OPERATOR_K_SCALES`, and
`HMX_OPERATOR_OUTPUT_SCALES`.

For a target-requant bank, build only the full-head capacity when all heads
will share one fused RPC. The heterogeneous ABI accepts `q_scale[H]`,
`k_scale[H]`, and `requant_scale[H]`, so each head may select a different
compiled bucket. Qwen2.5-1.5B uses H=12; both Qwen1.5 models use H=16 rather
than separate smaller-head operators.

## Numeric operation

Inputs are signed row-major INT8 tensors `Q=[H,M,K]` and `K=[H,N,K]`. HMX
computes exact INT32 accumulators, applies the signed-activation correction,
and requantizes on DSP:

```text
score_i8 = clamp(round((Q_i8 @ transpose(K_i8))
                       * q_scale * k_scale / output_scale), -128, 127)
```

The standard mllm path runs stable CPU INT8 Top-k directly on this output. It
does not use exact reranking, outlier fallback, dynamic scales, or DSP INT32
Top-k.

## Standalone smoke test

After deploying one ARM operator, the matching dispatcher skel, and the
generated test executable to an Android device:

```bash
scripts/run_operator_test.sh /path/to/output-bank/libhmx_qk_i8_<variant>.so
```

The test obtains shape/scales from operator metadata and compares the device
result with a CPU INT8 reference.
