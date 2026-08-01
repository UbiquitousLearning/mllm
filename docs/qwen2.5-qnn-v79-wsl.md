# Qwen2.5-1.5B on Qualcomm QNN HTP v79

This document records the tested WSL-to-Android deployment flow for mllm v1, Qwen2.5-1.5B-Instruct, and a Snapdragon device exposing HTP v79.

## Current status

| Item | Status |
| --- | --- |
| Android arm64 build | Passed |
| QNN runtime deployment | Passed |
| Custom op package build for HTP v79 | Passed |
| Real-device QNN/HTP execution | Passed, exit code 0 with HVX evidence |
| Log and answer pull-back to Windows | Passed |
| NPU answer-quality acceptance | **Not passed**; see [Known quality issue](#known-quality-issue) |

An execution `PASS` only proves that the QNN graph ran on HTP. It does not prove that the generated answer is correct.

## Tested environment

- Windows host with WSL 2
- Android NDK r26c
- Qualcomm AI Runtime (QAIRT/QNN) 2.48.0.260626
- Hexagon SDK 6.6.0.0
- Snapdragon HTP v79 device connected through ADB
- Qwen2.5-1.5B-Instruct
- NPU prefill model: `Qwen2.5-1.5B-Instruct_rotated-noshadow.mllm`
- CPU decode model: `Qwen2.5-1.5B-Instruct_rotated-Q40.mllm`

The versions above are the combination used for the successful v79 execution test. Other QAIRT/Hexagon SDK combinations have not been accepted by this deployment record.

## Host layout

The helper scripts default to the following layout:

```text
workspace/
├── mllm/                         # this repository
└── models/
    ├── Qwen2.5-1.5B-Instruct/
    ├── Qwen2.5-1.5B-Instruct_rotated-noshadow.mllm
    └── Qwen2.5-1.5B-Instruct_rotated-Q40.mllm
```

SDKs, model files, generated QNN contexts, binaries, and device logs must not be committed to Git.

## Configure WSL

Set paths to your local installations:

```bash
export ANDROID_NDK=/path/to/android-ndk-r26c
export ANDROID_NDK_ROOT="$ANDROID_NDK"
export QNN_SDK_ROOT=/path/to/qairt/2.48.0.260626
export HEXAGON_SDK_ROOT=/path/to/Hexagon_SDK/6.6.0.0
export HTP_ARCH=v79
```

If an SDK was extracted on Windows, symbolic links may have been materialized as tiny text files. Inspect first, then apply the repair only to the exact SDK directory:

```bash
./scripts/restore_windows_sdk_symlinks.sh "$HEXAGON_SDK_ROOT"
./scripts/restore_windows_sdk_symlinks.sh "$HEXAGON_SDK_ROOT" --apply
```

## Build

Build the custom op package for Android CPU and HTP v79:

```bash
./scripts/build_qnn_op_package.sh
```

Build the Android mllm binary with the external QAIRT headers:

```bash
./scripts/build_android_qnn.sh
```

Expected outputs include:

```text
bin-arm-qnn/demo_qwen_npu
mllm/backends/qnn/LLaMAOpPackageHtp/LLaMAPackage/build/aarch64-android/libQnnLLaMAPackage.so
mllm/backends/qnn/LLaMAOpPackageHtp/LLaMAPackage/build/hexagon-v79/libQnnLLaMAPackage.so
```

## Model conversion

The repository includes Qwen2.5 configuration examples and a small calibration corpus for conversion smoke tests:

```bash
cd tools/qnn_convertor
python -m pip install -r requirements-qwen2.5.txt
python prepare_calibration.py calibration/qwen2.5/val.jsonl
python get_distribution.py --config_file config/qwen2.5-1.5b-qnn.json
python export_qnn_model.py --config_file config/qwen2.5-1.5b-qnn.json
python export_rotate_model.py --config_file config/qwen2.5-1.5b-rotate.json
```

The bundled four-sample corpus is only a pipeline smoke test. It is not sufficient for production quantization or answer-quality acceptance. Replace it with a representative calibration set covering languages, prompt lengths, chat templates, technical text, and expected device workloads before producing release weights.

Convert the exported PyTorch artifacts to mllm format with the standard converter, then create the rotated Q4 CPU decoding model with the normal mllm quantizer. Keep the NPU and CPU artifacts generated from the same base model and rotation matrix.

## Push to the device

The WSL helper pushes the QAIRT runtime, v79 skeleton, custom op packages, tokenizer files, both models, and the demo binary:

```bash
export HOST_MODELS_DIR=/path/to/workspace/models
./scripts/run_qwen_qnn.sh
```

Files are deployed below `/data/local/tmp/mllm` by default. Override the destination with `MLLM_DEVICE_ROOT`.

## One-click inference from Windows

After the runtime, models, and binary have been deployed once, run:

```powershell
.\scripts\run_qwen_npu_device.ps1 `
  -Prompt "What is an LLM?" `
  -MaxNewTokens 12
```

Or use the CMD wrapper:

```cmd
scripts\run_qwen_npu_device.cmd -Prompt "Introduce yourself." -MaxNewTokens 12
```

Useful options:

- `-Serial <adb-serial>` selects a device when multiple devices are connected.
- `-AdbPath C:\path\to\adb.exe` selects a non-default ADB executable.
- `-PushBinary` pushes the current `bin-arm-qnn/demo_qwen_npu` before inference.
- `-OutputDir <path>` changes the local result directory.

Every run pulls two timestamped files into `device-results/`:

- `qwen_npu_<timestamp>_result.txt`: concise question and answer
- `qwen_npu_<timestamp>_full.log`: full QNN and inference log

The script also checks for `Number of HVX threads used` in the device log so a CPU-only process is not reported as NPU execution.

## Known quality issue

The v79 graph currently executes successfully, but the tested NPU model does not preserve answer quality. For example, the prompt `What is an LLM?` produced an unrelated first token during NPU prefill and then nonsensical continuation. A one-token test proves that the corruption starts before CPU autoregressive decoding.

The same tokenizer and rotated Q4 model produce coherent basic answers on the CPU-only path. This isolates the unresolved issue to the NPU prefill path, most likely one or more of:

- insufficient activation calibration;
- incorrect exported INT8 scales or weights;
- a numerical mismatch in QNN graph construction or the v79 custom-op path.

Before declaring the deployment production-ready:

1. Expand the calibration corpus substantially.
2. Compare FP32 and INT8 logits layer by layer and at the first generated token.
3. Add a conversion-time accuracy gate instead of accepting every profiled sample.
4. Run a fixed multilingual and instruction-following prompt suite on both CPU and NPU.
5. Inspect the pulled answer files; do not use process exit code as the quality criterion.

## Troubleshooting

- `QNN_SDK_ROOT must point to ... include/QNN`: select the extracted QAIRT root, not its parent directory.
- Missing `libQnnHtpV79Skel.so`: use a QAIRT package containing `lib/hexagon-v79/unsigned`.
- Hexagon compiler not found: verify `HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/*/Tools/bin/hexagon-clang++`.
- `adb` unavailable in WSL: place Windows `adb.exe` on `PATH` or export `ADB=/mnt/c/path/to/adb.exe`.
- Device file missing: rerun `scripts/run_qwen_qnn.sh`, or use `-PushBinary` when only the demo executable changed.
