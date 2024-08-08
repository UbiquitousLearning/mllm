# Qualcomm AI Engine Direct(QNN) backend

## QNN Environment Set Up
The QNN backend relies on the Qualcomm QNN framework and Hexagon SDK to compile LLM-specific operators. Please download them using [QPM](https://qpm.qualcomm.com/). The compiling environment only supports Linux now.

Version requirements:
* QNN: tools/Qualcomm® AI Stack/Qualcomm® AI Stack/v2.20
* Hexagon SDK: tools/Qualcomm Hexagon SDK Products/Qualcomm® Hexagon™ SDK 5.x/5.5.0.1

Install the SDK into the following paths:
* /mllm/src/backends/qnn/qualcomm_ai_engine_direct_220/
* /mllm/src/backends/qnn/HexagonSDK/

## Op Package Compile

To use QNN offload, the CPU & HTP QNN op package are needed, the following scripts will build QNN op package needed by the project.

```bash
export QNN_SDK_ROOT=mllm/src/backends/qnn/qualcomm_ai_engine_direct_220/
export ANDROID_NDK=/path/to/your/ndk
export PATH=$PATH:$ANDROID_NDK

source mllm/src/backends/qnn/HexagonSDK/setup_sdk_env.source
source $QNN_SDK_ROOT/bin/envsetup.sh

cd mllm/src/backends/qnn/LLaMAOpPackageHtp/LLaMAPackage/
make htp_aarch64 && make htp_v75
```