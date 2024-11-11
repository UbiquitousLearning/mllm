# Qualcomm AI Engine Direct(QNN) backend

Currently, this is only preliminary support and is under active development for better performance and more supported models.

## QNN Environment Set Up
The QNN backend relies on the Qualcomm QNN framework and Hexagon SDK to compile LLM-specific operators. Please download them using [QPM](https://qpm.qualcomm.com/). The compiling environment only supports Linux now.

Version requirements:
* QNN: [Linux v2.20](https://qpm.qualcomm.com/#/main/tools/details/qualcomm_neural_processing_sdk)
* Hexagon SDK: [Linux 5.5.0.1](https://qpm.qualcomm.com/#/main/tools/details/HexagonSDK5.x)  (Some accounts may have no permission to access this SDK and may need to contact Qualcomm for support.)

After downloading and installing the two SDKs use "qpm-cli", copy the SDK directories into the following paths:
* mllm/src/backends/qnn/qualcomm_ai_engine_direct_220/
* mllm/src/backends/qnn/HexagonSDK/

## Op Package Compile

To use QNN offload, the CPU & HTP QNN op package are needed, the following scripts will build QNN op package needed by the project.

```bash
export QNN_SDK_ROOT=mllm/src/backends/qnn/qualcomm_ai_engine_direct_220/
export ANDROID_NDK_ROOT=/path/to/your/ndk
export PATH=$PATH:$ANDROID_NDK_ROOT

source mllm/src/backends/qnn/HexagonSDK/setup_sdk_env.source
source $QNN_SDK_ROOT/bin/envsetup.sh

cd mllm/src/backends/qnn/LLaMAOpPackageHtp/LLaMAPackage/
make htp_aarch64 && make htp_v75
```

## Build & Rung 

Build the target with QNN backend.

```bash
cd ../script
./build_qnn_android.sh
```

Currently, there are two style of modeling, the Module API and the old implementation. The demo of the Module API is in `examples/demo_qwen_npu.cpp` which is in a **user friendly style**, and the old implementation is in `examples/main_qwen_npu.cpp` which supports **the chunk pipeline prefilling**.

Download the model from [here](https://huggingface.co/mllmTeam/qwen-1.5-1.8b-chat-mllm/blob/main/), or using the following instructions

```bash
mkdir ../models && cd ../models
# Download int8 model used by npu & q4k model used by cpu
wget https://huggingface.co/mllmTeam/qwen-1.5-1.8b-chat-mllm/resolve/main/qwen-1.5-1.8b-chat-int8.mllm?download=true  -O qwen-1.5-1.8b-chat-int8.mllm
wget https://huggingface.co/mllmTeam/qwen-1.5-1.8b-chat-mllm/resolve/main/qwen-1.5-1.8b-chat-q4k.mllm?download=true  -O qwen-1.5-1.8b-chat-q4k.mllm
```

Run on an android phone with at least 16GB of memory.

```bash
cd ../script
./run_qwen_npu.sh
```

There are two arguments in the executable. `-s` is for the sequence length of prefilling, the default value is 64 in the demo we provided. `-c` for type of QNN prefilling options, when it is set to 1, the input will be splited into many chunks of sequence 256 and be executed in a pipeline. When it is set to 0, the input will be executed in one chunk.

Result are as followed:

```
> ./main_qwen_npu -s 512 -c 1
[Q] <|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Give me a short introduction to large language model.<|im_end|>
<|im_start|>assistant

[A] The large language model is a type of artificial intelligence that is designed to generate human-like text based on the input it receives It is typically trained on large datasets of text, such as books, articles, and web pages, and uses statistical models to learn patterns and relationships in the data The goal of a large language model is to generate text that is coherent
```
