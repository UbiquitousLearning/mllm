<h1 align="center">
mllm
</h1>


<div align="center">

**Fast and lightweight multimodal LLM inference engine for mobile and edge devices**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-website-blue)](https://ubiquitouslearning.github.io/mllm/)
[![GitHub Stars](https://img.shields.io/github/stars/UbiquitousLearning/mllm.svg)](https://github.com/UbiquitousLearning/mllm/stargazers)

[📚 Documentation](https://ubiquitouslearning.github.io/mllm/) • [🚀 Quick Start](#quick-starts) • [💡 Examples](./examples/) • [🛠️ Installation](#installation)

</div>

## Latest News

- [2025 Nov 23] 🔥🔥🔥 MLLM v2 released!
- [2025 Aug 28] Support for MLLM V1 is ending soon. Before its retirement, V1 will integrate the following features: GPT-OSS. MLLM will then transition to V2, which can be viewed on the V2 branch. V2 will include brand-new capabilities:
  - A more Pythonic model authoring approach with eager execution
  - Compilation support for easier NPU integration
  - Support for parallel execution of multiple models
  - A more refined engineering implementation
- [2025 Jul 30] Add Rotation Quantization method for QNN backend models and support Qwen-2-VL 2B（ViT profiling will integrate in v2）

## Key Features

1. **Pythonic eager execution** – Rapid model development  
2. **Unified hardware support** – Arm CPU, OpenCL GPU, QNN NPU  
3. **Advanced optimizations** – Quantization, pruning, speculative execution  
4. **NPU-ready IR** – Seamless integration with NPU frameworks  
5. **Deployment toolkit** – SDK + CLI inference tool

## The Role of MLLM

MLLM is the central hub of the AI inference stack. It connects optimization algorithms like Speculative Decoding, Pruning, and Quantization above with AI Compiler/Runtime layers (CANN, CUDA, MLIR) below for hardware execution. Highlighted in red, MLLM uniquely bridges algorithm innovation and hardware optimization, making it the indispensable node linking software ecosystem and hardware acceleration.

<div align="center">
  <img src="./assets/mllm_role.png" width="80%">
</div>

The mllm framework integrates seamlessly with popular community frameworks' checkpoints. Through mllm-convertor, it directly ingests PyTorch and SafeTensors models, quantizes and converts them into mllm format, which are then loaded and executed by mllm Runtime.

<div align="center">
  <img src="./assets/mllm_workflow.png" width="80%">
</div>

## Supported Models

### mllm v2

| Model(v1)                                                                   | CPU  | Hexagon NPU <br> INT8 |
|-----------------------------------------------------------------------------|------|-----------------------|
| [Qwen3-0.6B](https://github.com/QwenLM/Qwen3)                     | [✔️ w4a8](https://www.modelscope.cn/models/mllmTeam/Qwen3-0.6B-w4a32kai)  |  | 
| [Qwen3-1.7B](https://github.com/QwenLM/Qwen3)                     | [✔️ w4a8](https://www.modelscope.cn/models/mllmTeam/Qwen3-1.7B-w4a8-i8mm-kai)  |  |
| [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR)       | [✔️ w4a8](https://www.modelscope.cn/models/mllmTeam/DeepSeek-OCR-w4a8-i8mm-kai)  |  |
| [SmolLM3](https://huggingface.co/blog/smollm3)| [✔️ w4a8](https://www.modelscope.cn/models/mllmTeam/SmolLM3-3B-w4a8-i8mm-kai)  |  |
| [Qwen2-VL-2B-Instruct](https://qwenlm.github.io/zh/blog/qwen2-vl/)|[✔️ w4a8](https://www.modelscope.cn/models/mllmTeam/Qwen2-VL-2B-Instruct-w4a32kai) ||
| [Qwen2-VL-7B-Instruct](https://qwenlm.github.io/zh/blog/qwen2-vl/)|[✔️ w4a8](https://www.modelscope.cn/models/mllmTeam/Qwen2-VL-7B-Instruct-w4a32kai)||
| [Qwen2.5-VL-3B-Instruct](https://qwenlm.github.io/blog/qwen2.5-vl/)|[✔️ w4a8](https://www.modelscope.cn/models/mllmTeam/Qwen2.5-VL-3B-Instruct-w4a32kai)||
| [Qwen2.5-VL-7B-Instruct](https://qwenlm.github.io/blog/qwen2.5-vl/)|[✔️ w4a8](https://www.modelscope.cn/models/mllmTeam/Qwen2.5-VL-7B-Instruct-w4a32kai)||

### mllm v1

| Model(v1)                                                                       | CPU <br> FP32 | CPU <br> INT4  | Hexagon NPU <br> INT8 |
|-----------------------------------------------------------------------------|------|-----|----------------------------|
| [LLaMA 2 7B](https://github.com/facebookresearch/llama)                   | [✔️](https://huggingface.co/mllmTeam/llama-2-7b-mllm/tree/main)  | [✔️](https://huggingface.co/mllmTeam/llama-2-7b-mllm/tree/main)   |  |
| [LLaMA 3 1B](https://github.com/meta-llama/llama3)                   | [✔️](https://huggingface.co/mllmTeam/llama-3.2-1b-mllm/tree/main)  | [✔️](https://huggingface.co/mllmTeam/llama-3.2-1b-mllm/tree/main)   |  |
| [LLaMA 3 3B](https://github.com/meta-llama/llama3)                   | [✔️](https://huggingface.co/mllmTeam/llama-3.2-3b-mllm/tree/main)  | [✔️](https://huggingface.co/mllmTeam/llama-3.2-3b-mllm/tree/main)   |  |
| [Alpaca 7B](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2)                | [✔️](https://huggingface.co/mllmTeam/chinese-alpaca-7b-mllm/tree/main)  | [✔️](https://huggingface.co/mllmTeam/chinese-alpaca-7b-mllm/tree/main)   |  |
| [TinyLLaMA 1.1B](https://github.com/jzhang38/TinyLlama)                     | [✔️](https://huggingface.co/mllmTeam/tinyllama-1.1b-mllm/tree/main)  | [✔️](https://huggingface.co/mllmTeam/tinyllama-1.1b-mllm/tree/main)   |  |
| [LLaVA 7B](https://github.com/haotian-liu/LLaVA)                            | [✔️](https://huggingface.co/mllmTeam/llava-1.5-7b-mllm/tree/main)  | [✔️](https://huggingface.co/mllmTeam/llava-1.5-7b-mllm/tree/main)   |  |
| [Gemma 2B](https://github.com/google/gemma_pytorch)                         | [✔️](https://huggingface.co/mllmTeam/gemma-2b-mllm/tree/main)  | [✔️](https://huggingface.co/mllmTeam/gemma-2b-mllm/tree/main)   |  |
| [Gemma 2 2B](https://github.com/google/gemma_pytorch)                         | [✔️](https://huggingface.co/mllmTeam/gemma-2-2b-mllm/tree/main)  | [✔️](https://huggingface.co/mllmTeam/gemma-2-2b-mllm/tree/main)   |  |
| [Qwen 1.5 0.5B](https://github.com/QwenLM/Qwen)                                 | [✔️](https://huggingface.co/mllmTeam/qwen-1.5-0.5b-mllm/tree/main)  | [✔️](https://huggingface.co/mllmTeam/qwen-1.5-0.5b-mllm/tree/main)   | ✔️ |
| [Qwen 1.5 1.8B](https://github.com/QwenLM/Qwen)                            | [✔️](https://huggingface.co/mllmTeam/qwen-1.5-1.8b-chat-mllm)  | [✔️](https://huggingface.co/mllmTeam/qwen-1.5-1.8b-chat-mllm)   | [✔️](https://huggingface.co/mllmTeam/qwen-1.5-1.8b-chat-mllm) |
| [Qwen 2.5 1.5B](https://github.com/QwenLM/Qwen2.5) | [✔️](https://huggingface.co/mllmTeam/qwen-2.5-1.5b-mllm/tree/main) | [✔️](https://huggingface.co/mllmTeam/qwen-2.5-1.5b-mllm/tree/main) | ✔️ |
| [Qwen 3 0.6B](https://github.com/QwenLM/Qwen3) | [✔️](https://huggingface.co/mllmTeam/qwen-3-0.6b-mllm/tree/main) | [✔️](https://huggingface.co/mllmTeam/qwen-3-0.6b-mllm/tree/main) | |
| [Mistral 7B](https://github.com/mistralai/mistral-src)                      | [✔️](https://huggingface.co/mllmTeam/mistral-7b-instruct-v0.2-mllm/tree/main)  | [✔️](https://huggingface.co/mllmTeam/mistral-7b-instruct-v0.2-mllm/tree/main)   |  |
| [Yi 6B](https://huggingface.co/01-ai/Yi-1.5-6B)                             | [✔️](https://huggingface.co/mllmTeam/yi-1.5-6b-chat-mllm/tree/main)  | [✔️](https://huggingface.co/mllmTeam/yi-1.5-6b-chat-mllm/tree/main)   |  |
| [StableLM 2 1.6B](https://github.com/Stability-AI/StableLM)                     | [✔️](https://huggingface.co/mllmTeam/stablelm-2-1.6b-chat-mllm/tree/main)  | [✔️](https://huggingface.co/mllmTeam/stablelm-2-1.6b-chat-mllm/tree/main)   |  |
| [OPT 1.3B](https://github.com/facebookresearch/metaseq/tree/main/projects/OPT)                     | [✔️](https://huggingface.co/mllmTeam/opt-1.3b-mllm/tree/main)  | [✔️](https://huggingface.co/mllmTeam/opt-1.3b-mllm/tree/main)   |  |
| [Phi 3 mini 3.8B](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)                     |  [✔️](https://huggingface.co/mllmTeam/phi-3-mini-instruct-mllm/tree/main)   | [✔️](https://huggingface.co/mllmTeam/phi-3-mini-instruct-mllm/tree/main)   |  |
| [MiniCPM 2B](https://huggingface.co/openbmb/MiniCPM-2B-dpo-fp32)                     |  [✔️](https://huggingface.co/mllmTeam/minicpm-2b-dpo-mllm/tree/main)   | [✔️](https://huggingface.co/mllmTeam/minicpm-2b-dpo-mllm/tree/main)   |  |
| [MiniCPM 3 4B](https://huggingface.co/openbmb/MiniCPM3-4B)                     |  [✔️](https://huggingface.co/mllmTeam/minicpm3-4b-mllm/tree/main)   | [✔️](https://huggingface.co/mllmTeam/minicpm3-4b-mllm/tree/main)   |  |
| [MiniCPM MoE 8x2B](https://huggingface.co/openbmb/MiniCPM-MoE-8x2B)                     |  [✔️](https://huggingface.co/mllmTeam/minicpm-moe-8x2b-mllm/tree/main)   | [✔️](https://huggingface.co/mllmTeam/minicpm-moe-8x2b-mllm/tree/main)   |  |
| [SmolLM 1.7B](https://huggingface.co/HuggingFaceTB/SmolLM-1.7B-Instruct)                     |  [✔️](https://huggingface.co/mllmTeam/smollm-1.7b-instruct-mllm/tree/main)   | [✔️](https://huggingface.co/mllmTeam/smollm-1.7b-instruct-mllm/tree/main)   |  |
| [DCLM 1B](https://huggingface.co/TRI-ML/DCLM-1B) | [✔️](https://huggingface.co/mllmTeam/dclm-1b-mllm/tree/main)| [✔️](https://huggingface.co/mllmTeam/dclm-1b-mllm/tree/main)| |
| [OpenELM 1.1B](https://github.com/apple/corenet/tree/main/projects/openelm) | [✔️](https://huggingface.co/mllmTeam/openelm-1.1b-mllm/tree/main)| [✔️](https://huggingface.co/mllmTeam/openelm-1.1b-mllm/tree/main)| |
[PhoneLM 1.5B](https://github.com/UbiquitousLearning/PhoneLM) | [✔️](https://huggingface.co/mllmTeam/phonelm-1.5b-mllm/tree/main)| [✔️](https://huggingface.co/mllmTeam/phonelm-1.5b-mllm/tree/main)| [✔️](https://huggingface.co/mllmTeam/phonelm-1.5b-mllm/tree/main)|
| [Fuyu 8B](https://www.adept.ai/blog/fuyu-8b)                                | [✔️](https://huggingface.co/mllmTeam/fuyu-8b-mllm/tree/main)  | [✔️](https://huggingface.co/mllmTeam/fuyu-8b-mllm/tree/main)   |  
| [Vision Transformer](https://github.com/google-research/vision_transformer) | [✔️](https://huggingface.co/mllmTeam/vit-base-patch16-224-mllm/tree/main)  | [✔️](https://huggingface.co/mllmTeam/vit-base-patch16-224-mllm/tree/main)   | 
| [CLIP](https://github.com/openai/CLIP)                                      | [✔️](https://huggingface.co/mllmTeam/clip-vit-base-patch32-mllm/tree/main)  | [✔️](https://huggingface.co/mllmTeam/clip-vit-base-patch32-mllm/tree/main)   |
| [ImageBind](https://github.com/facebookresearch/ImageBind) (3 modalities)   | [✔️](https://huggingface.co/mllmTeam/imagebind_huge-mllm/tree/main)  | [✔️](https://huggingface.co/mllmTeam/imagebind_huge-mllm/tree/main)   | 
| [LLaVA 7B](https://github.com/haotian-liu/LLaVA)                            | [✔️](https://huggingface.co/mllmTeam/llava-1.5-7b-mllm/tree/main)  | [✔️](https://huggingface.co/mllmTeam/llava-1.5-7b-mllm/tree/main)   |
| [Phi-3-Vision](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct)                            | [✔️](https://huggingface.co/mllmTeam/phi-3-vision-instruct-mllm/tree/main)  | [✔️](https://huggingface.co/mllmTeam/phi-3-vision-instruct-mllm/tree/main)   |
| [Qwen2-VL 2B](https://github.com/QwenLM/Qwen2-VL)                            | [✔️](https://huggingface.co/mllmTeam/qwen-2-vl-2b-instruct--mllm/tree/main)  | [✔️](https://huggingface.co/mllmTeam/qwen-2-vl-2b-instruct--mllm/tree/main)   | ✔️ |

## Tested Devices

| Device | OS | CPU | GPU | NPU |
| :---: | :---: | :---: | :---: | :---: |
| PC-X86-w/oAVX512  | Ubuntu 22.04  | ![build-passing](https://img.shields.io/badge/build-passing-green) | - | - |
| Nvidia A40  | Ubuntu 22.04  | - | ![build-passing](https://img.shields.io/badge/build-passing-green) | - |
| Xiaomi14-8Elite   | Android 15    | ![build-passing](https://img.shields.io/badge/build-passing-green) | - | ![build-pending](https://img.shields.io/badge/build-pending-gray) |
| OnePlus13-8Elite  | Android 15    | ![build-passing](https://img.shields.io/badge/build-passing-green) | - | ![build-pending](https://img.shields.io/badge/build-pending-gray) |
| MacMini-M4        | MacOS 15.5    | ![build-passing](https://img.shields.io/badge/build-passing-green) | - | - |
| OrangePi AI Pro(310B)        | Ubuntu 22.04    | - | - | ![build-pending](https://img.shields.io/badge/build-pending-gray) |
| OrangePi AI Studio(310P)        | Ubuntu 22.04    | - | - | ![build-pending](https://img.shields.io/badge/build-pending-gray) |

## Quick Starts

### Serving LLMs with mllm-cli

We have developed a C SDK wrapper for the MLLM C++ SDK to enable seamless integration with Golang. Leveraging this wrapper, we've built the mllm-cli command-line tool in Golang, which is about to be released soon.

### Inference with VLM using C++ API

The following example demonstrates how to perform inference on a multimodal vision-language model (VLM), specifically Qwen2-VL, using the mllm framework's C++ API. The process includes loading the model configuration, initializing the tokenizer, loading pretrained weights, processing image-text inputs, and performing streaming text generation.

```c++
auto qwen2vl_cfg        = Qwen2VLConfig(config_path);
auto qwen2vl_tokenizer  = Qwen2VLTokenizer(tokenizer_path);
auto qwen2vl            = Qwen2VLForCausalLM(qwen2vl_cfg);

qwen2vl.load(mllm::load(model_path));
auto inputs = qwen2vl_tokenizer.convertMessage({.prompt = prompt_text, .img_file_path = image_path});

for (auto& step : qwen2vl.chat(inputs)) { 
  std::wcout << qwen2vl_tokenizer.detokenize(step.cur_token_id) << std::flush; 
}
```

more examples can be found in [examples](./examples/)

### Custom Models  

MLLM offers a highly Pythonic API to simplify model implementation for users. For instance, consider the following concise `VisionMLP` implementation:  

```c++
class VisionMlp final : public nn::Module {
  int32_t dim_;
  int32_t hidden_dim_;

  nn::QuickGELU act_;
  nn::Linear fc_1_;
  nn::Linear fc_2_;

 public:
  VisionMlp() = default;

  inline VisionMlp(const std::string& name, const Qwen2VLConfig& cfg) : nn::Module(name) {
    dim_ = cfg.visual_embed_dim;
    hidden_dim_ = cfg.visual_embed_dim * cfg.visual_mlp_ratio;

    fc_1_ = reg<nn::Linear>("fc1", dim_, hidden_dim_, true, cfg.linear_impl_type);
    fc_2_ = reg<nn::Linear>("fc2", hidden_dim_, dim_, true, cfg.linear_impl_type);
    act_ = reg<nn::QuickGELU>("act");
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    return {fc_2_(act_(fc_1_(inputs[0])))};
  }
};
```

To utilize this `VisionMLP`, instantiate and execute it as follows:  

```c++
auto mlp = VisionMlp(the_mlp_name, your_cfg);
print(mlp);
auto out = mlp(Tensor::random({1, 1024, 1024}));
print(out);
```

### Model Tracing  

MLLM enables **computational graph extraction** through its `trace` API, converting dynamic model execution into an optimized static representation. This is essential for model optimization, serialization, and deployment. For example:  

```c++
auto ir = mllm::ir::trace(mlp, Tensor::random({1, 1024, 1024})); 
print(ir);
```

## Installation

### Arm Android

```shell
pip install -r requirements.txt
python task.py tasks/build_android.yaml
```

If you need to compile QNN Backends, please install the QNN SDK first. For instructions on setting up the QNN environment, please refer to [QNN README](mllm/backends/qnn/README.md).

Once the environment is configured, you can compile MLLM using the following command.

```shell
pip install -r requirements.txt
python task.py tasks/build_android_qnn.yaml
```

### X86 PC

```shell
pip install -r requirements.txt
python task.py tasks/build_x86.yaml
```

### OSX(Apple Silicon)

```shell
pip install -r requirements-mini.txt
python task.py tasks/build_osx_apple_silicon_accelerate.yaml
```

### Use Docker

The MLLM Team provides Dockerfile to help you get started quickly, and we recommend using Docker images. In the `./docker/` folder, we provide images for arm (cross-compile to arm, host: x86) and qnn (cross-compile to arm, host: x86). Both ARM and QNN images support compilation of X86 Backends.

```shell
git clone https://github.com/UbiquitousLearning/mllm.git
cd mllm/docker
docker build -t mllm_arm -f Dockerfile.arm .
docker run -it --cap-add=SYS_ADMIN --network=host --cap-add=SYS_PTRACE --shm-size=4G --security-opt seccomp=unconfined --security-opt apparmor=unconfined --name mllm_arm_dev mllm_arm bash
```

Important Notes:

1. Dockerfile.arm includes NDK downloads. By using this image, you agree to NDK's additional terms.
2. QNN SDK contains proprietary licensing terms. We don't bundle it in Dockerfile.qnn - please configure QNN SDK manually.

The details of how to use Dockerfile can be found in [Easy Setup with Docker and DevContainer for MLLM](docker/README.md)

### Building the C++ SDK

You can build the SDK using the following commands:

```shell
pip install -r requirements.txt
python task.py tasks/build_sdk_<platform>.yaml
# Example for macOS on Apple Silicon:
python task.py tasks/build_sdk_osx_apple_silicon.yaml
```

By default, the SDK installs to the root directory of the `mllm` project. To customize the installation path, modify the `-DCMAKE_INSTALL_PREFIX` option in the task YAML file.

Once installed, integrate this library into your CMake project using `find_package(mllm)`. Below is a minimal working example:

```cmake
cmake_minimum_required(VERSION 3.21)
project(fancy_algorithm VERSION 1.0.0 LANGUAGES CXX C ASM)

# Set C++20 standard and enable compile commands export
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find mllm library
find_package(mllm REQUIRED)

add_executable(fancy_algorithm main.cpp)

# Link against Mllm runtime and CPU backend targets
target_link_libraries(fancy_algorithm PRIVATE mllm::MllmRT mllm::MllmCPUBackend)
```

### Building the Documentation

You can build the documentation using the following commands:

```shell
pip install -r docs/requirements.txt
python task.py tasks/build_doc.yaml
```

If you need to generate Doxygen documentation, please ensure that Doxygen is installed on your system. Then, set the `enable_doxygen` option to `true` in the `tasks/build_doc.yaml` configuration file. Running `python task.py tasks/build_doc.yaml` afterward will generate the C++ API documentation.

## Model Convert

mllm provides a set of model converters to convert models from other popular model formats to MLLM. Before you start, please make sure you have installed the **pymllm** !!!.

```shell
bash ./scripts/install_pymllm.sh
```

**future:**

Once PyPI approves the creation of the mllm organization, we will publish it there. Afterwards, you can use the command below to install it in the future.

```shell
pip install pymllm
```

After installing pymllm, you can use the following command to convert the model:

```shell
mllm-convertor --input_path <your_model> --output_path <your_output_model> --cfg_path <your_config> --pipeline <builtin_pipeline>
```

For more usage instructions, please refer to `mllm-convertor --help`.

## Tools

### mllm-params-inspector

Parameter inspection tool for viewing model file parameters.

Usage:

```bash
./mllm-params-inspector -i /path/to/model.mllm [-iv v1|v2] [-n param_name]
```

Parameters:

```text
-i, --input: Model file path
-iv, --input_version: Model file version (v1 or v2), default is v1
-n, --name: Specific parameter name, only display information for that parameter
-h, --help: Show help information
```

Examples:

```bash
# View all parameter information in the model file
./mllm-params-inspector -i /path/to/model.mllm

# View specific parameter information
./mllm-params-inspector -i /path/to/model.mllm -n transformer.h.0.attn.c_attn.weight

# View v2 version model file
./mllm-params-inspector -i /path/to/model.mllm -iv v2
```

## Join us & Contribute

The mllm community continues to grow, with developers already contributing PRs. We extend our sincere gratitude to every follower and contributor. We've pinned our roadmap in the [Issues section](https://github.com/UbiquitousLearning/mllm/issues), where you can find features you'd like to contribute to and notify the mllm community by submitting issues.

mllm was born from the fertile soil of academic exploration, dedicated to the pure pursuit of multimodal large models. However, a gap always exists between academic "innovation" and industrial "stability." We candidly acknowledge our current shortcomings and firmly believe in the power of community collaboration. Whether you are a researcher, engineer, or tech enthusiast, every Issue, every PR, and every suggestion or word of encouragement helps build a more solid foundation for mllm. Let us join hands to transform this project, born in academia, into a true bridge connecting academia and industry.
Whether you specialize in hardware adaptation, model optimization, tool development, or documentation and ecosystem promotion, you will find opportunities to contribute here. We especially look forward to working with you to enhance X86 CPU and Ascend NPU support, explore cutting-edge quantization and pruning algorithms, refine a more user-friendly toolchain, and enrich our out-of-the-box model library. Through community collaboration, you can not only work closely with the core team and directly influence the project's evolution, but also leave your innovative mark on the frontier of on-device AI, enabling mllm to run on an ever-growing number of devices.

mllm exists because of the community and grows stronger through you. We look forward to walking alongside you to create a new era of on-device AI.

## Acknowledgements

mllm reuses many low-level kernel implementation from [ggml](https://github.com/ggerganov/ggml) on ARM CPU.
It also utilizes [stb](https://github.com/nothings/stb) and [wenet](https://github.com/wenet-e2e/wenet) for
pre-processing images and audios. mllm also has benefitted from following projects: [llama.cpp](https://github.com/ggerganov/llama.cpp) and [MNN](https://github.com/alibaba/MNN).

## License

### Overall Project License

This project is licensed under the terms of the MIT License. Please see the [LICENSE](./LICENSE) file in the root directory for the full text of the MIT License.

### Apache 2.0 Licensed Components

Certain component([wenet](https://github.com/wenet-e2e/wenet)) of this project is licensed under the Apache License 2.0.
These component is clearly identified in their respective subdirectories along with a copy of the Apache License 2.0.
For the full text of the Apache License 2.0, please refer to the [LICENSE-APACHE](./third_party/wenet_audio/LICENSE) file located in the relevant subdirectories.

## Citation

```bibtex
@article{xu2025fast,
  title={Fast On-device LLM Inference with NPUs},
  author={Xu, Daliang and Zhang, Hao and Yang, Liming and Liu, Ruiqi and Huang, Gang and Xu, Mengwei and Liu, Xuanzhe},
  booktitle={International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS)},
  year={2025}
}
@misc{yi2023mllm,
  title = {mllm: fast and lightweight multimodal LLM inference engine for mobile and edge devices},
  author = {Rongjie Yi and Xiang Li and Zhenyan Lu and Hao Zhang and Daliang Xu and Liming Yang and Weikai Xie and Chenghua Wang and Xuanzhe Liu and Mengwei Xu},
  year = {2023},
  publisher = {mllm Team},
  url = {https://github.com/UbiquitousLearning/mllm}
}
```

## Star History

<a href="https://www.star-history.com/#UbiquitousLearning/mllm&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=UbiquitousLearning/mllm&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=UbiquitousLearning/mllm&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=UbiquitousLearning/mllm&type=Date" />
 </picture>
</a>
