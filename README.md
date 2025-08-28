## Latest News

Here's a concise, scannable version optimized for GitHub README (prioritizing impact + brevity):

---

## Key Features

1. **Pythonic eager execution** – Rapid model development  
2. **Unified hardware support** – Arm CPU, OpenCL GPU, QNN NPU  
3. **Advanced optimizations** – Quantization, pruning, speculative execution  
4. **NPU-ready IR** – Seamless integration with NPU frameworks  
5. **Deployment toolkit** – SDK + CLI inference tool  

## Tested Devices

| Device | OS | CPU | GPU | NPU |
| :---: | :---: | :---: | :---: | :---: |
| PC-X86-w/oAVX512  | Ubuntu 22.04  | ![build-pending](https://img.shields.io/badge/build-pending-gray) | - | - |
| Xiaomi14-8Elite   | Android 15    | ![build-passing](https://img.shields.io/badge/build-passing-green) | - | ![build-pending](https://img.shields.io/badge/build-pending-gray) |
| OnePlus13-8Elite  | Android 15    | ![build-passing](https://img.shields.io/badge/build-passing-green) | - | ![build-pending](https://img.shields.io/badge/build-pending-gray) |
| MacMini-M4        | MacOS 15.5    | ![build-passing](https://img.shields.io/badge/build-passing-green) | - | - |

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

qwen2vl.streamGenerate(inputs,
                        {
                            {"do_sample", mllm::AnyValue(false)},
                            {"max_length", mllm::AnyValue(qwen2vl_cfg.max_cache_length)},
                        },
                        [&](int64_t token_id) {
                          auto str = qwen2vl_tokenizer.detokenize(token_id);
                          std::wcout << str << std::flush;
                        });
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
python task.py tasks/build_osx_apple_silicon.yaml
```

if you want to use apple's accelerate library, you can use the following command.

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

## Tools

## Join us & Contribute

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
