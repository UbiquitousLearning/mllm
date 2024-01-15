## Introduction
[![Documentation](https://img.shields.io/badge/view-docs-blue)]()
[![Actions Status](https://github.com/UbiquitousLearning/mllm/workflows/Tests/badge.svg)](https://github.com/UbiquitousLearning/mllm/actions)

mllm is a fast, lightweight, dependency-free multimodal LLM inference engine for mobile and edge devices.

Currently, mllm is mainly optimized for Android devices.

## Android Demo


##  Key Features

Currently mllm support models:

* S: Support and test well on mobile devices.

* U: It is under construction or still has some bugs.

* N: Not support yet

|                | FP32 | INT4 |
|----------------|------|------|
| Llama 7B       | S    | S    |
| Alpaca 7B      | S    | S    |
| TinyLlama 1.1B | S    | S    |
| Persimmon 8B   | S    | S    |
| fuyu 8B        | S    | S    |
| ViT            | S    | S    |
| Clip           | U    | U    |
| ImageBind      | U    | U    |

##  Quick Start

### Get the Code

```bash
git clone https://github.com/UbiquitousLearning/mllm
cd mllm
```
### Check prerequisites 

Building mllm requires following tools:
- CMake >= 3.18
- OpenMP Libs.
- Android NDK Toolchains >= 26

### Download models

```bash
mkdir models && cd models
wget .... >> llama-2-7b-chat-q4_k.mllm
```
### Run on Linux

- Build
    ```bash
    cd scripts
    ./build.sh
    ```
- Run
    ```bash
    cd ../bin
    ./main_llama
    ```
  
### Run on Android

- Build
    ```bash
    cd scripts
    ./build_android.sh
    ```
- Run
    ```bash
    ./llama_exe.sh
    ```
## Convert models
You can download models from [here](), or you can convert a pytorch/safetensor model to mllm model by yourself.

```bash
cd tools/convertor
pip install -r ./requirements.txt

# for one file pytorch model
python convert.py --input_model=model.pth --output_model=model.mllm --type=torch

# for multi-file pytorch model
python convert.py --input_model=pytorch_model.bin.index.json --output_model=model.mllm --type=torch

# for one file safetensor model
python convert.py --input_model=model.bin --output_model=model.mllm --type=safetensor

# for multi-file safetensor model
python convert.py --input_model=model.safetensors.index.json --output_model=model.mllm --type=safetensor
``` 

## Convert Vocabulary
You can convert vocabulary to mllm vocabulary as followed.
```bash
cd tools/convertor
python vocab.py tokenizer.model vocab.mllm
```

## Quantize models
You can quantize mllm model to int4 model by yourself. 
mllm only support two quantize modes: Q4_0 and Q4_K.
```bash
cd bin
./quantize model.mllm model_q4_0.mllm Q4_K
```


## Roadmap

See the [roadmap](https://mllm-landing.vercel.app/roadmap/roadmap/) here for more information

## Documentation

See the [documentation]() here for more information

## Contribution

Read the [contribution](https://mllm-landing.vercel.app/contributing/contributing/) before you contribute.

## Acknowledgments

mllm reuses many low-level kernel implementation from [ggml](https://github.com/ggerganov/ggml) on ARM CPU. 
It also utilizes [stb](https://github.com/nothings/stb) and [wenet](https://github.com/wenet-e2e/wenet) for handling images and audios.
Additionally, mllm refers to the following projects:

[//]: # (* [ggml]&#40;https://github.com/ggerganov/ggml&#41;)
[//]: # (* [stb]&#40;https://github.com/nothings/stb&#41;)
[//]: # (* [wenet]&#40;https://github.com/wenet-e2e/wenet&#41;)

* [llama.cpp](https://github.com/ggerganov/llama.cpp)
* [MNN](https://github.com/alibaba/MNN)
* [transformers](https://github.com/huggingface/transformers)
* [clip](https://github.com/openai/CLIP)
* [ImageBind](https://github.com/facebookresearch/ImageBind)

## License

MIT License
