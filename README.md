[![Website](https://img.shields.io/badge/website-visit-green)](https://ubiquitouslearning.github.io/mllm_website/)
[![Documentation](https://img.shields.io/badge/view-docs-blue)](https://ubiquitouslearning.github.io/mllm_website/introduction/getstarted/)
[![Actions Status](https://github.com/UbiquitousLearning/mllm/workflows/Tests/badge.svg)](https://github.com/UbiquitousLearning/mllm/actions)

**mllm** is a fast and lightweight <ins>multimodal LLM</ins> inference engine for mobile and edge devices.

- Plain C/C++ implementation without dependencies
- Optimized for multimodal LLMs like fuyu-8B
- Supported: ARM NEON and x86 AVX2
- 4-bit and 6-bit integer quantization

Wait.. why on-device multimodal LLM? - It's a key building block for [intelligent personal agent](https://arxiv.org/pdf/2401.05459.pdf), text-based image searching/retrieval, screen VQA, and many more exciting mobile apps, without giving away your private data (chat history, screenshots, taken photos, etc).

## Recent update
- [:fire::fire:Comming soon] Supporting Qualcomm NPU: [>1000 tokens/second prefilling!](https://arxiv.org/pdf/2407.05858v1)
- [2024 May 29] Support new model: Mistral V0.2 7B https://github.com/UbiquitousLearning/mllm/pull/83
- [2024 May 4] Support new model: QWen V1.5 0.5B https://github.com/UbiquitousLearning/mllm/pull/79
- [2024 April 9] Support new model: Gemma 2B https://github.com/UbiquitousLearning/mllm/pull/75


### Contents
- [Android Demo](#android-demo)
- [Support models](#support-models)
- [Quick Start](#quick-start)
    - [Get the Code](#get-the-code)
    - [Check prerequisites](#check-prerequisites)
    - [Run for Android](#run-for-android)
    - [Run for Linux](#run-for-linux)
- [Customization](#customization)
    - [Convert models](#convert-models)
    - [Convert vocabulary](#convert-vocabulary)
    - [Quantize models](#quantize-models)
- [Roadmap](#roadmap)
- [Documentation](#documentation)
- [Contribution](#contribution)
- [Acknowledgments](#acknowledgments)
- [License](#license)


## Android Demo

<table>
    <tr>
        <td>Demo of LLM chatting</td>
        <td>Demo of image understanding</td>
        <td>Demo of UI screen understanding</td>
    </tr>
    <tr>
        <td> <video src="https://github.com/UbiquitousLearning/mllm/assets/38753457/7a1eb892-8259-41ff-8c97-b773d16fce7f"> </td>
        <td> <video src="https://github.com/UbiquitousLearning/mllm/assets/38753457/32549658-5c74-4ce0-962f-6621c919faad"> </td>
        <td>  <video src="https://github.com/UbiquitousLearning/mllm/assets/38753457/fe234f27-1393-4ee2-84ce-254cee91a27f"> </td>
    </tr>
</table>
            
## Support models

[//]: # (* ✔️ : Support and test well on mobile devices.)

[//]: # ()
[//]: # (* ⚠️ : It is under construction or still has some bugs.)

[//]: # ()
[//]: # (* ❌  : Not support yet.)

|                                                                             | FP32 | INT4 |
|-----------------------------------------------------------------------------|-----|------|
| [LLaMA-1/2 7B](https://github.com/facebookresearch/llama)                   | [✔️](https://huggingface.co/mllmTeam/llama-2-7b-mllm/tree/main)  | [✔️](https://huggingface.co/mllmTeam/llama-2-7b-mllm/tree/main)   |
| [Alpaca 7B](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2)                | [✔️](https://huggingface.co/mllmTeam/chinese-alpaca-7b-mllm/tree/main)  | [✔️](https://huggingface.co/mllmTeam/chinese-alpaca-7b-mllm/tree/main)   |
| [TinyLLaMA 1.1B](https://github.com/jzhang38/TinyLlama)                     | [✔️](https://huggingface.co/mllmTeam/tinyllama-1.1b-mllm/tree/main)  | [✔️](https://huggingface.co/mllmTeam/tinyllama-1.1b-mllm/tree/main)   |
| [Fuyu 8B](https://www.adept.ai/blog/fuyu-8b)                                | [✔️](https://huggingface.co/mllmTeam/fuyu-8b-mllm/tree/main)  | [✔️](https://huggingface.co/mllmTeam/fuyu-8b-mllm/tree/main)   |
| [Vision Transformer](https://github.com/google-research/vision_transformer) | [✔️](https://huggingface.co/mllmTeam/vit-base-patch16-224-mllm/tree/main)  | [✔️](https://huggingface.co/mllmTeam/vit-base-patch16-224-mllm/tree/main)   |
| [CLIP](https://github.com/openai/CLIP)                                      | [✔️](https://huggingface.co/mllmTeam/clip-vit-base-patch32-mllm/tree/main)  | [✔️](https://huggingface.co/mllmTeam/clip-vit-base-patch32-mllm/tree/main)   |
| [ImageBind](https://github.com/facebookresearch/ImageBind) (3 modalities)   | [✔️](https://huggingface.co/mllmTeam/imagebind_huge-mllm/tree/main)  | [✔️](https://huggingface.co/mllmTeam/imagebind_huge-mllm/tree/main)   |
| [LLaVA 7B](https://github.com/haotian-liu/LLaVA)                            | [✔️](https://huggingface.co/mllmTeam/llava-1.5-7b-mllm/tree/main)  | [✔️](https://huggingface.co/mllmTeam/llava-1.5-7b-mllm/tree/main)   |
| [Gemma 2B](https://github.com/google/gemma_pytorch)                         | [✔️](https://huggingface.co/mllmTeam/gemma-2b-mllm/tree/main)  | [✔️](https://huggingface.co/mllmTeam/gemma-2b-mllm/tree/main)   |
| [Qwen 0.5B](https://github.com/QwenLM/Qwen)                         | [✔️](https://huggingface.co/mllmTeam/qwen-1.5-0.5b-mllm/tree/main)  | [✔️](https://huggingface.co/mllmTeam/qwen-1.5-0.5b-mllm/tree/main)   |
| [Mistral 7B](https://github.com/mistralai/mistral-src)                         | [✔️](https://huggingface.co/mllmTeam/mistral-7b-instruct-v0.2-mllm/tree/main)  | [✔️](https://huggingface.co/mllmTeam/mistral-7b-instruct-v0.2-mllm/tree/main)   |

## Quick Start

### Get the Code

```bash
git clone https://github.com/UbiquitousLearning/mllm
cd mllm
```

### Check prerequisites

Building mllm requires following tools:

- gcc(11.4+) / clang (11.0+)
- CMake >= 3.18
- Android NDK Toolchains >= 26

### Try it on Android

#### Build

  ```bash
  export ANDROID_NDK=/path/to/your/ndk
  cd scripts
  ./build_android.sh
  ```

#### Run Fuyu-8B

Download the model from [here](https://huggingface.co/mllmTeam/fuyu-8b-mllm/tree/main/), or using the following instructions

```bash
mkdir ../models && cd ../models
# Download fuyu-8b-q4_k.mllm
wget https://huggingface.co/mllmTeam/fuyu-8b-mllm/resolve/main/fuyu-8b-q4_k.mllm?download=true  -O fuyu-8b-q4_k.mllm
```

Run on an android phone with at least 12GB of memory.

```bash
cd ../script
./run_fuyu.sh
```

Result are as followed:

```
> ./demo_fuyu
[Q] [../assets/bus.png]Generate a coco-style caption.

[A]  A blue bus driving down▁the▁road next▁to a sidewalk.
[Q] [../assets/two_cats.jpg]What's this?

[A]  Two cats laying on▁top▁of a pink blanket.
```

#### Run LLaMA-2-7B

Download model from [here](https://huggingface.co/mllmTeam/llama-2-7b-mllm/tree/main/), or using the following instructions

```bash
mkdir ../models && cd ../models
# Dowload llama-2-7b-chat-q4_k.mllm
wget https://huggingface.co/mllmTeam/llama-2-7b-mllm/resolve/main/llama-2-7b-chat-q4_k.mllm?download=true  -O llama-2-7b-chat-q4_k.mllm
```

Run on an android phone with at least 12GB of memory.

```bash
cd ../script
./run_llama.sh
```

Result are as followed:

```
> ./demo_llama
[Q]  Hello, who are you?
[A]

I am a bot, I'm here to help answer your questions and provide information. I'm just an AI, I don't have personal experiences or feelings like humans do, but I'm here to help you with any questions you may have. Is there something specific you would like to know or talk about?
[Q]  What can you do?
[A]
I can help with a wide range of tasks, including:
  
* Answering questions on various topics, such as science, history, technology, and more.
* Providing information on different subjects, such as culture, entertainment, and lifestyle.
* Helping with language-related tasks, such as translating text or improving grammar and syntax.
* Generating text, such as articles, stories, or even entire books.
* Assisting with coding
[Q]  Please introduce Beijing University of Posts and Telecommunications.
[A]
Beijing University of Posts and Telecommunications (BUPT) is a public research university located in Beijing, China. It was established in 1955 and is one of the top universities in China, with a strong focus on engineering, technology, and telecommunications.
BUPT offers a wide range of undergraduate and graduate programs in fields such as computer science, electrical engineering, software engineering, and communication engineering. The university
```

#### Run ImageBind

Download model from [here](https://huggingface.co/mllmTeam/imagebind_huge-mllm/tree/main), or using the following instructions

```bash
mkdir ../models && cd ../models
# Download imagebind_huge-q4_k.mllm
wget https://huggingface.co/mllmTeam/imagebind_huge-mllm/resolve/main/imagebind_huge-q4_k.mllm?download=true -O imagebind_huge-q4_k.mllm 
```

Run on an android phone with at least 4GB of memory.

```bash
cd ../script
./run_imagebind.sh
```

Result are as followed:
```
> ./demo_imagebind 
vision X text :
0.9985647 0.0013827 0.0000526 
0.0000365 0.9998636 0.0000999 
0.0000115 0.0083149 0.9916736 
vision X audio :
0.8054272 0.1228001 0.0717727 
0.0673458 0.8429284 0.0897258 
0.0021967 0.0015335 0.9962698 
```


### Run for Linux

#### Build
```bash
cd scripts
./build.sh
 ```

#### Run Fuyu-8B

```bash
cd ./bin
./demo_fuyu -m ../models/fuyu-8b-q4_k.mllm -v ../vocab/fuyu_vocab.mllm
 ```

#### Run LLaMA-2-7B

```bash
cd ./bin
./demo_llama -m ../models/llama-2-7b-chat-q4_k.mllm -v ../vocab/llama_vocab.mllm
```


#### Run ImageBind

```bash
cd ./bin
./demo_imagebind -m ../models/imagebind_huge-q4_k.mllm -v ../vocab/clip_vocab.mllm
```


## Customization

### Convert models

You can download models from [here](https://huggingface.co/mllmTeam), or you can convert a pytorch/safetensor model to
mllm model by yourself.

```bash
cd tools/convertor
pip install -r ./requirements.txt

# for one file pytorch model
python converter.py --input_model=model.pth --output_model=model.mllm --type=torch

# for multi-file pytorch model
python converter.py --input_model=pytorch_model.bin.index.json --output_model=model.mllm --type=torch

# for one file safetensor model
python converter.py --input_model=model.bin --output_model=model.mllm --type=safetensor

# for multi-file safetensor model
python converter.py --input_model=model.safetensors.index.json --output_model=model.mllm --type=safetensor
``` 

### Convert vocabulary

You can convert vocabulary to mllm vocabulary as followed.

```bash
cd tools/convertor
python vocab.py --input_file=tokenizer.json --output_file=vocab.mllm --type=Unigram
```

### Quantize models

You can quantize mllm model to int4 model by yourself.
mllm only support two quantize modes: Q4_0 and Q4_K.

```bash
cd bin
./quantize model.mllm model_q4_0.mllm Q4_K
```

## Roadmap

- More backends like QNN
- More models like PandaGPT
- More optimizations like LUT-GEMM
- [More..](https://ubiquitouslearning.github.io/mllm_website/roadmap/roadmap/)

## Documentation

See the [documentation](https://ubiquitouslearning.github.io/mllm_website/introduction/getstarted/) here for more
information

## Contribution

Read the [contribution](https://ubiquitouslearning.github.io/mllm_website/contributing/contributing/) before you
contribute.

## Acknowledgments

mllm reuses many low-level kernel implementation from [ggml](https://github.com/ggerganov/ggml) on ARM CPU.
It also utilizes [stb](https://github.com/nothings/stb) and [wenet](https://github.com/wenet-e2e/wenet) for
pre-processing images and audios.
mllm also has benefitted from following projects: [llama.cpp](https://github.com/ggerganov/llama.cpp)
and [MNN](https://github.com/alibaba/MNN).

## License

### Overall Project License

This project is licensed under the terms of the MIT License. Please see the [LICENSE](LICENSE) file in the root
directory for the full text of the MIT License.

### Apache 2.0 Licensed Components

Certain component([wenet](https://github.com/wenet-e2e/wenet)) of this project is licensed under the Apache License 2.0.
These component is clearly identified in their respective subdirectories along with a copy of the Apache License 2.0.
For the full text of the Apache License 2.0, please refer to the [LICENSE-APACHE](third_party/wenet_audio/LICENSE) file
located in the relevant subdirectories.

