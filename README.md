## Introduction
[![Documentation](https://img.shields.io/badge/view-docs-blue)]()


mllm is an open-source large model edge inference framework, mainly supporting CPU inference on Android devices, and also supports accelerating inference through GPU and NPU methods. Currently, mllm supports the inference of large models like llama, fuyu, vit, imagebind, clip, etc.

[//]: # (The architecture diagram of mllm is as follows:)

[//]: # (&#40;图&#41;)

mllm provides a series of [example programs](examples), including the implementation of llama, clip, fuyu, vit, imagebind, and more using the mllm framework. In addition, mllm also offers [an example app](android) for Android devices, where you can upload models to your phone via adb to experience the effects of different models' inference on mllm.


### News




##  Key Features

Currently mllm support models:

* S: Support and test well on mobile devices.

* U: It is under construction or still has some bugs.

* N: Not support yet

|           | Normal | INT4 |
|-----------| ------ |------|
| llama     | S      | S    |
| apacha    | S      | S    |
| persimmon | S      | S    |
| fuyu      | S      | S    |
| vit       | S      | S    |
| Clip      | U      | U    |
| ImageBind | U      | U    |

##  Quick Start

Here are the end-to-end binary build and model conversion steps for the LLaMA-7B model.

### Get the Code

```bash
git clone https://github.com/UbiquitousLearning/mllm
cd mllm
```

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

## Roadmap

See the [roadmap]() here for more information

## Documentation

See the [documentation]() here for more information

## Contribution

Read the [contribution]() before you contribute.

## Acknowledgments

mllm refers to the following projects:

* [MNN](https://github.com/alibaba/MNN)
* [llama.cpp](https://github.com/ggerganov/llama.cpp)
* [transformers](https://github.com/huggingface/transformers)
* [clip](https://github.com/openai/CLIP)
* [ImageBind](https://github.com/facebookresearch/ImageBind)

## Citation

```

```

## License

Apache 2.0