# Introduction
[![Documentation](https://img.shields.io/badge/view-docs-blue)]()


mllm is an open-source large model edge inference framework, mainly supporting CPU inference on Android devices, and also supports accelerating inference through GPU and NPU methods. Currently, mllm supports the inference of large models like llama, fuyu, vit, imagebind, clip, etc.

The architecture diagram of mllm is as follows:

(图)

mllm provides a series of [example programs](examples), including the implementation of llama, clip, fuyu, vit, imagebind, and more using the mllm framework. In addition, mllm also offers [an example app](android) for Android devices, where you can upload models to your phone via adb to experience the effects of different models' inference on mllm.


# News




#  Key Features

Currently mllm support models:

* S: Support and test well on mobile devices.

* U: It is under construction or still has some bugs.

* N: Not support yet

|           | Normal | FP16 |
| --------- | ------ | ---- |
| llama     | S      |      |
| fuyu      | S      |      |
| vit       | S      |      |
| ImageBind | U      |      |
| Clip      | U      |      |

## Performance
放性能指标？

## Roadmap

# Documentation

See the [documentation]() here for more information

# Contribution

Read the [contribution]() before you contribute.

# Acknowledgments

mllm refers to the following projects:

* [MNN](https://github.com/alibaba/MNN)
* [llama.cpp](https://github.com/ggerganov/llama.cpp)
* [transformers](https://github.com/huggingface/transformers)
* [clip](https://github.com/openai/CLIP)
* [ImageBind](https://github.com/facebookresearch/ImageBind)

# Citation

```

```

# License

Apache 2.0