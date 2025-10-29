.. raw:: html

   <h1 align="center">
   MLLM
   </h1>

   <h3 align="center">
   <span style="color:#2563eb">M</span>obile x <span style="color:#8b5cf6">M</span>ultimodal
   </h3>
 
   <p align="center">
   Fast and lightweight LLM inference engine for mobile and edge devices
   </p>

   <p align="center">
   | Arm CPU | X86 CPU | Qualcomm NPU(QNN) |
   </p>

MLLM is a lightweight, on-device inference engine optimized for multi-modal models. It supports diverse hardware platforms including ARM CPUs, x86 architectures, and Qualcomm NPUs. Featuring a Torch-like API, MLLM enables developers to rapidly deploy AI algorithms directly on edge devices—ideal for future AI PCs, smart assistants, drones, satellites, and embodied intelligence applications.

Latest News
-----------

Key Features
------------

1. **Pythonic eager execution** - Rapid model development
2. **Unified hardware support** - Arm CPU, OpenCL GPU, QNN NPU
3. **Advanced optimizations** - Quantization, pruning, speculative execution
4. **NPU-ready IR** - Seamless integration with NPU frameworks
5. **Deployment toolkit** - SDK + CLI inference tool

Tested Devices
--------------

+---------------------+----------------+------------------------+----------+------------------------+
| Device              | OS             | CPU                    | GPU      | NPU                    |
+=====================+================+========================+==========+========================+
| PC-X86-w/oAVX512    | Ubuntu 22.04   | |build-pending|        | -        | -                      |
+---------------------+----------------+------------------------+----------+------------------------+
| Xiaomi14-8Elite     | Android 15     | |build-passing|        | -        | |build-pending|        |
+---------------------+----------------+------------------------+----------+------------------------+
| OnePlus13-8Elite    | Android 15     | |build-passing|        | -        | |build-pending|        |
+---------------------+----------------+------------------------+----------+------------------------+
| MacMini-M4          | MacOS 15.5     | |build-passing|        | -        | -                      |
+---------------------+----------------+------------------------+----------+------------------------+

.. |build-pending| image:: https://img.shields.io/badge/build-pending-gray
   :alt: build-pending
.. |build-passing| image:: https://img.shields.io/badge/build-passing-green
   :alt: build-passing

Quick Starts
-------------

Serving LLMs with mllm-cli
~~~~~~~~~~~~~~~~~~~~~~~~~~

We have developed a C SDK wrapper for the MLLM C++ SDK to enable seamless integration with Golang. Leveraging this wrapper, we've built the mllm-cli command-line tool in Golang, which is about to be released soon.

Inference with VLM using C++ API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following example demonstrates how to perform inference on a multimodal vision-language model (VLM), specifically Qwen2-VL, using the mllm framework's C++ API. The process includes loading the model configuration, initializing the tokenizer, loading pretrained weights, processing image-text inputs, and performing streaming text generation.

.. code-block:: c++

   auto qwen2vl_cfg        = Qwen2VLConfig(config_path);
   auto qwen2vl_tokenizer  = Qwen2VLTokenizer(tokenizer_path);
   auto qwen2vl            = Qwen2VLForCausalLM(qwen2vl_cfg);

   qwen2vl.load(mllm::load(model_path));
   auto inputs = qwen2vl_tokenizer.convertMessage({.prompt = prompt_text, .img_file_path = image_path});

   for (auto& step : qwen2vl.chat(inputs)) { 
      std::wcout << qwen2vl_tokenizer.detokenize(step.cur_token_id) << std::flush; 
   }

more examples can be found in `examples <./examples/>`_

Custom Models
~~~~~~~~~~~~~

MLLM offers a highly Pythonic API to simplify model implementation for users. For instance, consider the following concise ``VisionMLP`` implementation:

.. code-block:: c++

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

To utilize this ``VisionMLP``, instantiate and execute it as follows:

.. code-block:: c++

   auto mlp = VisionMlp(the_mlp_name, your_cfg);
   print(mlp);
   auto out = mlp(Tensor::random({1, 1024, 1024}));
   print(out);

Model Tracing
~~~~~~~~~~~~~

MLLM enables **computational graph extraction** through its ``trace`` API, converting dynamic model execution into an optimized static representation. This is essential for model optimization, serialization, and deployment. For example:

.. code-block:: c++

   auto ir = mllm::ir::trace(mlp, Tensor::random({1, 1024, 1024})); 
   print(ir);

Installation
-------------


Arm Android
~~~~~~~~~~~

.. code-block:: shell

   pip install -r requirements.txt
   python task.py tasks/build_android.yaml

If you need to compile QNN Backends, please install the QNN SDK first. For instructions on setting up the QNN environment, please refer to `QNN README <mllm/backends/qnn/README.md>`_.

Once the environment is configured, you can compile MLLM using the following command.

.. code-block:: shell

   pip install -r requirements.txt
   python task.py tasks/build_android_qnn.yaml

X86 PC
~~~~~~~~~~~

.. code-block:: shell

   pip install -r requirements.txt
   python task.py tasks/build_x86.yaml

OSX (Apple Silicon)
~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

   pip install -r requirements-mini.txt
   python task.py tasks/build_osx_apple_silicon.yaml

if you want to use apple's accelerate library, you can use the following command.

.. code-block:: shell

   pip install -r requirements-mini.txt
   python task.py tasks/build_osx_apple_silicon_accelerate.yaml

Use Docker
~~~~~~~~~~~

The MLLM Team provides Dockerfile to help you get started quickly, and we recommend using Docker images. In the ``./docker/`` folder, we provide images for arm (cross-compile to arm, host: x86) and qnn (cross-compile to arm, host: x86). Both ARM and QNN images support compilation of X86 Backends.

.. code-block:: shell

   git clone https://github.com/UbiquitousLearning/mllm.git
   cd mllm/docker
   docker build -t mllm_arm -f Dockerfile.arm .
   docker run -it --cap-add=SYS_ADMIN --network=host --cap-add=SYS_PTRACE --shm-size=4G --security-opt seccomp=unconfined --security-opt apparmor=unconfined --name mllm_arm_dev mllm_arm bash

Important Notes:

1. Dockerfile.arm includes NDK downloads. By using this image, you agree to NDK's additional terms.
2. QNN SDK contains proprietary licensing terms. We don't bundle it in Dockerfile.qnn - please configure QNN SDK manually.

The details of how to use Dockerfile can be found in `Easy Setup with Docker and DevContainer for MLLM <docker/README.md>`_

Building the C++ SDK
~~~~~~~~~~~~~~~~~~~~

You can build the SDK using the following commands:

.. code-block:: shell

   pip install -r requirements.txt
   python task.py tasks/build_sdk_<platform>.yaml
   # Example for macOS on Apple Silicon:
   python task.py tasks/build_sdk_osx_apple_silicon.yaml

By default, the SDK installs to the root directory of the ``mllm`` project. To customize the installation path, modify the ``-DCMAKE_INSTALL_PREFIX`` option in the task YAML file.

Once installed, integrate this library into your CMake project using ``find_package(mllm)``. Below is a minimal working example:

.. code-block:: cmake

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

Building the Documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~

You can build the documentation using the following commands:

.. code-block:: shell

   pip install -r docs/requirements.txt
   python task.py tasks/build_doc.yaml

If you need to generate Doxygen documentation, please ensure that Doxygen is installed on your system. Then, set the ``enable_doxygen`` option to ``true`` in the ``tasks/build_doc.yaml`` configuration file. Running ``python task.py tasks/build_doc.yaml`` afterward will generate the C++ API documentation.

Model Convert
---------------

mllm provides a set of model converters to convert models from other popular model formats to MLLM. Before you start, please make sure you have installed the **pymllm** !!!

.. code-block:: shell

   bash ./scripts/install_pymllm.sh

**future:**

Once PyPI approves the creation of the mllm organization, we will publish it there. Afterwards, you can use the command below to install it in the future.

.. code-block:: shell

   pip install pymllm

After installing pymllm, you can use the following command to convert the model:

.. code-block:: shell

   mllm-convertor --input_path <your_model> --output_path <your_output_model> --cfg_path <your_config> --pipeline <builtin_pipeline>

For more usage instructions, please refer to ``mllm-convertor --help``.

Tools
-----

Join us & Contribute
--------------------

Acknowledgements
----------------

mllm reuses many low-level kernel implementation from `ggml <https://github.com/ggerganov/ggml>`_ on ARM CPU.
It also utilizes `stb <https://github.com/nothings/stb>`_ and `wenet <https://github.com/wenet-e2e/wenet>`_ for
pre-processing images and audios. mllm also has benefitted from following projects: `llama.cpp <https://github.com/ggerganov/llama.cpp>`_ 
and `MNN <https://github.com/alibaba/MNN>`_.

License
--------

Overall Project License
~~~~~~~~~~~~~~~~~~~~~~~

This project is licensed under the terms of the MIT License. Please see the `LICENSE <LICENSE>`_ file in the root
directory for the full text of the MIT License.

Apache 2.0 Licensed Components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Certain component(`wenet <https://github.com/wenet-e2e/wenet>`_) of this project is licensed under the Apache License 2.0.
These component is clearly identified in their respective subdirectories along with a copy of the Apache License 2.0.
For the full text of the Apache License 2.0, please refer to the `LICENSE-APACHE <third_party/wenet_audio/LICENSE>`_ file
located in the relevant subdirectories.

Citation
--------

.. code-block:: bibtex

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


Documents
----------

.. toctree::
   :maxdepth: 2

   quick_start/index

.. toctree::
   :maxdepth: 2

   service/index

.. toctree::
   :maxdepth: 2

   arch/index

.. toctree::
   :maxdepth: 2

   compile/index

.. toctree::
   :maxdepth: 2

   quantization/index

.. toctree::
   :maxdepth: 2

   cache/index

.. toctree::
   :maxdepth: 2

   cpu_backend/index

.. toctree::
   :maxdepth: 2

   qnn_backend/index

.. toctree::
   :maxdepth: 2

   api/index

.. toctree::
   :maxdepth: 2

   contribute/index

.. toctree::
   :maxdepth: 2

   talks/index

.. toctree::
   :maxdepth: 2

   algorithms/index

.. toctree::
   :maxdepth: 2

   qa/index

.. toctree::
   :maxdepth: 2
   :caption: Pymllm API

   autoapi/pymllm/index

.. toctree::
   :maxdepth: 2
   :caption: C++ API

   CppAPI/library_root
