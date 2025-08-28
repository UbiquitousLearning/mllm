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

Features
--------

Usage Examples
--------------

Tested Devices
--------------

+---------------------+----------------+----------+----------+----------+
| Device              | OS             | CPU      | GPU      | NPU      |
+=====================+================+==========+==========+==========+
| PC-X86-w/oAVX512    | Ubuntu 22.04   | ✅ PASS  | x        | x        |
+---------------------+----------------+----------+----------+----------+
| Xiaomi14-8Elite     | Android 15     | ✅ PASS  | x        | ✅ PASS  |
+---------------------+----------------+----------+----------+----------+
| OnePlus13-8Elite    | Android 15     | PENDING  | x        | PENDING  |
+---------------------+----------------+----------+----------+----------+
| MacMini-M4          | MacOS 15.5     | ✅ PASS  | x        | x        |
+---------------------+----------------+----------+----------+----------+

Installation
-------------


Arm Android
~~~~~~~~~~~

.. code-block:: shell

   pip install -r requirements.txt
   python task.py tasks/build_android.yaml

If you need to compile QNN Backends, please install the QNN SDK first. For instructions on setting up the QNN environment, please refer to `QNN README <mllm/backends/qnn/README.md>`_.

Once the environment is configured, you can compile MLLM using the following command:

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

   brew install libomp
   pip install -r requirements-mini.txt
   python task.py tasks/build_osx_apple_silicon.yaml

Use Docker
~~~~~~~~~~~

The MLLM Team provides Dockerfile to help you get started quickly, and we recommend using Docker images. In the ``./docker/`` folder, we provide images for arm (cross-compile to arm, host: x86) and qnn (cross-compile to arm, host: x86). Both ARM and QNN images support compilation of X86 Backends.

.. code-block:: bash

   git clone https://github.com/UbiquitousLearning/mllm.git
   cd mllm/docker
   docker build -t mllm_arm -f Dockerfile.arm .
   docker run -it --cap-add=SYS_ADMIN --network=host --cap-add=SYS_PTRACE --shm-size=4G --security-opt seccomp=unconfined --security-opt apparmor=unconfined --name mllm_arm_dev mllm_arm bash

.. important::

   1. Dockerfile.arm includes NDK downloads. By using this image, you agree to NDK's additional terms.
   2. QNN SDK contains proprietary licensing terms. We don't bundle it in Dockerfile.qnn - please configure QNN SDK manually.

The details of how to use Dockerfile can be found in `Easy Setup with Docker and DevContainer for MLLM <docker/README.md>`_

OpenCL Backend
~~~~~~~~~~~~~~~~~~~

Quick Starts
-------------

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
----------------
.. code-block:: shell

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

   arch/index

.. toctree::
   :maxdepth: 2

   compile/index

.. toctree::
   :maxdepth: 2

   quantization/index

.. toctree::
   :maxdepth: 2

   cpu_backend/index

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
   :caption: C++ API

   CppAPI/library_root
