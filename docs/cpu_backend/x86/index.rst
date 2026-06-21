CPU X86 Backend
===============

Overview
--------

The MLLM X86 CPU backend provides optimized inference on x86 processors using Highway's cross-platform SIMD abstractions.
`Google Highway <https://github.com/google/highway>`_ is a C++ library that delivers portable SIMD
(Single Instruction Multiple Data) operations, allowing high-performance neural network computations
while maintaining compatibility across various x86 microarchitectures.

Key Features:

- **Portable SIMD Operations**: Highway abstracts platform-specific instructions, supporting multiple
  x86 targets (SSE4, AVX2, AVX-512) with the same codebase
- **Runtime Dispatch**: Automatically selects the best available instruction set for the target CPU
- **Quantized Inference**: Optimized kernels for Q4 and other quantization formats
- **Cross-Platform Compatibility**: Maintains backward compatibility from older CPUs (SSE4) to modern
  processors (AVX-512)

This backend leverages Highway's cross-platform SIMD abstractions to achieve high performance on modern
x86 processors while maintaining portability across different CPU models and microarchitectures.


Running MLLM on X86 Architecture
--------------------------------

This guide explains how to build and run the **mllm** inference framework on x86 processors.
The X86 backend uses Highway for portable SIMD operations,
enabling efficient vectorized computations across different x86 CPU models.


Prerequisites
~~~~~~~~~~~~~

Before building the x86 backend, install the required build toolchain and Python runtime.

Install system dependencies:

.. code-block:: bash

   sudo apt update
   sudo apt install build-essential cmake ninja-build python3 python3-pip

Install Python dependencies:

.. code-block:: bash

   pip install -r requirements.txt


Recommended Versions (Used in Testing):

- **Operating System**: Linux (Ubuntu 22.04 LTS)
- **CPU**: x86-64 processor with at least SSE4 support
  (better performance with AVX2 or AVX-512)
- **C/C++ compiler**: GCC/G++ (validated with GCC 11.4.0)
- **Python**: validated with Python 3.10
- **Build tools**: CMake + Ninja
  (validated with CMake 3.31.2 and Ninja 1.11.1)

.. note::

   The default toolchain on Ubuntu 22.04 LTS is sufficient to build the project.
   Ubuntu 24.04 LTS has not been validated yet, but it is expected to work as well.


Step 1: Build the X86 Backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before building, ensure you have completed the environment setup described above.

Run the provided build task to compile MLLM with X86-optimized kernels:

.. code-block:: bash

   python task.py tasks/build_x86.yaml

This command configures and builds the project with Highway SIMD operations enabled
for x86 architecture.


Step 2: Acquire Model Assets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~


1. Download the original model from Hugging-Face (or any other reputable source).

   Typical files you need:

   * ``config.json``
   * ``tokenizer.json`` / ``tokenizer.model``
   * PyTorch / Safetensors checkpoints (``.bin``, ``.safetensors``)

2. Place everything under a single directory, e.g. ``~/models/Qwen3-0.6B``.

.. note::
   Models obtained from hosting platforms such as Hugging Face or ModelScope (via ``git clone`` or their official CLI) are already organized in a single directory that contains ``config.json``, ``tokenizer.json``, ``tokenizer.model``, checkpoint shards, etc.

   You can download Qwen3-0.6B from ModelScope with the following command:

   .. code-block:: bash

      git clone https://www.modelscope.cn/Qwen/Qwen3-0.6B.git


3. Download pre-converted models from **our HuggingFace organization** (recommended on x86):

Due to current compatibility issues with the mllm-converter on x86 architecture, we recommend downloading pre-converted quantized models from our HuggingFace organization `mllmTeam <https://huggingface.co/UbiquitousLearning>`_:

Example command:

.. code-block:: bash

   wget https://huggingface.co/mllmTeam/qwen-3-0.6b-mllm/blob/main/qwen-3-0.6b-q4_k.mllm

.. note::

   If you prefer to convert models yourself, Please refer to :doc:`How to Support a New LLM: Step-by-Step <../../quick_start/how_to_model>`, specifically **Step 2**, to download and convert the model.




Step 3: Run Inference
~~~~~~~~~~~~~~~~~~~~~

Once you have the model assets, run inference using the compiled binary.

Command Parameters:

- ``-m``: Path to the quantized MLLM model file
- ``-mv``: Model version (``v1`` or ``v2``)
- ``-t``: Path to the tokenizer file
- ``-c``: Path to the model configuration file

Example Command:

.. code-block:: bash

   /path/to/build/bin/mllm-qwen3-runner \
     -m /path/to/model/qwen-3-0.6b-q4_k.mllm \
     -mv v1 \
     -t /path/to/tokenizer/tokenizer.json \
     -c /path/to/config/config_0.6B_w4a32_kai.json

.. caution::
   You can use ``mllm/examples/qwen3/config_0.6B_w4a32_kai.json`` as the configuration file for Qwen3-0.6B quantized with Q4_K. 
   But remember to change the ``linear_impl_type`` to ``Default`` because we are using the default linear implementation in the x86 backend.

Performance
~~~~~~~~~~~~~~~~~~~~~

After inference completes, the program automatically outputs a performance summary.

The following metrics were measured on an **Intel Core i9-14900K**
with a **Qwen3-0.6B** model (Q4 quantization). Example Output:

.. code-block:: text

      ============== Performance Summary ===============
      Total time          :  667525.00 μs
      Prefill time        :  123295.00 μs (194.66 tokens/s)
      Decode time         :  544230.00 μs ( 49.61 tokens/s)
      TTFT                :  123443.00 μs
      Prefill tokens      :         24
      Decode steps        :         27
      Avg decode time     :   20156.67 μs/token
      ==================================================

- **Prefill throughput**: 194.66 tokens/s —
  The model processes input tokens at this rate during the prefill phase
- **Decode throughput**: 49.61 tokens/s —
  The model generates output tokens at this rate during decoding
- **Time-to-first-token (TTFT)**: 123.4 ms —
  Time from request submission to receiving the first generated token
- **Average decode latency**: 20.16 ms/token —
  Average time to generate each subsequent token

.. note::

   The x86 PC platform is not currently the primary optimization focus for the MLLM framework,
   so inference speeds are relatively slower. Ongoing optimizations are planned for future releases.


Factors Affecting Performance
^^^^^^^^^^^^^^^^^^^^^

Performance may vary depending on:

- CPU model and specifications
- System load and thermal conditions
- Model size and quantization method
- Input prompt length and output length
