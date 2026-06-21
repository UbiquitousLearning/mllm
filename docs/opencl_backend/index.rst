OpenCL Backend
==============

Overview
--------
The OpenCL backend in MLLM is designed to enable Large Language Model (LLM) inference on a wide range of devices that support the OpenCL standard, such as mobile GPUs (Adreno, Mali) and desktop GPUs. This document outlines the current preliminary design and implementation details.

.. note::
   This is an initial implementation. Significant optimizations in memory management and inference speed are planned for future updates.

Design
------

Memory Management
~~~~~~~~~~~~~~~~~
The memory management is handled by the ``OpenCLAllocator`` class.

*   **Mechanism**: It implements a basic memory pool mechanism to reduce the overhead of frequent memory allocation and deallocation.
*   **Implementation**:
    *   It maintains a ``memory_pool_`` (a map of buffer sizes to ``cl_mem`` objects).
    *   When ``alloc`` is called, it checks the pool for an available buffer of suitable size. If found, it reuses it; otherwise, it creates a new ``cl_mem`` buffer using ``clCreateBuffer``.
    *   When ``free`` is called, the buffer is not immediately released to the OpenCL runtime but returned to the pool for future reuse.
    *   Thread safety is managed via ``std::mutex``.

Model Implementation
~~~~~~~~~~~~~~~~~~~~
The model implementation (e.g., Llama) follows the standard MLLM module structure but is adapted for the OpenCL backend.

*   **Device Type**: Tensors and Modules are initialized or moved to the ``mllm::kOpenCL`` device.
*   **KV Cache**: Uses ``nn::StaticCache`` configured for ``kOpenCL`` to store key-value pairs on the GPU memory.
*   **Data Flow**: Input tensors (like token sequences) are moved to the OpenCL device before inference. Intermediate computations (Attention, MLP) happen on the device.

Usage
-----
To use the OpenCL backend, the application must initialize it and move the model and inputs to the appropriate device.

.. code-block:: cpp

    // Initialize the backend
    mllm::initOpenCLBackend();

    // Load model and move to OpenCL device
    auto llama = mllm::models::llama::LlamaForCausalLM("", llama_cfg);
    llama.load(param);
    llama.to(mllm::kOpenCL);

    // Prepare inputs
    inputs["sequence"] = inputs["sequence"].to(mllm::kOpenCL);

Current Limitations & Future Work
---------------------------------

As a preliminary implementation, there are several areas identified for improvement:

1.  **Memory Management**:
    *   The current pooling strategy is basic.
    *   **Optimization Needed**: More advanced allocators (e.g., sub-allocators, better fragmentation handling) are needed to reduce memory footprint and allocation overhead.

2.  **Inference Speed**:
    *   The current performance is functional but not fully optimized.
    *   **Optimization Needed**: Kernel tuning (work-group sizes, memory access patterns), operator fusion, and minimizing host-device synchronization are required to improve throughput and latency.

3.  **Operator Support**:
    *   Currently supports a subset of operators required for models like Llama. Support for more operators and architectures will be added.
