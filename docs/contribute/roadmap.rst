Roadmap & Help wanted!
======================

August - October 2025
---------------------

P0
~~~

Model Supports
^^^^^^^^^^^^^^^^

Transform models supported by v1 to v2.

- Qwen3 Series
- Qwen2 Series
- Llama3 Series

Performance Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^

- Inplace kernels for all backends

  - MulbyConst
  - AddFrom
  - Activation Functions

    - Sigmoid
    - GeLU
    - QuickGeLU
    - SiLU
    - ReLU, ReLU2
  - LayerNorm
  - RMSNorm
  - Softmax

- Fused Kernels

  - Softmax + TopK
  - Matmul + RoPE
  - Softmax + Causal Mask

- Well optimized models (modeling_xxx_fast version)

  - Using Fused Kernels
  - Using inplace operators
  - Manually free tensors before its lifetime ends

- Kernel Selector Table (Tune)

  - GEMV and GEMM kernels tile size
  - Thread numbers

Arm Kernel support
^^^^^^^^^^^^^^^^^^

- Arm I8-Gemm and I8-Gemv Kernels. (Co-works with bitspack)
- Arm U1-7 Group Quantized Embedding Kernels. (Co-works with bitspack)
- More KleidiAI Kernels (SME Supports)
- Optimizing MLLM-BLAS-SGEMV and MLLM-BLAS-SGEMM Kernels, for Shapes in LLM Scenarios.
- Full coverage of the correctness of current Arm operators
- MXFP4 Linear Kernels

X86 Backend support
^^^^^^^^^^^^^^^^^^^^

- Highway kernels for dbg purpose

QNN Backend support
^^^^^^^^^^^^^^^^^^^^

- Migration from mllmv1 to mllm v2
- QNN Kernel Benchmarks

CANN Backend support
^^^^^^^^^^^^^^^^^^^^

- CANN Kernels

Quantization
^^^^^^^^^^^^^^

- Model Convertor & Quantizer
- Shared weight Embedding(For tie-embedding scenario).

Applications
^^^^^^^^^^^^^

- Multi-turn Chat
- mllm-cli's modelscope integration

P1
~~~

pymllm API
^^^^^^^^^^^

- C++ Tensor and Python Tensor lifetime conflict in some test cases.


Tests
^^^^^^

- PPL Tests
