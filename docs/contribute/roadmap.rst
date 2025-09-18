Roadmap & Help wanted!
======================

August - October 2025
---------------------

P0
~~~

Benchmarks
^^^^^^^^^^^^

Benchmark MLLM, llama.cpp, mnn.

- W4A32 & PPL

  - Qwen3
  - Qwen2.5VL

Model Supports
^^^^^^^^^^^^^^^^

Transform models supported by v1 to v2.

- Qwen3 Series
- Qwen2 Series
- Llama3 Series
- TinyLlama

Performance Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^

Using 1. Manually memory planning 2. Fused kernels 3. Inplace Operators etc. To archive high performance in eager mode.

- Inplace kernels for all backends

  - MulbyConst
  - AddFrom
  - Activation Functions

    - Sigmoid
    - GeLU
    - QuickGeLU
    - ✅ SiLU
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

- !!! Kernel Selector Table (Tune)

  - GEMV and GEMM kernels tile size
  - Thread numbers

- Quantized KVCache
- MllmBlas used in Qwen2.5-VL is slow, use ggml's matmul(llama file) in the feature. 

Arm Kernel support
^^^^^^^^^^^^^^^^^^

- ✅ MLLM-BLAS fp32 GEMM Kernels (transpose_a=False, transpose_b=True) [@chenghua]
- Element-wise Kernels has slightly performance issues
- ✅ Arm I8-Gemm and I8-Gemv Kernels. (Co-works with bitspack) [@chenghua]
- Arm U1-7 Group Quantized Embedding Kernels. (Co-works with bitspack)
- More KleidiAI Kernels (SME Supports)
- Optimizing MLLM-BLAS-SGEMV and MLLM-BLAS-SGEMM Kernels, for Shapes in LLM Scenarios.
- Full coverage of the correctness of current Arm operators
- MXFP4 Linear Kernels
- ✅ Paged Attention Kernels (Attentions as one of outputs)

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

Applications & Productions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

Long term 2025
---------------------

P1
~~~

FFI ABI
^^^^^^^^^^^

- One C_api for all languages(Using tvm-ffi, thanks @tianqi)

ARM PMU Tools Workflow
^^^^^^^^^^^^^^^^^^^^^^^^

- A Kernel Benchmark workflow that using PMU in ARM Arch.
- Software Pipeline & multi-issue will be benefited.

