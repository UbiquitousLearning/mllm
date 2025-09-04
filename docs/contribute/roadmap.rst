Roadmap & Help wanted!
======================

August - October 2025
---------------------

P0
~~~

Arm Kernel support
^^^^^^^^^^^^^^^^^^

- Arm I8-Gemm and I8-Gemv Kernels. (Co-works with bitspack)
- Arm U1-7 Group Quantized Embedding Kernels. (Co-works with bitspack)
- More KleidiAI Kernels (SME Supports)
- Optimizing MLLM-BLAS-SGEMV and MLLM-BLAS-SGEMM Kernels, for Shapes in LLM Scenarios.
- Full coverage of the correctness of current Arm operators
- Kernel Selector Table (Tune)

X86 Backend support
^^^^^^^^^^^^^^^^^^^^

- Highway kernels for dbg purpose

QNN Backend support
^^^^^^^^^^^^^^^^^^^^

- Migration from mllmv1 to mllm v2
- QNN Kernel Benchmarks

Quantization
^^^^^^^^^^^^^^

- Model Convertor & Quantizer
- Shared weight Embedding(For tie-embedding scenario).

P1
~~~

pymllm API
^^^^^^^^^^^

- C++ Tensor and Python Tensor lifetime conflict in some test cases.
