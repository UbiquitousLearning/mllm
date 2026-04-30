pymllm Kernels and Acceleration
===============================

总览
----------------------------------------

``pymllm`` 的性能路径由多类加速组件共同组成：

- FlashInfer：paged KV cache attention。
- CUDA Graph：decode 阶段减少 CPU launch overhead。
- Triton：W8A8 per-token activation quantization。
- CUTLASS：W8A8 INT8 Tensor Core GEMM。
- ``mllm-kernel``：基于 TVM-FFI / torch extension 的 JIT kernel 工具包。

这些组件不是彼此替代关系，而是在不同层次承担职责。attention backend 解决 KV cache
attention；CUDA Graph 解决重复 decode step 的 launch overhead；Triton 和 CUTLASS 解决量化
linear 的核心计算；``mllm-kernel`` 为项目内自定义 CUDA/C++ kernel 提供封装、缓存和工具。

mllm-kernel
----------------------------------------

``mllm-kernel`` 是 mllm 项目中的高性能 kernel 包。当前 Python 侧主要包含：

- ``mllm_kernel.cuda.jit``：CUDA JIT kernel wrapper。
- ``mllm_kernel.cpu.jit``：CPU JIT kernel wrapper。
- ``mllm_kernel.jit_utils``：JIT 编译、缓存、注册表和工具函数。

CUDA JIT kernel 的典型结构是：

.. code-block:: text

   Python wrapper
       -> @jit(...)
       -> include CUDA/C++ source
       -> export TVM-FFI typed function
       -> compile on first use
       -> reuse cached shared library

默认 JIT 缓存目录为：

.. code-block:: text

   ~/.cache/mllm_kernel/

``mllm-kernel`` 的 JIT 路径与 SGLang 的 ``jit_kernel`` 设计关系更直接：二者都强调轻量
JIT、运行时选择模板实例、避免大型 AOT torch extension 带来的长编译周期。与此同时，SGLang
的 ``sgl-kernel`` AOT kernel 仍然是重要参考，尤其适合对照量化 GEMM 的语义和性能。

TVM-FFI JIT 路径
----------------------------------------

``mllm_kernel.jit_utils.jit`` decorator 会将 Python 函数包装成一个按需编译的 kernel 调用。
它负责：

- 根据 tensor device 推断 CPU/CUDA 目标。
- 将 Python 参数转换为 C++ template 参数。
- 拼接 C++/CUDA source 和 export wrapper。
- 调用 TVM-FFI 编译并加载 shared library。
- 将编译结果缓存到 ``~/.cache/mllm_kernel``。

这种方式适合小而明确的自定义 kernel，例如：

- ``create_kv_indices``：构造 FlashInfer KV index metadata。
- ``store_cache``：将 K/V 写入 KVPool。
- ``gptq_marlin_repack``：Marlin weight layout 转换。
- ``gptq_marlin_gemm``：W4A16 Marlin GEMM。

W8A8 CUTLASS kernel 当前使用 ``torch.utils.cpp_extension.load`` 编译。这是因为 CUTLASS
模板和 include 体系较重，当前以稳定通过 Jetson SM87 编译为优先。

FlashInfer Attention
----------------------------------------

``pymllm.layers.attention.flashinfer_backend.FlashInferAttnBackend`` 封装 FlashInfer 的 paged
KV cache attention。它负责：

- 为 prefill 和 decode 准备 ``kv_indptr``、``kv_indices``、``kv_last_page_len`` 等 metadata。
- 管理全局 workspace buffer。
- 根据是否存在 sliding window 选择 wrapper dispatch。
- 在 decode 中根据 GQA group size 和 KV dtype 决定是否使用 tensor core 路径。
- 为 CUDA Graph capture / replay 提供专用 metadata 初始化接口。

prefill 和 decode 使用不同 wrapper：

.. code-block:: text

   prefill / extend
       BatchPrefillWithPagedKVCacheWrapper
       BatchPrefillWithRaggedKVCacheWrapper

   decode
       BatchDecodeWithPagedKVCacheWrapper

attention backend 只负责 attention 计算和 metadata，不负责请求调度和 KV slot 生命周期。KV slot
的分配、释放和 prefix cache 命中由 scheduler / model runner 侧完成。

CUDA Graph
----------------------------------------

``pymllm.executor.cuda_graph_runner.CudaGraphRunner`` 用于 decode step 的 CUDA Graph capture
和 replay。它的目标是减少小 batch decode 中 CPU launch overhead。

初始化阶段会按一组离散 batch size 捕获 graph：

.. code-block:: text

   [1, 2, 4, 8, 12, 16, 24, 32, ...]

每个 captured graph 复用预分配输入 buffer：

- ``input_ids``
- ``req_pool_indices``
- ``seq_lens``
- ``out_cache_loc``
- ``positions``
- ``mrope_position_deltas``

replay 时，真实 batch 会被 padding 到最近的 captured batch size。attention backend 会走专用
``init_forward_metadata_replay_cuda_graph`` 路径，避免使用普通动态 metadata 初始化。

CUDA Graph 只覆盖 decode 主路径。调试模型、调试 attention metadata 或定位 shape 问题时，可以
使用 ``--server.disable_cuda_graph`` 暂时关闭。

W4A16 Marlin
----------------------------------------

W4A16 路径复用 Marlin kernel。checkpoint 权重先以 ``weight_packed`` 和 ``weight_scale``
加载，然后在 post-load 阶段转换为 Marlin runtime layout。

关键 kernel：

- ``mllm_kernel.cuda.jit.gptq_marlin_repack``
- ``mllm_kernel.cuda.jit.gptq_marlin``

执行约束包括：

- SM80+
- output partition 可被 64 整除
- input partition 可被 128 整除
- group size 当前主路径为 32

这种路径适合 AWQ / W4A16 类权重量化模型，activation 保持 FP16/BF16。

W8A8 Triton + CUTLASS
----------------------------------------

W8A8 路径包含两个核心 kernel：

1. ``pymllm.quantization.kernels.int8_activation_triton.per_token_quant_int8``
2. ``mllm_kernel.cuda.jit.int8_scaled_mm_cutlass.int8_scaled_mm``

运行时链路：

.. code-block:: text

   [M, K] fp16/bf16 activation
       -> Triton per-token absmax + round + int8 cast
       -> [M, K] int8 + [M, 1] fp32 scale
       -> CUTLASS int8 GEMM with per-row/per-col scales
       -> [M, N] fp16/bf16 output

CUTLASS kernel 要求 ``mat_b`` 为 ``[K, N]`` column-major，因此 W8A8 scheme 会在
``process_weights_after_loading`` 中把 checkpoint 的 ``[N, K]`` INT8 weight 转成对应布局。

当前 CUTLASS include 查找顺序为：

1. ``CUTLASS_HOME/include``
2. ``flashinfer`` bundled CUTLASS
3. 系统 include 目录

如果找不到 CUTLASS 头文件，W8A8 初始化会失败。生产环境建议在镜像中固定 CUTLASS 来源，避免
不同节点使用不同版本头文件。

GDN decode kernel
----------------------------------------

Qwen3.5 等 hybrid 模型可能包含 GDN / linear attention 层。``pymllm`` 为这类模型保留了：

- ``pymllm.layers.attention.gdn_backend``
- ``pymllm.layers.attention.hybrid_backend``
- ``mllm_kernel.cuda.jit.gdn_decode``
- ``MambaRadixCache`` / GDN state cache 相关结构

当前文档重点覆盖 Qwen3 / Qwen3-VL 主路径。GDN 相关路径仍应以具体模型和测试结果为准。

调试与观测
----------------------------------------

常用检查命令：

.. code-block:: bash

   python3 -m mllm_kernel show-env
   python3 -m mllm_kernel show-config
   python3 -m pymllm show-config

当首次运行时间异常长时，应区分：

- 模型权重加载时间。
- FlashInfer / CUDA context 初始化时间。
- CUTLASS JIT 编译时间。
- CUDA Graph capture 时间。
- 实际 prefill/decode 时间。
