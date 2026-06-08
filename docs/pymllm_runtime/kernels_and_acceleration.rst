pymllm Kernels and Acceleration
===============================

总览
----------------------------------------

``pymllm`` 的性能由几类加速组件分工撑起来，它们解决的不是同一个问题：

- **FlashInfer**：paged KV cache attention。
- **CUDA Graph**：减少 decode 阶段的 CPU launch overhead。
- **Triton**：W8A8 的 per-token activation quantization。
- **CUTLASS**：W8A8 的 INT8 Tensor Core GEMM。
- **mllm-kernel**：基于 TVM-FFI / torch extension 的 JIT kernel 工具包。

简单说：attention backend 管 KV cache attention，CUDA Graph 管重复 decode step 的 launch
开销，Triton 和 CUTLASS 管量化 linear 的核心计算，``mllm-kernel`` 则为项目内自定义的
CUDA / C++ kernel 提供封装、缓存和工具。

mllm-kernel
----------------------------------------

``mllm-kernel`` 是 mllm 里的高性能 kernel 包，Python 侧目前主要是：

- ``mllm_kernel.cuda.jit``：CUDA JIT kernel wrapper。
- ``mllm_kernel.cpu.jit``：CPU JIT kernel wrapper。
- ``mllm_kernel.jit_utils``：JIT 编译、缓存、注册表和工具函数。

一个 CUDA JIT kernel 的典型结构：

.. code-block:: text

   Python wrapper
       -> @jit(...)
       -> include CUDA/C++ source
       -> export TVM-FFI typed function
       -> 首次使用时编译
       -> 之后复用缓存的 shared library

默认 JIT 缓存目录：

.. code-block:: text

   ~/.cache/mllm_kernel/

``mllm-kernel`` 的 JIT 思路和 SGLang 的 ``jit_kernel`` 关系更近：都强调轻量 JIT、运行时选模板
实例、避开大型 AOT torch extension 那种动辄几分钟的编译。同时 SGLang 的 ``sgl-kernel`` AOT
kernel 仍是重要参考，对照量化 GEMM 的语义和性能时尤其有用。

TVM-FFI JIT 路径
----------------------------------------

``mllm_kernel.jit_utils.jit`` 这个 decorator 把一个 Python 函数包成按需编译的 kernel 调用，
负责：

- 根据 tensor device 推断 CPU / CUDA 目标。
- 把 Python 参数转成 C++ template 参数。
- 拼 C++/CUDA source 和 export wrapper。
- 调 TVM-FFI 编译并加载 shared library。
- 把编译结果缓存到 ``~/.cache/mllm_kernel``。

这种方式适合小而明确的自定义 kernel，比如：

- ``create_kv_indices``：构造 FlashInfer 的 KV index metadata。
- ``store_cache``：把 K/V 写进 KVPool。
- ``gptq_marlin_repack``：Marlin 权重 layout 转换。
- ``gptq_marlin_gemm``：W4A16 Marlin GEMM。

W8A8 的 CUTLASS kernel 目前是个例外，用 ``torch.utils.cpp_extension.load`` 编译——CUTLASS 的
模板和 include 体系太重，现阶段优先保证它能在 Jetson SM87 上稳定编过。

FlashInfer Attention
----------------------------------------

``pymllm.layers.attention.flashinfer_backend.FlashInferAttnBackend`` 封装了 FlashInfer 的
paged KV cache attention，负责：

- 为 prefill 和 decode 准备 ``kv_indptr``、``kv_indices``、``kv_last_page_len`` 等 metadata。
- 管理全局 workspace buffer。
- 根据有没有 sliding window 选 wrapper dispatch。
- decode 时按 GQA group size 和 KV dtype 决定走不走 tensor core 路径。
- 给 CUDA Graph capture / replay 提供专用的 metadata 初始化接口。

prefill 和 decode 用不同 wrapper：

.. code-block:: text

   prefill / extend
       BatchPrefillWithPagedKVCacheWrapper
       BatchPrefillWithRaggedKVCacheWrapper

   decode
       BatchDecodeWithPagedKVCacheWrapper

attention backend 只管 attention 计算和 metadata，不碰请求调度和 KV slot 生命周期。KV slot
的分配、释放、prefix cache 命中是 scheduler / model runner 那边的事。

CUDA Graph
----------------------------------------

``pymllm.executor.cuda_graph_runner.CudaGraphRunner`` 负责 decode step 的 CUDA Graph capture
和 replay，目的就是把小 batch decode 里的 CPU launch overhead 压下去。

初始化时按一组离散 batch size 捕获 graph：

.. code-block:: text

   [1, 2, 4, 8, 12, 16, 24, 32, ...]

每个 captured graph 复用预分配好的输入 buffer：

- ``input_ids``
- ``req_pool_indices``
- ``seq_lens``
- ``out_cache_loc``
- ``positions``
- ``mrope_position_deltas``

replay 时真实 batch 会 padding 到最近的 captured batch size，attention backend 走专用的
``init_forward_metadata_replay_cuda_graph`` 路径，而不是普通的动态 metadata 初始化。

CUDA Graph 只覆盖 decode 主路径。调试模型、查 attention metadata 或定位 shape 问题时，可以
用 ``--server.disable_cuda_graph`` 临时关掉。

W4A16 Marlin
----------------------------------------

W4A16 复用 Marlin kernel。checkpoint 权重先以 ``weight_packed`` 和 ``weight_scale`` 加载，
再在 post-load 阶段转成 Marlin 的 runtime layout。

关键 kernel：

- ``mllm_kernel.cuda.jit.gptq_marlin_repack``
- ``mllm_kernel.cuda.jit.gptq_marlin``

执行约束：

- SM80+
- output partition 能被 64 整除
- input partition 能被 128 整除
- group size 主路径目前是 32

这条路径适合 AWQ / W4A16 这类权重量化模型，activation 保持 FP16/BF16。

W8A8 Triton + CUTLASS
----------------------------------------

W8A8 有两个核心 kernel：

1. ``pymllm.quantization.kernels.int8_activation_triton.per_token_quant_int8``
2. ``mllm_kernel.cuda.jit.int8_scaled_mm_cutlass.int8_scaled_mm``

运行时链路：

.. code-block:: text

   [M, K] fp16/bf16 activation
       -> Triton per-token absmax + round + int8 cast
       -> [M, K] int8 + [M, 1] fp32 scale
       -> CUTLASS int8 GEMM（per-row / per-col scale）
       -> [M, N] fp16/bf16 output

CUTLASS kernel 要求 ``mat_b`` 是 ``[K, N]`` column-major，所以 W8A8 scheme 会在
``process_weights_after_loading`` 里把 checkpoint 的 ``[N, K]`` INT8 weight 转成对应布局。

CUTLASS 头文件默认用 ``flashinfer`` bundled 的那份；要换版本就设 ``CUTLASS_HOME``。如果头文件
找不到，W8A8 初始化会直接失败。生产环境建议在镜像里把 CUTLASS 来源固定下来，免得不同节点用上
不同版本的头文件。

GDN decode kernel
----------------------------------------

Qwen3.5 这类 hybrid 模型可能带 GDN / linear attention 层。``pymllm`` 给它们预留了：

- ``pymllm.layers.attention.gdn_backend``
- ``pymllm.layers.attention.hybrid_backend``
- ``mllm_kernel.cuda.jit.gdn_decode``
- ``MambaRadixCache`` / GDN state cache 相关结构

本文档重点还是 Qwen3 / Qwen3-VL 主路径，GDN 相关路径以具体模型和测试结果为准。

调试与观测
----------------------------------------

几条常用检查命令：

.. code-block:: bash

   python3 -m mllm_kernel show-env
   python3 -m mllm_kernel show-config
   python3 -m pymllm show-config

首次运行特别慢的时候，要分清楚时间花在哪：模型权重加载、FlashInfer / CUDA context 初始化、
CUTLASS JIT 编译、CUDA Graph capture，还是真正的 prefill / decode。别把首次 JIT 或 kernel
初始化的开销当成稳态瓶颈。
