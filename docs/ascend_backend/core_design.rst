Ascend Backend Design
=====================

总览
----

Ascend Backend 将 mLLM 的 Tensor、Op、Module 和 Dispatcher 体系接入 Ascend NPU。
当前已经实现为可支撑 Qwen Ascend 推理的后端路径，包含：

- Backend 注册、Allocator 注册和 Dispatcher 注册。
- Ascend 设备内存池和 Tensor 数据搬运。
- ATB/ACL 公共封装。
- 单算子 eager 执行路径。
- ATB graph 构建和执行路径。
- 面向 Qwen decoder 的 graph plugin。

架构
----

架构图如下：

.. code-block:: text

   ┌──────────────────────────────────────────────────────────────────────────────┐
   │                                MLLM Framework                                │
   │                                                                              │
   │  ┌──────────────┐   ┌──────────────┐   ┌────────────────┐   ┌─────────────┐ │
   │  │ nn::Module   │   │ nn::Layer/Op │   │ Tensor/Storage │   │ Task System │ │
   │  └──────┬───────┘   └──────┬───────┘   └───────┬────────┘   └──────┬──────┘ │
   │         │                  │                   │                   │        │
   │         └──────────────────┴───────────────────┴───────────────────┘        │
   │                                      │                                       │
   │                                      ▼                                       │
   │                  Context / DispatcherManager / MemoryManager                 │
   └──────────────────────────────────────┬───────────────────────────────────────┘
                                          │
                                          ▼
   ┌──────────────────────────────────────────────────────────────────────────────┐
   │                         Ascend Backend Infrastructure                         │
   │                                                                              │
   │  ┌────────────────────────────────────────────────────────────────────────┐  │
   │  │                         AscendBackend                                  │  │
   │  │   backend registration / op factory registry / allocator binding       │  │
   │  └───────────────┬───────────────────────────────┬────────────────────────┘  │
   │                  │                               │                           │
   │                  ▼                               ▼                           │
   │  ┌──────────────────────────────┐   ┌────────────────────────────────────┐  │
   │  │       Eager Op Path           │   │            Graph Path              │  │
   │  │  AscendDispatcher             │   │  AscendGraphBuilder               │  │
   │  │      reshape/setup/forward    │   │      tensor-name graph construction│  │
   │  │  Ascend Ops                   │   │  AscendGraphExecutor              │  │
   │  │      Linear/RMSNorm/RoPE/...  │   │      setup/workspace/execute/sync  │  │
   │  └───────────────┬──────────────┘   └────────────────┬───────────────────┘  │
   │                  │                                   │                      │
   │                  │                                   ▼                      │
   │                  │                    ┌──────────────────────────────────┐  │
   │                  │                    │        Graph Plugin Ops          │  │
   │                  │                    │  AttentionWithKVCache            │  │
   │                  │                    │  LinearW8A8 / DynamicLinearW8A8  │  │
   │                  │                    │  CausalMask / Round / Clamp      │  │
   │                  │                    └────────────────┬─────────────────┘  │
   │                  │                                     │                    │
   │                  └───────────────────┬─────────────────┘                    │
   │                                      ▼                                      │
   │  ┌────────────────────────────────────────────────────────────────────────┐  │
   │  │                    Shared Ascend Services                              │  │
   │  │  AscendCommon: ATB context / ACL stream / tensor desc / checks         │  │
   │  │  AscendAllocator + AscendMemoryManager + AscendMemoryPool              │  │
   │  └────────────────────────────────────┬───────────────────────────────────┘  │
   └───────────────────────────────────────┼──────────────────────────────────────┘
                                           │
                                           ▼
   ┌──────────────────────────────────────────────────────────────────────────────┐
   │                              Ascend Runtime                                  │
   │                                                                              │
   │  ┌──────────────────┐   ┌──────────────────┐   ┌─────────────────────────┐  │
   │  │ ATB Operations   │   │ ACL / ACLNN APIs │   │ CANN Toolkit Runtime    │  │
   │  └────────┬─────────┘   └────────┬─────────┘   └────────────┬────────────┘  │
   │           └───────────────────────┴──────────────────────────┘               │
   │                                      │                                       │
   │                                      ▼                                       │
   │                         Ascend NPU / Device Memory                           │
   └──────────────────────────────────────────────────────────────────────────────┘

初始化流程
----------

``initAscendBackend()`` 是 Ascend Backend 的统一入口：

1. 创建 Ascend 设备内存池。
2. 注册 ``AscendBackend``。
3. 将 ``AscendAllocator`` 注册到 mLLM ``MemoryManager``。
4. 创建并注册 ``AscendDispatcher``。

初始化后，``model.to(kAscend)`` 会让模型中的算子实例化为 Ascend 后端实现。

核心组件
--------

AscendBackend
~~~~~~~~~~~~~

``AscendBackend`` 负责注册 Ascend 支持的 op factory，并在构造时枚举设备信息。
当前注册的算子覆盖 Qwen Ascend 推理所需的主要路径。

AscendDispatcher
~~~~~~~~~~~~~~~~

``AscendDispatcher`` 接收 ``kExecuteOp`` 和 ``kExecuteModule`` 任务。

- 对 ``kExecuteOp``，按 ``reshape -> setup -> forward`` 生命周期执行单个算子。
- 对 ``kExecuteModule``，当前直接调用模块 ``forward``，后续可进一步接入模块级 graph execute。

AscendCommon
~~~~~~~~~~~~

``AscendCommon`` 封装 ATB/ACL 常用能力：

- ``getGlobalAtbContext()``：懒初始化 ATB context、ACL stream，并设置 execute stream。
- ``getGlobalAtbStream()`` / ``syncGlobalAtbStream()``：统一 stream 获取和同步。
- ``fillAtbTensorDesc()`` / ``fillAtbTensor()``：将 mLLM Tensor 转为 ATB Tensor。
- ``MLLM_ACL_CHECK`` / ``MLLM_ATB_CHECK``：统一错误检查。
- 测试辅助函数：Ascend tensor 准备、回拷、数值验证和计时。

内存管理
--------

Ascend 内存管理由 ``AscendMemoryManager`` 和 ``AscendMemoryPool`` 组成。

``AscendMemoryManager`` 按 device 维护内存池，并提供：

- ``createMemoryPool()``
- ``allocateBlock()``
- ``freeBlock()``
- ``getBlockPtr()``
- ``printStats()``

``AscendMemoryPool`` 使用大块设备内存作为池，维护 used/free block，并统计分配、释放、
复用和内存浪费情况。Tensor 分配通过 Ascend allocator 接入 mLLM 的统一内存管理。

单算子执行路径
--------------

单算子路径仍然是 Ascend Backend 的基础执行方式：

1. mLLM 创建 op task。
2. ``AscendDispatcher`` 调用 op 的 ``reshape``。
3. ``setup`` 阶段创建或准备 ATB/ACL 资源。
4. ``forward`` 阶段构造 ``VariantPack`` 并执行 ATB/ACL op。
5. 必要时同步 global ATB stream。

该路径适合单算子验证、graph 回退和不适合进入 graph 的辅助算子。

支持算子
--------

.. list-table::
   :header-rows: 1

   * - 算子
     - 主要实现
     - 说明
   * - Add / Sub / Mul
     - ATB Elewise
     - 支持基础 elementwise 路径。
   * - X2X
     - ACL memcpy / dtype/device 路径
     - 负责 CPU/Ascend 数据搬运。
   * - SiLU
     - ATB Activation
     - Qwen MLP 使用。
   * - Linear FP16
     - ATB Linear
     - 常规 FP16 linear。
   * - Linear W8A8
     - ATB Linear + quant artifacts
     - 静态 W8A8 主路径。
   * - RMSNorm
     - ATB RmsNorm
     - Qwen decoder 使用。
   * - MatMul
     - ATB Linear/MatMul
     - attention 相关基础能力。
   * - Softmax
     - ATB Softmax
     - attention 相关基础能力。
   * - Concat / Slice
     - ATB
     - Tensor 组合与切片。
   * - Transpose
     - ATB Transpose
     - attention layout 转换。
   * - Embedding
     - ACLNN Embedding
     - 310B 上替代 ATB Gather 路径。
   * - Gather
     - ACLNN/辅助实现
     - 数据索引路径。
   * - RoPE
     - ATB RoPE
     - 输入约定为 Q/K/cos/sin/pos。
   * - Fill / Copy
     - ACL/辅助实现
     - Tensor 初始化与复制。
   * - CausalMask
     - host/Ascend 辅助逻辑
     - prefill/decode mask 生成。

Graph 执行
----------

Ascend graph 路径由 ``AscendGraphBuilder`` 和 ``AscendGraphExecutor`` 组成。

``AscendGraphBuilder`` 是 ATB ``GraphOpBuilder`` 的轻量封装，使用 tensor name API
构建 graph：

- ``beginGraph()``：声明 graph 名称、输入名和输出名。
- ``addOperation()``：按输入/输出 tensor name 添加 ATB op。
- ``reshape()``：在 graph 内创建 reshape view，不触发设备拷贝。
- ``build()``：返回 ATB graph operation。

``AscendGraphExecutor`` 持有 graph operation，并负责：

- 将 mLLM Tensor 填充为 ATB Tensor。
- 调用 ``Setup`` 获取 workspace 大小。
- 按需分配或扩容 workspace。
- 调用 ``Execute``。
- 同步 global ATB stream。
- 在打开 profiling 时统计 setup、alloc、execute、sync 和 total 时间。

Graph profiling 由环境变量控制：

.. code-block:: bash

   export MLLM_PROFILE_ASCEND_GRAPH=1
   export MLLM_PROFILE_ASCEND_GRAPH_EVERY=20

Graph Plugin
------------

为了支持 Qwen decoder graph，Ascend Backend 提供若干 ATB plugin operation。

``AscendAttentionWithKVCachePluginOperation``
  将 attention、KV cache 更新、prefill/decode 子图和可选 setup bucket 封装为 graph op。
  该 plugin 避免在 Qwen graph 外拆散 attention 流程，是 decoder graph 的核心节点。

``AscendLinearW8A8PluginOperation``
  封装静态 W8A8 linear pipeline：
  ``x_fp16 -> muls(inv_scale_x) -> round -> clamp -> cast int8 -> ATB Linear W8A8``。
  这是 Qwen Ascend W8A8 graph 的主路径。

``AscendDynamicLinearW8A8PluginOperation``
  封装 per-token dynamic activation quantization。该路径保留为实验和调试能力，
  默认不接入 Qwen Ascend graph。

``AscendCausalMaskPluginOperation`` / ``AscendCausalMaskTensorPluginOperation``
  用于 graph 内 mask 生成和 attention mask tensor 准备。

``AscendRoundPluginOperation`` / ``AscendClampPluginOperation``
  将 ACLNN round/clamp 包装成 ATB plugin，供 W8A8 量化链路复用。

当前限制
--------

- Ascend graph 主要服务 Qwen decoder，不是通用 graph compiler。
- Dynamic W8A8 eager 路径需要显式环境变量打开，仅用于调试。
- Graph plugin 和 attention 相关路径仍有继续优化空间，后续可结合实际 profiling 结果推进。
