pymllm Runtime Design
=====================

总览
----------------------------------------

``pymllm`` 是 mllm 的 Python serving runtime。它不是传统意义上的 mllm C++
Backend，而是一套围绕 PyTorch/CUDA 生态构建的在线推理服务运行时。当前实现面向
Jetson Orin 等边缘 GPU 设备，重点支持 Qwen3、Qwen3-VL 和 Qwen3.5 系列模型。

它的设计参考了 SGLang serving runtime 的核心分层，但进行了明显收缩：当前主路径以
单机单 GPU 为目标，优先保证在 Jetson 上可运行、可调试、可扩展，而不是覆盖大规模
分布式 serving 的全部复杂度。

.. figure:: ../_static/img/pymllm-arch.png
   :width: 100%
   :alt: pymllm runtime architecture
   :align: center

   Figure 1: pymllm runtime architecture.

整体分层
----------------------------------------

从开发者视角看，``pymllm`` 可以分为五层：

1. **服务入口层**：FastAPI HTTP server，提供 OpenAI-compatible API 和原生
   ``/generate`` API。
2. **配置层**：``ServerConfig``、``ModelConfig``、``QuantizationConfig`` 统一解析
   模型路径、dtype、调度参数、缓存参数、量化参数和加速开关。
3. **控制面**：``Engine`` 启动 tokenizer、scheduler、detokenizer 子进程，并在主进程中
   维护 request/response 状态。
4. **数据面**：scheduler 持有 GPU-owning ``ModelRunnerProcess``，负责 batch 构造、
   KV cache 分配、prefix cache 命中、forward 和 sampling。
5. **加速层**：FlashInfer、CUDA Graph、Triton、CUTLASS 和 ``mllm-kernel`` 提供 attention、
   quantization、GEMM 和缓存写入等高频算子。

进程拓扑
----------------------------------------

``Engine`` 在启动时创建三个子进程，并在主进程中保留 request/response 管理逻辑：

.. code-block:: text

   Main Process
     ├── FastAPI Server
     ├── Engine
     └── RequestResponseProcess
            │
            │ ZMQ
            ▼
   TokenizerProcess
            │
            │ ZMQ or shared queue
            ▼
   SchedulerProcess
     └── ModelRunnerProcess  (in-process, owns GPU resources)
            │
            │ ZMQ
            ▼
   DetokenizerProcess
            │
            │ ZMQ
            ▼
   RequestResponseProcess

这个拓扑的核心取舍是：GPU 资源由 scheduler 进程内的 ``ModelRunnerProcess`` 直接持有。
这样 scheduler 可以在同一进程中完成调度、KV cache 资源释放、prefix cache 更新和模型
forward，避免再引入 model worker 进程之间的 GPU 资源同步。

请求生命周期
----------------------------------------

一次 chat completion 请求的典型路径如下：

1. HTTP server 接收请求并转换为 ``GenerateReqInput``。
2. ``RequestResponseProcess`` 为请求分配 request id，并把请求送入 tokenizer。
3. ``TokenizerProcess`` 调用 tokenizer / processor，生成 ``TokenizedGenerateReqInput``。
4. ``SchedulerProcess`` 接收 tokenized request，创建 ``Req``，放入等待队列。
5. scheduler 根据 token budget、running request 数量和 prefill/decode 状态构造
   ``ScheduleBatch``。
6. ``ModelRunnerProcess`` 为 batch 分配 request slot 和 KV slot，执行 prefix matching。
7. ``ModelRunner`` 构造 ``ForwardBatch``，初始化 attention backend metadata，调用模型
   ``forward``，并对 logits 做 sampling。
8. scheduler 更新每个 ``Req`` 的输出 token、finished reason 和 timing 字段。
9. ``DetokenizerProcess`` 将 token id 转回文本。
10. HTTP server 以普通 JSON 或 SSE streaming 形式返回结果。

控制面：Engine 与配置
----------------------------------------

``pymllm.configs.server_config.ServerConfig`` 是服务运行时的主配置对象。它覆盖：

- 模型和 tokenizer：``model_path``、``tokenizer_path``、``load_format``、``dtype``。
- HTTP server：``host``、``port``、``api_key``、``served_model_name``。
- 调度与内存：``max_running_requests``、``max_total_tokens``、``max_prefill_tokens``、
  ``mem_fraction_static``。
- 加速后端：``attention_backend``、``gdn_decode_backend``、``disable_cuda_graph``、
  ``enable_torch_compile``。
- IPC 与多模态传输：``enable_shared_queue``、``tensor_transport_mode``、
  ``cuda_ipc_pool_size_mb``。
- 观测与调试：``log_level``、``decode_log_interval``。

``Engine`` 启动前会加载 HuggingFace config，解析 EOS token、默认输出长度和 dtype，并确保
model/tokenizer 路径可用。启动后，``Engine`` 会监控子进程健康状态；任一核心子进程异常退出，
服务会被标记为 unhealthy。

调度器
----------------------------------------

``SchedulerProcess`` 是 pymllm 的中心调度组件。它负责：

- 接收 tokenized requests。
- 将输入请求转换为内部 ``Req`` 状态。
- 根据 prefill/decode 状态构造 ``ScheduleBatch``。
- 控制 ``max_running_requests``、``max_total_tokens``、``max_prefill_tokens`` 等资源约束。
- 在请求结束或中止时释放 request slot 和 KV slot。
- 将 decode token 发送给 detokenizer。

当前调度策略以 FCFS 和单 GPU 资源约束为主。``max_prefill_tokens`` 用于限制一轮调度
可接纳的 prefill token 数；长 prompt 的运行时 chunked prefill 切分仍待后续接入。

ModelRunner
----------------------------------------

``ModelRunner`` 是真正执行模型 forward 的组件。它在初始化阶段完成：

1. 设置 CUDA device 和默认 dtype。
2. 加载模型类和 safetensors 权重。
3. 解析模型 metadata，例如 layer 数、head 数、head dim、context length。
4. 初始化 request-to-token pool、token-to-KV pool 和 KV allocator。
5. 初始化 attention backend。
6. 预热 cuBLAS。
7. 按配置捕获 decode CUDA Graph。

forward 阶段分为 extend 和 decode 两类：

- **extend / prefill**：处理 prompt token，写入 KV cache，并返回每个请求最后一个 token 的
  logits。
- **decode**：每个请求生成一个新 token，复用已有 KV cache 和 attention metadata。

KV cache 与 prefix cache
----------------------------------------

``pymllm.mem_cache.memory_pool`` 中的 KV 管理采用三层结构：

.. code-block:: text

   ReqToTokenPool
       maps (request slot, position) -> kv index

   TokenToKVPoolAllocator
       manages free integer KV slots

   KVPool
       stores per-layer K/V tensors on GPU

``TokenToKVPoolAllocator`` 使用 free-list 管理 KV slot，并通过批量释放接口降低大量请求结束或
prefix cache eviction 时的开销。``KVPool`` 在条件满足时会调用 ``mllm-kernel`` 的
``store_cache`` JIT kernel 写入 K/V；否则回退到 PyTorch indexing。

Prefix cache 当前有三种实现：

- ``RadixCache``：标准 radix-tree prefix cache。
- ``ChunkCache``：关闭 radix cache 时使用的简单缓存路径。
- ``MambaRadixCache``：为包含 GDN / Mamba-like 状态的 hybrid 模型预留的状态缓存路径。

当启用 ``RadixCache`` 时，extend batch 会先执行 prefix matching。命中的 prefix token 不再
重复计算，但对应 radix tree 节点会被 lock，直到请求结束或资源释放时再 unlock。

IPC 与多模态数据传输
----------------------------------------

普通控制消息通过 ZMQ 传输。多模态请求中的大 tensor 可以走 shared queue fast path，
由 ``enable_shared_queue`` 和 ``tensor_transport_mode`` 控制。

``tensor_transport_mode`` 支持三种模式：

.. list-table::
   :header-rows: 1

   * - 模式
     - 行为
     - 适用场景
   * - ``default``
     - GPU tensor 先拷到 CPU，再放入 POSIX shared memory。
     - 最稳妥，调试优先。
   * - ``cuda_ipc``
     - GPU tensor 通过 CUDA IPC handle 跨进程共享。
     - 避免 GPU->CPU 拷贝，但长服务中可能有 PyTorch IPC 生命周期问题。
   * - ``cuda_ipc_pool``
     - 使用预分配 GPU workspace，发送方回收 chunk。
     - 面向生产服务的推荐 GPU tensor 传输方式。

与 mllm C++ Backend 的关系
----------------------------------------

``pymllm`` 和 ``cpu_backend``、``qnn_backend``、``ascend_backend`` 的层级不同：

- C++ Backend 接入的是 mllm C++ 的 Tensor、Op、Module、Dispatcher 和设备 allocator。
- ``pymllm`` 接入的是 Python/PyTorch serving pipeline，主要服务于在线推理、模型加载、
  KV cache、调度和 CUDA kernel 集成。
- ``mllm-kernel`` 是两者可以共享思想的低层 kernel 工具包，但当前 ``pymllm`` 更直接依赖
  其中的 Python JIT CUDA kernel。
