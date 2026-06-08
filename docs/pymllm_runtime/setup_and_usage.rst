pymllm Setup and Usage
======================

总览
----------------------------------------

``pymllm`` 是 mllm 面向 Python / CUDA 生态的推理服务运行时，主要跑在 NVIDIA Jetson
Orin 系列边缘 GPU（Orin NX / AGX Orin）上。它针对 Orin Ampere Tensor Core 的 INT8
算力做了系统级适配，支持 BF16 原生推理以及 W4A16、W8A8_INT8 两种量化方案，兼顾推理
速度与模型精度，目前已完成对 Qwen3、Qwen3-VL、Qwen3.5 的支持，并对外提供一套
OpenAI-compatible 的 HTTP API。

环境要求
----------------------------------------

下面是当前已经跑通的一组版本：

.. list-table::
   :header-rows: 1

   * - 组件
     - 版本或说明
   * - JetPack / Jetson Linux
     - JetPack ``6.2.1`` / Jetson Linux ``36.4.4`` (L4T ``R36.4.4``)
   * - Python
     - ``3.10.12``
   * - PyTorch
     - ``2.4.0``
   * - torchvision
     - ``0.19.0a0+48b1edf``
   * - transformers
     - ``5.3.0``
   * - safetensors
     - ``0.7.0``
   * - flashinfer
     - ``0.6.7``
   * - Triton Language
     - ``triton==3.6.0`` aarch64 wheel
   * - CUDA
     - ``12.6``
   * - GPU
     - Jetson Orin NX，SM87

安装依赖
----------------------------------------

克隆仓库后，进入根目录安装 ``pymllm`` 和 ``mllm-kernel``：

.. code-block:: bash

   cd <repo-root>
   SKBUILD_WHEEL_CMAKE=false python3 -m pip install -e .
   python3 -m pip install -e <repo-root>/mllm-kernel --no-deps --no-build-isolation

``triton`` 和 ``flashinfer`` 有两个来源，任选其一：

.. code-block:: bash

   # 方式一：从 Jetson AI Lab 装 Jetson wheel。
   python3 -m pip install --extra-index-url https://pypi.jetson-ai-lab.io/ triton flashinfer

   # 方式二：从官方 PyPI 固定 Triton 版本，FlashInfer 仍从 Jetson AI Lab 装。
   python3 -m pip install --index-url https://pypi.org/simple triton==3.6.0
   python3 -m pip install --extra-index-url https://pypi.jetson-ai-lab.io/ flashinfer

在 aarch64 上，Triton wheel 能不能开箱即用，主要取决于 wheel 来源以及
``ptxas`` / ``cuda.h`` 的查找路径。在上面这组已验证环境里，官方 PyPI 的
``triton==3.6.0`` manylinux aarch64 wheel 更接近开箱即用；如果用 Jetson AI Lab
的 wheel 碰到 ``ptxas`` 或 CUDA 头文件找不到的问题，显式设置 ``TRITON_PTXAS_PATH``
和 ``CPATH`` 再重试通常能解决。装完后建议用 ``per_token_quant_int8`` 之类的最小
kernel 跑一次 smoke test，确认 Triton 真的能编译。

W8A8 首次运行的 JIT 编译
----------------------------------------

W8A8 的 INT8 GEMM 走 CUTLASS，依赖 CUTLASS 头文件。默认情况下不需要额外配置——
``flashinfer`` 自带了一份 bundled CUTLASS，可以直接用；如果想换成自己的版本，设置
``CUTLASS_HOME`` 即可。

第一次调用 W8A8 kernel 会触发一次 JIT 编译，编译产物缓存在：

.. code-block:: text

   ~/.cache/mllm_kernel/cutlass_int8_scaled_mm/

之后复用缓存，不会再编译。想重新验证首次编译行为时，删掉这个目录再跑一次就行。

启动服务
----------------------------------------

服务入口是 ``pymllm.server.launch``，启动后提供 ``/health``、``/v1/models``、
``/v1/completions``、``/v1/chat/completions``、``/generate`` 等接口。

W4A16 / W8A8 量化模型和 BF16 原生模型共用同一个入口，运行时会读 ``config.json``
里的量化配置，自动走 W4A16 或 W8A8 路径。一条典型的量化模型启动命令：

.. code-block:: bash

   cd <repo-root>

   python3 -m pymllm.server.launch \
     --server.model_path <quantized-model-path> \
     --server.dtype float16 \
     --quantization.method compressed-tensors \
     --server.host 0.0.0.0 \
     --server.port 30000 \
     --server.mem_fraction_static 0.8 \
     --server.max_running_requests 1 \
     --server.max_total_tokens 4096 \
     --server.disable_radix_cache \
     --server.log_level debug

BF16 / FP16 原生模型用同一条命令，去掉 ``--quantization.method`` 即可。

常用参数
----------------------------------------

.. list-table::
   :header-rows: 1

   * - 参数
     - 说明
   * - ``--server.model_path``
     - 模型权重目录，通常是 HuggingFace 或 ModelScope 格式。
   * - ``--server.tokenizer_path``
     - tokenizer 目录；不设置时默认等于 ``model_path``，一般不用单独传。
   * - ``--server.dtype``
     - 模型运行 dtype，可选 ``auto``、``float16``、``bfloat16``。
   * - ``--quantization.method compressed-tensors``
     - 启用 ``compressed-tensors`` 权重加载和量化线性层执行路径。
   * - ``--server.mem_fraction_static``
     - ``模型权重 + KV cache pool`` 占 GPU 总显存的静态预算比例。设太小，KV pool 预算
       不足会导致启动报错；设太大，留给 activation 和 CUDA Graph 的动态空间不够。
       Jetson 上 Qwen3-VL-2B 量化模型一般在 ``0.5``–``0.8`` 之间起调。
   * - ``--server.max_running_requests``
     - 同时运行的请求数。Jetson 小显存环境一般从 ``1`` 开始调。
   * - ``--server.max_total_tokens``
     - KV cache token pool 的容量上限，是整个 worker 全局共享的池子（不是单请求上限）。
       实际容量取 ``min(profile 可承载 token 数, max_total_tokens)``，不会绕过显存 profile。
   * - ``--server.disable_radix_cache``
     - 关闭 Radix Cache，改用 ``ChunkCache``。

OpenAI-compatible 请求
----------------------------------------

健康检查：

.. code-block:: bash

   curl -s --noproxy '*' http://127.0.0.1:30000/v1/models ; echo

文本请求：

.. code-block:: bash

   curl -s --noproxy '*' http://127.0.0.1:30000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
       "model": "default",
       "messages": [{"role": "user", "content": "你好，只回复：ok"}],
       "max_tokens": 8,
       "temperature": 0.0,
       "stream": false
     }' ; echo

图文请求里的图片路径要用服务进程可访问的绝对路径，不要带 ``file://`` 前缀：

.. code-block:: bash

   cat > /tmp/mm_req_path.json <<'JSON'
   {
     "model": "default",
     "messages": [
       {
         "role": "user",
         "content": [
           {"type": "text", "text": "请描述这张图片。"},
           {"type": "image_url", "image_url": {"url": "/workspace/test.png"}}
         ]
       }
     ],
     "max_tokens": 128,
     "temperature": 0.0,
     "stream": false
   }
   JSON

   curl -s --noproxy '*' http://127.0.0.1:30000/v1/chat/completions \
     -H "Content-Type: application/json" \
     --data @/tmp/mm_req_path.json ; echo

Benchmark
----------------------------------------

``bench_one_batch`` 是一个低层离线 benchmark。它直接初始化
``pymllm.executor.model_runner.ModelRunner``，绕过 HTTP server、tokenizer、scheduler、
detokenizer 这些进程，只测模型本身一次静态 prefill 加逐 token decode 的开销，因此适合
分析模型 forward、KV cache、attention、CUDA Graph 和量化 kernel 的模型级表现，也方便
验证 fused projection、residual-carry 这类模型图优化。它测不到在线服务的 TTFT / ITL /
E2E，这两个口径不要混用。

目前 ``bench_one_batch`` 支持三种测速口径：

- **纯文本**：用 synthetic token ids 测纯文本的 prefill / decode；
- **视觉编码（vit_prefill）**：同步墙钟只包住视觉 encoder（``self.visual(...)``），
  反映纯视觉编码速度；
- **多模态 prefill（multimodal_prefill）**：覆盖“视觉编码 + 图像/文本 token 的 LLM
  prefill”，反映完整多模态 prefill 速度。

纯文本用法：

.. code-block:: bash

   PYTHONPATH="$PWD:$PWD/mllm-kernel" python3 -m pymllm.bench_one_batch \
     --server.model_path <model-or-quantized-model-path> \
     --server.dtype float16 \
     --quantization.method compressed-tensors \
     --server.mem_fraction_static 0.8 \
     --server.max_running_requests 1 \
     --server.max_total_tokens 2048 \
     --server.log_level info \
     --run-name qwen3vl_w8a8_bench_one_batch \
     --batch-size 1 \
     --input-len 256 512 1024 \
     --output-len 128 \
     --result-filename <result-jsonl-path>

``--batch-size``、``--input-len``、``--output-len`` 都支持多个值，脚本会遍历所有组合
并把结果追加到 JSONL 文件。``output_len`` 用的是总输出 token 语义：prefill 之后已经
拿到第一个 next token，后续 decode loop 再跑 ``output_len - 1`` 步。

多模态 prefill 用法。给 ``--image`` 传一张真实图片，再显式传 ``--input-len`` 时，长度
口径是 ``image placeholder tokens + text prompt tokens`` 的目标总长——脚本只在文本
token 上做补齐或截断，绝不动 image token，因此可以用同一张图 sweep
``314/512/1024/2048`` 等不同总长，测包含视觉编码的完整多模态 prefill 速度：

.. code-block:: bash

   PYTHONPATH="$PWD:$PWD/mllm-kernel" python3 -m pymllm.bench_one_batch \
     --server.model_path <qwen3-vl-quantized-model-path> \
     --server.trust_remote_code true \
     --server.dtype float16 \
     --quantization.method compressed-tensors \
     --server.mem_fraction_static 0.8 \
     --server.max_running_requests 1 \
     --server.disable_cuda_graph \
     --batch-size 1 \
     --input-len 314 512 1024 2048 \
     --output-len 1 \
     --image <image-path> \
     --prompt "Describe this image." \
     --run-name qwen3vl_w8a8_multimodal_prefill \
     --result-filename <result-jsonl-path>

JSONL 里 ``vit_prefill_ms`` 只包住 ``self.visual(...)``，``multimodal_prefill_*``
则是完整 VIT + LLM prefill 的别名字段，两者口径不同。在 AGX Orin 32GB 上的实测中，
W8A8 在长 prefill 上明显领先 FP16 / W4A16。

脚本的整体执行流程大致是：

.. code-block:: text

   pymllm.bench_one_batch CLI
     |
     |-- 解析 GlobalConfig 参数和 BenchArgs
     |-- 加载 HuggingFace AutoConfig 到 cfg.model.hf_config
     |
     |-- ModelRunner.initialize()
     |     |-- 加载模型和量化配置
     |     |-- 初始化 KV pool 和 attention backend
     |     |-- 按需 capture decode CUDA Graph
     |
     |-- warmup 一次
     |
     |-- 遍历每个 (batch_size, input_len, output_len):
     |     |-- 清空 req_to_token_pool 和 token_to_kv_pool_allocator
     |     |-- 构造 synthetic input_ids
     |     |-- prefill：分配 request/KV slot，写 KV 映射，跑 forward + sampling
     |     |-- decode loop：逐步分配 KV slot，跑 forward + sampling，更新 seq_lens
     |
     |-- 追加 JSONL 结果行

Profile
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``bench_one_batch`` 内置了 profile 入口，方便在本地看 kernel timeline。目前有两条路径：

- **torch.profiler（已支持）**：``--profile-activities CPU GPU``（默认），输出
  ``.trace.json.gz`` timeline，可以直接在 Perfetto / chrome://tracing 里看。输出目录
  由 ``PYMLLM_TORCH_PROFILER_DIR`` 指定，默认 ``/tmp``。
- **Nsight Systems / nsys（实验性）**：``--profile-activities CUDA_PROFILER`` 通过
  ``cudaProfilerStart/Stop`` 驱动 nsys，需要外层用
  ``nsys --capture-range=cudaProfilerApi`` 包住命令。这条路径还在打磨中，部分场景下
  可能不够顺手，仅作为可选的深入分析手段。
