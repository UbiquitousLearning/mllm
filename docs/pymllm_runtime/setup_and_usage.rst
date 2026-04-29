pymllm Setup and Usage
======================

总览
----------------------------------------

``pymllm`` 是 mllm 面向 Python 生态的推理服务运行时，主要面向 NVIDIA Jetson
Orin 系列边缘 GPU 设备，例如 Jetson Orin NX 与 Jetson AGX Orin。它覆盖
Qwen3 / Qwen3-VL 的 BF16、W4A16 和 W8A8 推理路径，并提供 OpenAI-compatible
HTTP API。

环境要求
----------------------------------------

当前推荐基于 `jetson-containers <https://github.com/dusty-nv/jetson-containers>`_
提供的 Jetson PyTorch/CUDA 基础镜像进行开发。这样可以避免在 Jetson 上手工处理
PyTorch、CUDA、cuDNN、Python ABI 等基础依赖。

已验证环境如下：

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

在 Jetson 容器中克隆仓库后，进入仓库根目录安装 ``pymllm`` 和 ``mllm-kernel``：

.. code-block:: bash

   cd <repo-root>
   SKBUILD_WHEEL_CMAKE=false python3 -m pip install -e .
   python3 -m pip install -e <repo-root>/mllm-kernel --no-deps --no-build-isolation

``transformers`` 可按项目需要自行安装。``triton`` 和 ``flashinfer`` 可以从
Jetson AI Lab 的 wheel 源安装，也可以从官方 PyPI 或对应上游项目安装：

.. code-block:: bash

   # 方式一：从 Jetson AI Lab 安装 Jetson wheel。
   python3 -m pip install --extra-index-url https://pypi.jetson-ai-lab.io/ triton flashinfer

   # 方式二：从官方 PyPI 固定 Triton，再单独安装 FlashInfer。
   python3 -m pip install --index-url https://pypi.org/simple triton==3.6.0
   python3 -m pip install --extra-index-url https://pypi.jetson-ai-lab.io/ flashinfer

在 Jetson / aarch64 上，Triton wheel 的可用性会受到 wheel 来源、CUDA 路径和
``ptxas`` / ``cuda.h`` 查找路径影响。Jetson AI Lab 源提供面向 JetPack 6 /
CUDA 12.6 的 Triton wheel；在已验证环境中，官方 PyPI 的 ``triton==3.6.0``
manylinux aarch64 wheel 更接近开箱即用。若使用 Jetson AI Lab wheel 遇到
``ptxas`` 或 CUDA 头文件查找问题，可显式设置 ``TRITON_PTXAS_PATH`` 和
``CPATH`` 后重试。无论选择哪个来源，都建议用最小 Triton kernel 或
``per_token_quant_int8`` 做 smoke test。

最小导入检查：

.. code-block:: bash

   python3 - <<'PY'
   import pymllm
   import mllm_kernel

   print("pymllm import ok")
   print("mllm_kernel import ok")
   PY

CUTLASS 头文件
----------------------------------------

W8A8 的高性能 GEMM 路径依赖 CUTLASS 头文件。当前查找顺序为：

1. ``CUTLASS_HOME/include``
2. ``flashinfer`` 内置的 ``data/cutlass/include``
3. ``/usr/local/include``、``/usr/include``、``/usr/local/cuda/include``

首次调用 CUTLASS W8A8 kernel 会触发 JIT 编译，编译产物会复用：

.. code-block:: text

   ~/.cache/mllm_kernel/cutlass_int8_scaled_mm/

如果需要重新验证首次编译行为，可以删除该目录后再次运行。

启动服务
----------------------------------------

``pymllm`` 的服务入口是 ``pymllm.server.launch``。服务启动后会提供
``/health``、``/v1/models``、``/v1/completions``、``/v1/chat/completions``、
``/generate`` 等接口。

W4A16 / W8A8 量化模型
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``compressed-tensors`` 量化模型使用同一个启动入口。运行时会根据模型
``config.json`` 中的量化配置识别 W4A16 或 W8A8 路径。

.. code-block:: bash

   cd <repo-root>

   python3 -m pymllm.server.launch \
     --server.model_path <quantized-model-path> \
     --server.tokenizer_path <quantized-model-path> \
     --server.load_format safetensors \
     --server.dtype float16 \
     --quantization.method compressed-tensors \
     --server.host 0.0.0.0 \
     --server.port 30000 \
     --server.attention_backend auto \
     --server.gdn_decode_backend pytorch \
     --server.mem_fraction_static 0.05 \
     --server.max_running_requests 1 \
     --server.max_total_tokens 256 \
     --server.max_prefill_tokens 128 \
     --server.disable_radix_cache \
     --server.disable_cuda_graph \
     --server.log_level debug

BF16 原生模型
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

BF16 或 FP16 原生模型不需要设置 ``--quantization.method``：

.. code-block:: bash

   cd <repo-root>

   python3 -m pymllm.server.launch \
     --server.model_path <model-path> \
     --server.tokenizer_path <model-path> \
     --server.load_format safetensors \
     --server.dtype bfloat16 \
     --server.host 0.0.0.0 \
     --server.port 30000 \
     --server.attention_backend auto \
     --server.mem_fraction_static 0.05 \
     --server.max_running_requests 1 \
     --server.max_total_tokens 256 \
     --server.max_prefill_tokens 128 \
     --server.disable_radix_cache \
     --server.log_level info

常用参数
----------------------------------------

.. list-table::
   :header-rows: 1

   * - 参数
     - 说明
   * - ``--server.model_path``
     - 模型权重目录，通常是 HuggingFace 或 ModelScope 格式。
   * - ``--server.tokenizer_path``
     - tokenizer 目录；不设置时默认等于 ``model_path``。
   * - ``--server.dtype``
     - 模型运行 dtype，可选 ``auto``、``float16``、``bfloat16``、``float32``。
   * - ``--quantization.method compressed-tensors``
     - 启用 ``compressed-tensors`` 权重加载与线性层执行路径。
   * - ``--server.max_running_requests``
     - 同时运行的请求数。Jetson 小显存环境下通常从 ``1`` 开始调试。
   * - ``--server.max_total_tokens``
     - KV cache token pool 的总容量上限。
   * - ``--server.max_prefill_tokens``
     - 单轮 prefill 可处理的 token 上限。
   * - ``--server.disable_radix_cache``
     - 关闭 Radix Cache，改用 ``ChunkCache``。
   * - ``--server.disable_cuda_graph``
     - 关闭 decode CUDA Graph，便于调试动态路径。

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

图文请求中，图片路径需要是容器内可访问的绝对路径，不要带 ``file://`` 前缀：

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

开发与测试
----------------------------------------

常用单元测试：

.. code-block:: bash

   pytest pymllm/tests/test_compressed_tensors_config.py -q
   pytest pymllm/tests/test_compressed_tensors_runtime.py -q
   pytest pymllm/tests/test_qwen3_model_registry.py -q
   pytest pymllm/tests/test_qwen3_weight_loading.py -q
   pytest pymllm/tests/test_qwen3_forward_timing.py -q
   pytest mllm-kernel/tests/test_int8_scaled_mm_cutlass.py -q

模型级 benchmark：
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``bench_one_batch`` 是对齐 SGLang 口径的低层离线 benchmark。它直接初始化
``pymllm.executor.model_runner.ModelRunner``，绕过 HTTP server、tokenizer 进程、
scheduler 进程和 detokenizer 进程，用 synthetic text-only token ids 测一次静态
prefill，再测逐 token decode。该工具适合分析模型 forward、KV cache、attention、
CUDA Graph 与量化 kernel 的模型级开销，不代表在线服务的 TTFT / ITL / E2E 指标。

典型用法：

.. code-block:: bash

   PYTHONPATH="$PWD:$PWD/mllm-kernel" python3 -m pymllm.bench_one_batch \
     --server.model_path <model-or-quantized-model-path> \
     --server.tokenizer_path <model-or-quantized-model-path> \
     --server.load_format safetensors \
     --server.dtype float16 \
     --quantization.method compressed-tensors \
     --server.mem_fraction_static 0.1 \
     --server.max_running_requests 1 \
     --server.max_total_tokens 2048 \
     --server.disable_radix_cache \
     --server.log_level info \
     --run-name qwen3vl_w8a8_bench_one_batch \
     --batch-size 1 \
     --input-len 256 512 1024 \
     --output-len 128 \
     --result-filename /tmp/pymllm_bench_one_batch.jsonl

其中 ``--batch-size``、``--input-len`` 和 ``--output-len`` 都支持多个值，脚本会遍历
所有组合并向 JSONL 文件追加结果。``output_len`` 采用 SGLang 的总输出 token 语义：
prefill 后已得到第一个 next token，后续 decode loop 执行 ``output_len - 1`` 步。

执行结构：

.. code-block:: text

   pymllm.bench_one_batch CLI
     |
     |-- parse GlobalConfig args and BenchArgs
     |-- load HuggingFace AutoConfig into cfg.model.hf_config
     |
     |-- ModelRunner.initialize()
     |     |-- load model and quantization config
     |     |-- initialize KV pools and attention backend
     |     |-- optionally capture decode CUDA Graph
     |
     |-- warmup once
     |
     |-- for each (batch_size, input_len, output_len):
           |
           |-- clear req_to_token_pool and token_to_kv_pool_allocator
           |-- build synthetic input_ids
           |-- prefill:
           |     allocate request slots and KV slots
           |     write prompt KV mapping
           |     prepare ForwardBatch(EXTEND)
           |     synchronize, run forward + sampling, synchronize
           |
           |-- decode loop:
                 allocate one KV slot per request
                 write current token mapping
                 prepare ForwardBatch(DECODE)
                 synchronize, run forward + sampling, synchronize
                 update seq_lens and next token ids
     |
     |-- append JSONL result rows

Profile 辅助入口：
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``bench_one_batch`` 保留了基于 ``torch.profiler`` 的 profile 参数，主要用于本地
kernel timeline 分析。当前公开 benchmark 记录没有使用 profile 结果，因此它不作为标准
性能数据口径的一部分。使用前建议先用较小的 ``input_len`` / ``output_len`` 做一次
trace 生成验证，再扩大到正式 case。

.. code-block:: bash

   PYMLLM_TORCH_PROFILER_DIR=/tmp \
   PYTHONPATH="$PWD:$PWD/mllm-kernel" python3 -m pymllm.bench_one_batch \
     --server.model_path <model-path> \
     --server.tokenizer_path <model-path> \
     --server.load_format safetensors \
     --server.dtype bfloat16 \
     --server.mem_fraction_static 0.1 \
     --server.max_running_requests 1 \
     --server.max_total_tokens 2048 \
     --server.log_level info \
     --batch-size 1 \
     --input-len 256 \
     --output-len 128 \
     --profile \
     --profile-stage decode \
     --profile-steps 1

已知限制
----------------------------------------

- W8A8 CUTLASS 当前通过 JIT 编译，首次启动有明显编译开销。
- W8A8 激活量化使用 Triton kernel；decode 小 batch 下固定量化开销仍是后续优化点。
- Qwen3-VL 的 ViT、``lm_head``、embedding 和 LayerNorm 不在当前 W8A8 量化范围内。
- 当前文档中的 Jetson 性能与稳定性结论主要来自 Orin NX / SM87，需要在其他 GPU 上重新验证。
- OpenAI-compatible API 的服务级指标和 ``bench_one_batch`` 的模型级指标口径不同，不应直接混用。
