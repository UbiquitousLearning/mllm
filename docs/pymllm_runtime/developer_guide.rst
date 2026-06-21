pymllm Developer Guide
======================

总览
----------------------------------------

本文档面向希望为 ``pymllm`` 增加模型、量化格式、kernel 或性能优化的开发者。当前代码处在
快速演进阶段，推荐遵循“小步验证、边界清晰、先单测后服务级验证”的工作方式。

开发环境建议
----------------------------------------

推荐使用 editable install，便于修改 Python 代码后直接验证：

.. code-block:: bash

   cd <repo-root>
   SKBUILD_WHEEL_CMAKE=false python3 -m pip install -e .
   python3 -m pip install -e <repo-root>/mllm-kernel --no-deps --no-build-isolation

最小检查：

.. code-block:: bash

   python3 - <<'PY'
   import pymllm
   import mllm_kernel
   print("ok")
   PY

``mllm-kernel`` 的 JIT 编译产物会写入 ``~/.cache/mllm_kernel``。正常修改后重新运行
会触发相应 kernel 的加载或编译；只有在验证首次编译行为、排查失败缓存、或更换 CUTLASS
等外部头文件来源时，才需要清理对应缓存：

.. code-block:: bash

   rm -rf ~/.cache/mllm_kernel/<kernel-cache-name>

新增模型
----------------------------------------

新增模型时，优先复用现有 ``pymllm.layers`` 和 ``pymllm.executor`` 约定，而不是把
HuggingFace 模型直接包进服务。

推荐步骤：

1. 新增 ``pymllm/models/<model_name>.py``。
2. 在 ``pymllm/models/__init__.py`` 注册 architecture 字符串。
3. 实现模型类，保持 ``forward(input_ids, positions, forward_batch)`` 风格。
4. 所有 linear layer 都接受 ``quant_method``。
5. 实现 ``load_weights``，处理 checkpoint key、stacked projection 和 tied embedding。
6. 增加最小单测。
7. 最后做服务级 smoke test。

最小测试建议：

.. code-block:: bash

   pytest pymllm/tests/test_<model>_model_registry.py -q
   pytest pymllm/tests/test_<model>_weight_loading.py -q
   pytest pymllm/tests/test_<model>_forward_timing.py -q

新增量化 scheme
----------------------------------------

新增量化路径时，不建议在模型文件里写格式判断。推荐保持以下分层：

.. code-block:: text

   QuantizationConfig
       parses checkpoint config
       decides whether a layer is quantized

   LinearMethod
       owns linear layer lifecycle

   Scheme
       owns checkpoint-facing params
       owns post-load layout conversion
       owns kernel apply path

``create_weights`` 应注册 checkpoint-facing 参数名。``process_weights_after_loading`` 应作为
checkpoint layout 到 runtime kernel layout 的唯一转换边界。``apply`` 中只做 forward 必需的
runtime 计算，不应重复做权重 repack。

新增量化路径至少需要覆盖：

- config 解析测试。
- ``ignore`` / prefix 匹配测试。
- 参数注册 shape/dtype 测试。
- post-load layout 转换测试。
- forward correctness 或 smoke test。

新增 CUDA JIT kernel
----------------------------------------

若 kernel 适合走 ``mllm-kernel`` 的 TVM-FFI JIT 路径，推荐结构如下：

.. code-block:: text

   mllm-kernel/mllm_kernel/cuda/csrc/<area>/<kernel>.cuh
   mllm-kernel/mllm_kernel/cuda/jit/<kernel>.py
   mllm-kernel/tests/test_<kernel>.py
   mllm-kernel/benchmarks/bench_<kernel>.py

Python wrapper 应负责：

- 校验输入 shape、dtype、device。
- 分配输出 tensor。
- 调用 ``@jit`` 包装后的 compiled module。
- 暴露稳定、简洁的 Python API。

CUDA/C++ source 应尽量只表达 kernel 语义，不混入 checkpoint 配置解析或模型层逻辑。

如果 kernel 依赖 CUTLASS 等重模板库，可以先做编译 spike。确认 Jetson 目标设备上的编译时间、
缓存路径、include 来源和内存占用后，再决定使用 TVM-FFI JIT、torch extension JIT 或 AOT 构建。

服务级验证
----------------------------------------

服务级 smoke test 应覆盖：

- ``/v1/models`` 可返回。
- 文本 ``/v1/chat/completions`` 可完成。
- 图文模型能处理容器内图片绝对路径。
- streaming 与 non-streaming 至少各测一次。
- 中止请求或客户端断连不会泄漏 running request。

示例：

.. code-block:: bash

   curl -s --noproxy '*' http://127.0.0.1:30000/v1/models ; echo

   curl -s --noproxy '*' http://127.0.0.1:30000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
       "model": "default",
       "messages": [{"role": "user", "content": "只回复 ok"}],
       "max_tokens": 8,
       "temperature": 0.0,
       "stream": false
     }' ; echo

性能验证
----------------------------------------

性能数据需要固定口径，否则不同记录之间很难比较。建议记录：

- commit hash。
- JetPack / L4T 版本。
- GPU 型号和 compute capability。
- PyTorch、Triton、FlashInfer、CUDA 版本。
- 模型路径和量化格式。
- 启动命令。
- prompt token 数、max tokens、temperature。
- 是否启用 radix cache、CUDA Graph、shared queue。
- 是否包含首次 JIT 编译。

对服务级请求，建议丢弃第一次 warmup 结果，记录第 2/3 次请求的 prefill/decode 统计。
对 kernel microbench，建议单独记录 warmup、重复次数、输入 shape 和 dtype。

常见问题定位
----------------------------------------

启动失败
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

优先确认：

- ``pymllm`` 和 ``mllm_kernel`` 是否来自预期源码目录或安装版本。
- ``model_path`` 和 ``tokenizer_path`` 是否在容器内可见。
- ``transformers`` 是否能读取目标 ``config.json``。
- CUDA 是否可用，``torch.cuda.get_device_capability()`` 是否符合量化 kernel 要求。

W8A8 编译失败
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

优先确认：

- ``CUTLASS_HOME`` 是否设置正确。
- ``flashinfer`` 是否包含 bundled CUTLASS。
- ``~/.cache/mllm_kernel/cutlass_int8_scaled_mm/`` 是否存在旧的失败缓存。
- 当前 GPU 是否为 SM80-SM89。

请求卡住或 CPU 占用高
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

优先确认：

- scheduler 是否启用了 idle sleep。
- tokenizer / scheduler / detokenizer 子进程是否全部存活。
- 是否有请求已经断连但未 abort。
- ``max_total_tokens`` 是否过小导致 KV allocation 反复失败和 eviction。

输出异常
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

优先确认：

- tokenizer chat template 是否符合目标模型。
- EOS token 是否从 config、generation_config 或 tokenizer 中正确解析。
- 量化模型的 ``ignore`` 是否覆盖视觉分支、embedding、norm 和 lm_head 等不应量化模块。
- ``process_weights_after_loading`` 是否已执行。

贡献建议
----------------------------------------

开发时尽量保持以下边界：

- 服务协议变化放在 ``pymllm/server``。
- 请求/响应结构放在 ``pymllm/engine/io_struct.py``。
- 调度策略放在 ``pymllm/orchestrator/scheduler_process.py``。
- GPU 资源和 forward 逻辑放在 ``pymllm/executor``。
- 模型结构放在 ``pymllm/models``。
- 基础层放在 ``pymllm/layers``。
- 量化格式放在 ``pymllm/quantization``。
- 自定义 kernel 放在 ``mllm-kernel``。

这样可以避免把一次模型适配写成跨层补丁，也方便后续把同一能力复用到更多模型和设备。
