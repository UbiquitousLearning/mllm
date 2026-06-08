pymllm Developer Guide
======================

总览
----------------------------------------

这份文档写给想给 ``pymllm`` 加模型、加量化格式、加 kernel 或做性能优化的开发者。代码还在快速
演进，建议的工作方式是“小步验证、边界清晰、先单测再服务级验证”。

开发环境建议
----------------------------------------

推荐用 editable install，改完 Python 代码能直接验证：

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

``mllm-kernel`` 的 JIT 编译产物写在 ``~/.cache/mllm_kernel``。正常改完代码重新跑，会按需触发
对应 kernel 的加载或编译；只有在验证首次编译行为、排查失败缓存、或者换了 CUTLASS 之类外部头
文件来源时，才需要手动清对应缓存：

.. code-block:: bash

   rm -rf ~/.cache/mllm_kernel/<kernel-cache-name>

新增模型
----------------------------------------

加模型时，优先复用现有的 ``pymllm.layers`` 和 ``pymllm.executor`` 约定，别把 HuggingFace 模型
整个塞进服务。

推荐步骤：

1. 新增 ``pymllm/models/<model_name>.py``。
2. 在 ``pymllm/models/__init__.py`` 注册 architecture 字符串。
3. 实现模型类，保持 ``forward(input_ids, positions, forward_batch)`` 的风格。
4. 所有 linear layer 都接受 ``quant_method``。
5. 实现 ``load_weights``，处理好 checkpoint key、stacked projection 和 tied embedding。
6. 补最小单测。
7. 最后做服务级 smoke test。

最小测试建议：

.. code-block:: bash

   pytest pymllm/tests/test_<model>_model_registry.py -q
   pytest pymllm/tests/test_<model>_weight_loading.py -q
   pytest pymllm/tests/test_<model>_forward_timing.py -q

新增量化 scheme
----------------------------------------

加量化路径时，别在模型文件里写格式判断。保持这三层：

.. code-block:: text

   QuantizationConfig
       解析 checkpoint config
       决定某个 layer 是否量化

   LinearMethod
       承接 linear layer 生命周期

   Scheme
       管 checkpoint-facing 参数
       管 post-load layout 转换
       管 kernel apply 路径

``create_weights`` 注册 checkpoint-facing 的参数名。``process_weights_after_loading`` 是
checkpoint layout 转 runtime kernel layout 的唯一边界。``apply`` 里只做 forward 必需的 runtime
计算，不要重复做权重 repack。

新增量化路径至少要覆盖：

- config 解析测试。
- ``ignore`` / prefix 匹配测试。
- 参数注册的 shape / dtype 测试。
- post-load layout 转换测试。
- forward correctness 或 smoke test。

新增 CUDA JIT kernel
----------------------------------------

如果 kernel 适合走 ``mllm-kernel`` 的 TVM-FFI JIT 路径，推荐这个结构：

.. code-block:: text

   mllm-kernel/mllm_kernel/cuda/csrc/<area>/<kernel>.cuh
   mllm-kernel/mllm_kernel/cuda/jit/<kernel>.py
   mllm-kernel/tests/test_<kernel>.py
   mllm-kernel/benchmarks/bench_<kernel>.py

Python wrapper 负责：

- 校验输入的 shape、dtype、device。
- 分配输出 tensor。
- 调 ``@jit`` 包好的 compiled module。
- 对外暴露一个稳定、干净的 Python API。

CUDA / C++ source 尽量只表达 kernel 语义，别混进 checkpoint 配置解析或模型层逻辑。

如果 kernel 依赖 CUTLASS 这种重模板库，建议先做一次编译 spike：把 Jetson 目标设备上的编译
时间、缓存路径、include 来源和内存占用摸清楚，再决定用 TVM-FFI JIT、torch extension JIT 还是
AOT 构建。

服务级验证
----------------------------------------

服务级 smoke test 至少要覆盖：

- ``/v1/models`` 能返回。
- 文本 ``/v1/chat/completions`` 能跑完。
- 图文模型能处理容器内的图片绝对路径。
- streaming 和 non-streaming 各测一次。
- 中止请求或客户端断连时不会泄漏 running request。

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

性能数据一定要固定口径，否则不同记录之间根本没法比。建议每次都记下：

- commit hash。
- JetPack / L4T 版本。
- GPU 型号和 compute capability。
- PyTorch、Triton、FlashInfer、CUDA 版本。
- 模型路径和量化格式。
- 启动命令。
- prompt token 数、max tokens、temperature。
- 有没有开 radix cache、CUDA Graph、shared queue。
- 是否包含首次 JIT 编译。

服务级请求建议丢掉第一次 warmup 的结果，记第 2 / 3 次请求的 prefill / decode 统计。kernel
microbench 则要单独记 warmup、重复次数、输入 shape 和 dtype。

常见问题定位
----------------------------------------

启动失败
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

先看：

- ``pymllm`` 和 ``mllm_kernel`` 是不是来自预期的源码目录或安装版本。
- ``model_path`` 和 ``tokenizer_path`` 在容器内能不能看到。
- ``transformers`` 能不能读目标 ``config.json``。
- CUDA 可不可用，``torch.cuda.get_device_capability()`` 满不满足量化 kernel 的要求。

W8A8 编译失败
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

先看：

- ``CUTLASS_HOME`` 设没设对。
- ``flashinfer`` 里有没有 bundled CUTLASS。
- ``~/.cache/mllm_kernel/cutlass_int8_scaled_mm/`` 是不是有旧的失败缓存。
- 当前 GPU 是不是 SM80–SM89。

请求卡住或 CPU 占用高
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

先看：

- scheduler 有没有启用 idle sleep。
- tokenizer / scheduler / detokenizer 子进程是不是都还活着。
- 是不是有请求已经断连但没 abort。
- ``max_total_tokens`` 是不是太小，导致 KV allocation 反复失败和 eviction。

输出异常
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

先看：

- tokenizer 的 chat template 对不对得上目标模型。
- EOS token 有没有从 config、generation_config 或 tokenizer 里正确解析出来。
- 量化模型的 ``ignore`` 有没有覆盖视觉分支、embedding、norm、lm_head 这些不该量化的模块。
- ``process_weights_after_loading`` 跑没跑。

贡献建议
----------------------------------------

开发时尽量守住这些边界：

- 服务协议变化放 ``pymllm/server``。
- 请求 / 响应结构放 ``pymllm/engine/io_struct.py``。
- 调度策略放 ``pymllm/orchestrator/scheduler_process.py``。
- GPU 资源和 forward 逻辑放 ``pymllm/executor``。
- 模型结构放 ``pymllm/models``。
- 基础层放 ``pymllm/layers``。
- 量化格式放 ``pymllm/quantization``。
- 自定义 kernel 放 ``mllm-kernel``。

守住这些边界，一次模型适配就不会写成跨层补丁，后面把同一份能力复用到更多模型和设备也更省事。
