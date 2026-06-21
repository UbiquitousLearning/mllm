Qwen Ascend
===========

总览
----

Qwen Ascend 是当前 Ascend Backend 的主要端到端验证路径，代码位于
``mllm/models/qwen_ascend``，示例位于 ``examples/qwen_ascend``。

该模型路径覆盖：

- Qwen 配置读取。
- tokenizer 和 chat template。
- Ascend KV cache。
- RoPE cache。
- decoder eager 路径。
- decoder graph FP16 / W8A8 路径。
- QA generation 和 forward smoke test。

文件结构
--------

.. list-table::
   :header-rows: 1

   * - 文件
     - 说明
   * - ``configuration_qwen_ascend.hpp``
     - Qwen Ascend 配置读取。
   * - ``tokenization_qwen_ascend.hpp``
     - tokenizer、message template 和输入构造。
   * - ``modeling_qwen_ascend.hpp``
     - 模型结构：MLP、Attention、Decoder、LM。
   * - ``qwen_ascend_decoder_graph.hpp``
     - decoder graph 构建和 graph forward。
   * - ``qwen_ascend_graph_ops.hpp``
     - graph op factory、runner 和 graph 环境。
   * - ``qwen_ascend_rope.hpp``
     - RoPE position id、inv_freq 和 cache。

模型结构
--------

``QwenAscendForCausalLM`` 继承 ``ARGeneration`` 和 ``nn::Module``，内部包含：

- ``QwenAscendText``：decoder block 列表和 final RMSNorm。
- ``QwenAscendDecoder``：self-attention、MLP、decoder graph runner。
- ``QwenAscendAttention``：Q/K/V/O projection、RoPE、causal mask、KV cache update。
- ``QwenAscendMLP``：gate/up/down projection 和 SiLU。
- ``AscendKVCache``：按 layer 保存 K/V cache。
- ``QwenAscendRoPECache``：缓存 sin/cos embedding。

推理时，输入 token id 先经过 embedding，然后进入多层 decoder，最后通过 lm head 生成
logits。生成流程复用 mLLM 的 ``ARGeneration::chat`` 接口。

加载顺序
--------

Qwen Ascend 示例会先将模型移动到 Ascend，再加载权重：

.. code-block:: cpp

   auto model = QwenAscendForCausalLM(cfg);
   model.to(mllm::kAscend);
   model.load(mllm::load(model_path, file_version));

这个顺序很重要。对于 W8A8 权重，``AscendLinearOp::load()`` 需要直接读取模型文件中的
``scale`` 和 ``scale_x`` 参数，并准备后续 graph 所需的量化 artifacts。

KV Cache
--------

``AscendKVCache`` 是 Ascend 专用 KV cache。它按 layer 保存 K/V buffer：

.. code-block:: text

   K cache: [1, kv_heads, max_cache_length, head_dim]
   V cache: [1, kv_heads, max_cache_length, head_dim]

主要接口：

- ``updateKVCache()``：eager 路径写入当前 step 的 K/V，并返回当前有效 cache。
- ``advanceSeqCnt()``：graph/plugin 已经更新 cache 后推进 sequence count。
- ``clearCache()``：开始新一轮生成前清空 cache 状态。

GQA 的 K/V repeat 不在 cache 内做，而是在 attention 计算中处理。

RoPE Cache
----------

``qwen_ascend_rope.hpp`` 提供：

- ``makeLocalRoPEPositionIds()``
- ``makeRoPEInvFreq()``
- ``QwenAscendRoPECache``

RoPE cache 负责为当前 sequence 准备 sin/cos tensor，避免重复构造相同位置编码。
Ascend RoPE 当前按 ATB RoPE 的输入约定组织 Q/K、cos、sin 和 position ids。

Decoder Graph
-------------

Qwen Ascend decoder 默认优先使用 graph 路径：

.. code-block:: bash

   export MLLM_ASCEND_QWEN_DECODER_GRAPH=1

设为 ``0`` 可关闭 graph，回退 eager：

.. code-block:: bash

   export MLLM_ASCEND_QWEN_DECODER_GRAPH=0

``QwenAscendDecoder::canUseGraph()`` 会检查 graph 开关、输入形状和算子状态。
首次进入 graph 路径时，``ensureGraphExecutor()`` 会构建并缓存当前 layer 的 graph executor。

Graph 目前分两类：

- FP16 decoder graph：使用 ATB Linear、RMSNorm、RoPE、Transpose、Attention plugin、SiLU 和 Add。
- W8A8 decoder graph：将 Linear 节点替换为 ``AscendLinearW8A8PluginOperation``。

Attention graph 节点通过 ``AscendAttentionWithKVCachePluginOperation`` 封装 prefill/decode
子图、KV cache 访问和 sequence length 更新。

W8A8 路径
---------

Qwen Ascend 支持静态 W8A8 linear。加载 INT8 权重时，``AscendLinearOp`` 会准备：

- ``scale_x``：activation per-tensor scale。
- ``scale_w``：weight per-channel scale。
- ``deq_scale_npu``：``scale_x * scale_w`` 转换后的 ATB dequant 参数。
- ``deq_scale_w_npu``：仅包含 ``scale_w`` 的 dequant 参数，供 dynamic debug 路径使用。
- ``bias_int32_npu``：ATB W8A8 linear 所需 int32 bias。

默认生产路径使用静态校准的 W8A8 graph plugin：

.. code-block:: text

   x_fp16
     -> x * (1 / scale_x)
     -> round
     -> clamp [-128, 127]
     -> cast int8
     -> ATB Linear W8A8
     -> y_fp16

Dynamic W8A8 eager 路径需要显式打开：

.. code-block:: bash

   export MLLM_ASCEND_ENABLE_DYNAMIC_W8A8=1

该路径用于精度分析和调试，不是默认推理路径。

示例运行模式
------------

``examples/qwen_ascend/main.cpp`` 支持两种模式。

默认 QA generation：

.. code-block:: bash

   ./mllm-qwen-ascend-runner \
     -m /path/to/model.mllm \
     -c /path/to/config.json \
     -t /path/to/tokenizer.json \
     -p "请用一句话介绍你自己。" \
     -g 64

Forward smoke test：

.. code-block:: bash

   ./mllm-qwen-ascend-runner \
     -m /path/to/model.mllm \
     -c /path/to/config.json \
     --forward_smoke_test \
     -s 8

QA generation 会输出生成 token，并在结束后打印 ``perfSummary()``。当外层通过
``max_new_tokens`` 截断生成时，示例会在已经进入 decode 阶段后补充 decode 结束时间，
保证 decode 速度统计可用。

性能与调试
----------

常用调试方式：

- 关闭 graph：``MLLM_ASCEND_QWEN_DECODER_GRAPH=0``。
- 打开 graph profiling：``MLLM_PROFILE_ASCEND_GRAPH=1``。
- 调整 profiling 打印间隔：``MLLM_PROFILE_ASCEND_GRAPH_EVERY=20``。
- 打开 dynamic W8A8 eager 调试：``MLLM_ASCEND_ENABLE_DYNAMIC_W8A8=1``。

如果默认 QA generation 的首个 prefill 较慢，需要区分冷启动开销和 steady-state 性能。
首次真实 forward 会包含 Ascend/ATB/runtime/内存池等初始化成本。
