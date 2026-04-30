pymllm Models and Quantization
==============================

总览
----------------------------------------

``pymllm`` 的模型实现遵循 PyTorch ``nn.Module`` 风格，并通过 HuggingFace
``config.architectures`` 字段选择模型类。当前重点支持 Qwen3 family：

- ``Qwen3ForCausalLM``：文本模型，例如 Qwen3-0.6B。
- ``Qwen3VLForConditionalGeneration``：图文模型，例如 Qwen3-VL-2B-Instruct。
- ``Qwen3_5ForCausalLM`` 和 ``Qwen3_5ForConditionalGeneration``：hybrid attention / GDN
  相关模型骨架。

量化系统以 linear layer 为核心，使用插件式 ``LinearMethodBase`` 生命周期：

.. code-block:: text

   QuantizationConfig
       -> get_quant_method(layer, prefix)
       -> LinearMethodBase
            -> create_weights()
            -> process_weights_after_loading()
            -> apply()

模型注册
----------------------------------------

模型注册表位于 ``pymllm/models/__init__.py``。运行时会根据 HuggingFace config 中的
architecture 字符串懒加载模型类：

.. code-block:: text

   "Qwen3ForCausalLM"
       -> pymllm.models.qwen3.Qwen3ForCausalLM

   "Qwen3VLForConditionalGeneration"
       -> pymllm.models.qwen3_vl.Qwen3VLForConditionalGeneration

   "Qwen3_5ForCausalLM"
       -> pymllm.models.qwen3_5.Qwen3_5ForCausalLM

这种注册方式让服务启动阶段只导入目标模型所需的代码，避免在命令行工具或轻量检查中提前加载
大量 PyTorch/CUDA 依赖。

Qwen3 文本模型
----------------------------------------

``Qwen3ForCausalLM`` 使用标准 decoder-only 结构：

- token embedding
- 多层 decoder block
- Q/K Norm
- 1D RoPE
- MLP
- final norm
- lm head

它复用 ``RadixAttention``、``RMSNorm``、``MLP``、``ColumnParallelLinear`` 和
``RowParallelLinear`` 等基础层。与 Qwen3-VL 文本分支相比，Qwen3 文本模型使用 1D RoPE，
不需要多模态 M-RoPE 的三维 position 逻辑。

Qwen3-VL 图文模型
----------------------------------------

``Qwen3VLForConditionalGeneration`` 在文本 decoder 外增加视觉输入处理和 M-RoPE 位置编码。
在一次图文请求中：

1. tokenizer / processor 处理 messages 和图片路径。
2. ``TokenizerProcess`` 生成 token ids 和多模态输入 tensor。
3. 多模态 tensor 通过 ZMQ 或 shared queue 送到 scheduler。
4. 模型 forward 中先处理视觉侧输入，再进入语言模型 prefill/decode。
5. decode 阶段使用每个请求保存的 ``mrope_position_delta`` 修正位置。

当前 W8A8 量化主要覆盖语言 decoder 的线性层；视觉 encoder、embedding、LayerNorm 和
``lm_head`` 保持全精度。

量化配置解析
----------------------------------------

服务启动时，``ModelRunner`` 会解析量化配置。优先级为：

1. 命令行 ``--quantization.method``。
2. checkpoint 目录中的量化配置文件。
3. ``config.json`` 中的 ``quantization_config`` 字段。

``compressed-tensors`` 路径使用 ``pymllm.quantization.methods.compressed_tensors``，
当前支持两类签名：

.. list-table::
   :header-rows: 1

   * - 签名
     - 格式
     - 权重
     - 激活
     - 执行路径
   * - W4A16
     - ``pack-quantized``
     - 4-bit packed weight
     - FP16/BF16 activation
     - Marlin WNA16 GEMM
   * - W8A8
     - ``int-quantized``
     - INT8 static weight
     - INT8 dynamic per-token activation
     - Triton quant + CUTLASS INT8 GEMM

``ignore`` 字段会让匹配前缀的模块跳过量化。例如 Qwen3-VL 的视觉分支通常保留为全精度。

W4A16 / AWQ Marlin 路径
----------------------------------------

W4A16 路径面向 ``compressed-tensors`` 的 ``pack-quantized`` checkpoint。当前支持的
约束是：

- ``format == "pack-quantized"``
- ``weights.num_bits == 4``
- ``weights.group_size == 32``
- ``weights.symmetric == true``
- ``actorder == null``
- GPU capability 不低于 SM80

权重加载和执行分为三个阶段：

.. code-block:: text

   checkpoint tensors
       weight_packed / weight_scale / weight_shape
          │
          ▼
   process_weights_after_loading()
       gptq_marlin_repack()
       marlin_permute_scales()
       create runtime-only zero/g_idx placeholders
          │
          ▼
   apply()
       gptq_marlin_gemm()

``create_weights`` 注册与 checkpoint 对齐的参数名，保证 safetensors 加载逻辑可以按名称写入。
``process_weights_after_loading`` 是 checkpoint layout 到 runtime kernel layout 的边界，repack
不应放在通用权重加载器或每次 forward 中。

W8A8 INT8 路径
----------------------------------------

W8A8 路径面向 ``compressed-tensors`` 的 ``int-quantized`` checkpoint。当前支持的约束是：

- ``format == "int-quantized"``
- ``weights.num_bits == 8``
- ``weights.type == "int"``
- ``weights.strategy == "channel"``
- ``weights.dynamic == false``
- ``weights.symmetric == true``
- ``input_activations.num_bits == 8``
- ``input_activations.type == "int"``
- ``input_activations.strategy == "token"``
- ``input_activations.dynamic == true``
- ``input_activations.symmetric == true``
- W8A8 CUTLASS 路径当前支持 Ampere / SM8x GPU（SM80-SM89）。已验证目标为
  Jetson Orin SM87；Hopper / SM90 暂不包含在当前支持范围内。

执行链路如下：

.. code-block:: text

   x(fp16/bf16)
       │
       ▼
   per_token_quant_int8()        [Triton]
       │
       ├── x_q(int8)
       └── x_scale(float32)
       │
       ▼
   int8_scaled_mm()              [CUTLASS]
       │
       └── output(fp16/bf16)

checkpoint 中的 INT8 权重通常是 ``[N, K]`` row-major。``process_weights_after_loading``
会将其转换为 ``[K, N]`` column-major 视图并整理 ``weight_scale``，以满足 CUTLASS kernel
接口约定。

LinearMethod 生命周期
----------------------------------------

所有 linear layer 都持有一个 ``quant_method``：

- 未量化时使用 ``UnquantizedLinearMethod``，注册普通 ``weight`` 并调用 ``F.linear``。
- 量化时由 ``QuantizationConfig.get_quant_method(layer, prefix)`` 返回具体方法。

典型生命周期：

1. 模型构造时，linear layer 调用 ``quant_method.create_weights`` 注册参数。
2. ``model.load_weights`` 根据参数名和 ``weight_loader`` 写入 checkpoint tensor。
3. 所有权重加载完成后，``ModelRunner`` 遍历模块并调用
   ``process_weights_after_loading``。
4. forward 时，linear layer 委托 ``quant_method.apply`` 执行。

这个边界使新增量化方法时不需要改动模型主逻辑，只需要实现新的 config 和 scheme。

新增模型的建议流程
----------------------------------------

新增模型时建议遵循以下顺序：

1. 在 ``pymllm/models/`` 中新增模型文件。
2. 在 ``pymllm/models/__init__.py`` 注册 HuggingFace architecture 字符串。
3. 实现最小 forward 接口：``forward(input_ids, positions, forward_batch)``。
4. 复用现有基础层，并确保 linear layer 接受 ``quant_method``。
5. 实现 ``load_weights``，处理 checkpoint 前缀、stacked projection 和 tied embedding。
6. 增加 registry、weight loading、forward timing 的单元测试。
7. 最后再做服务级 smoke test。

新增量化方法的建议流程
----------------------------------------

新增量化方法时建议保持三层结构：

1. ``QuantizationConfig``：解析 checkpoint 配置，决定某个 layer 是否量化。
2. ``LinearMethod``：承接 layer 生命周期。
3. ``Scheme``：处理具体格式的参数注册、post-load 转换和 kernel apply。

不要把 checkpoint 格式判断写入模型类，也不要把 runtime repack 隐藏在通用
``weight_loader`` 中。这样可以保证模型结构、权重格式和 kernel layout 三者的边界清晰。
