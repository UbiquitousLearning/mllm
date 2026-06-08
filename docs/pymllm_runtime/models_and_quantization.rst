pymllm Models and Quantization
==============================

总览
----------------------------------------

``pymllm`` 的模型实现就是标准的 PyTorch ``nn.Module`` 写法，运行时按 HuggingFace config
里的 ``architectures`` 字段挑模型类。当前重点是 Qwen3 family：

- ``Qwen3ForCausalLM``：文本模型，例如 Qwen3-0.6B。
- ``Qwen3VLForConditionalGeneration``：图文模型，例如 Qwen3-VL-2B-Instruct。
- ``Qwen3_5ForCausalLM`` / ``Qwen3_5ForConditionalGeneration``：hybrid attention / GDN
  方向的模型骨架。

量化系统围绕 linear layer 展开，用一套插件式的 ``LinearMethodBase`` 生命周期把格式细节
和模型主逻辑隔开：

.. code-block:: text

   QuantizationConfig
       -> get_quant_method(layer, prefix)
       -> LinearMethodBase
            -> create_weights()
            -> process_weights_after_loading()
            -> apply()

模型注册
----------------------------------------

模型注册表在 ``pymllm/models/__init__.py``。运行时按 HuggingFace config 里的 architecture
字符串懒加载对应模型类：

.. code-block:: text

   "Qwen3ForCausalLM"
       -> pymllm.models.qwen3.Qwen3ForCausalLM

   "Qwen3VLForConditionalGeneration"
       -> pymllm.models.qwen3_vl.Qwen3VLForConditionalGeneration

   "Qwen3_5ForCausalLM"
       -> pymllm.models.qwen3_5.Qwen3_5ForCausalLM

懒加载的好处是：服务启动时只导入目标模型用到的代码，命令行工具或轻量检查不会被迫提前拉起
一大堆 PyTorch / CUDA 依赖。

Qwen3 文本模型
----------------------------------------

``Qwen3ForCausalLM`` 是标准的 decoder-only 结构：token embedding、多层 decoder block、
Q/K Norm、1D RoPE、MLP、final norm、lm head。它复用 ``RadixAttention``、``RMSNorm``、
``MLP``、``ColumnParallelLinear``、``RowParallelLinear`` 这些基础层。和 Qwen3-VL 的文本分支
比，区别在于这里用的是 1D RoPE，不需要多模态 M-RoPE 那套三维 position 逻辑。

Qwen3-VL 图文模型
----------------------------------------

``Qwen3VLForConditionalGeneration`` 在文本 decoder 之外多了视觉输入处理和 M-RoPE 位置编码。
一次图文请求大致是这样走的：

1. tokenizer / processor 处理 messages 和图片路径。
2. ``TokenizerProcess`` 产出 token ids 和多模态输入 tensor。
3. 多模态 tensor 通过 ZMQ 或 shared queue 送到 scheduler。
4. 模型 forward 里先过视觉侧输入，再进语言模型的 prefill / decode。
5. decode 阶段用每个请求保存的 ``mrope_position_delta`` 修正位置。

当前 W8A8 量化主要覆盖语言 decoder 的线性层；视觉 encoder、embedding、LayerNorm 和
``lm_head`` 保持全精度。

Fused projection 与 shard-aware loading
----------------------------------------

Qwen3 / Qwen3-VL 的 text decoder 用了 fused QKV projection 和 fused gate/up projection。
对非量化模型，这减少了 projection 层的 module 边界；对 W8A8 和 W4A16 路径，它还顺手省掉了
把同一层拆成多次 activation quant、GEMM 或 Marlin 调用的开销。

checkpoint 里的权重往往还是 HuggingFace 常见的分离形式，比如 ``q_proj``、``k_proj``、
``v_proj`` 和 ``gate_proj``、``up_proj``。``MergedLinear`` 用 shard-aware 的 ``weight_loader``
把这些分离 tensor 写进 fused 参数，运行时布局保持 ``[Q, K, V]`` 或 ``[gate, up]``。权重加载
完之后，``process_weights_after_loading`` 再去做 W8A8 layout 转换或 W4A16 Marlin repack。

Qwen3 / Qwen3-VL decoder 还用 residual-carry 的形式组织 RMSNorm 的 fused add 路径。在
Qwen3-VL 里，如果需要注入 deepstack embedding，运行时会先把当前 residual sum 物化出来，再
执行注入并重置 carry，避免破坏图文 prefill 的语义。

量化配置解析
----------------------------------------

服务启动时 ``ModelRunner`` 解析量化配置，优先级是：

1. 命令行 ``--quantization.method``。
2. checkpoint 目录里的量化配置文件。
3. ``config.json`` 里的 ``quantization_config`` 字段。

``compressed-tensors`` 路径走 ``pymllm.quantization.methods.compressed_tensors``，目前支持
两类签名：

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

``ignore`` 字段会让前缀匹配上的模块跳过量化，比如 Qwen3-VL 的视觉分支通常整体保留全精度。

W4A16 / AWQ Marlin 路径
----------------------------------------

W4A16 面向 ``compressed-tensors`` 的 ``pack-quantized`` checkpoint。当前的约束是：

- ``format == "pack-quantized"``
- ``weights.num_bits == 4``
- ``weights.group_size == 32``
- ``weights.symmetric == true``
- ``actorder == null``
- GPU capability 不低于 SM80

权重加载和执行分三步：

.. code-block:: text

   checkpoint tensors
       weight_packed / weight_scale / weight_shape
          │
          ▼
   process_weights_after_loading()
       gptq_marlin_repack()
       marlin_permute_scales()
       建好 runtime-only 的 zero / g_idx 占位
          │
          ▼
   apply()
       gptq_marlin_gemm()

``create_weights`` 注册和 checkpoint 对齐的参数名，让 safetensors 加载逻辑能按名字写进去。
``process_weights_after_loading`` 是 checkpoint layout 转 runtime kernel layout 的那条边界，
repack 只该放在这里，不该塞进通用权重加载器，更不该每次 forward 都做。

W8A8 INT8 路径
----------------------------------------

W8A8 面向 ``compressed-tensors`` 的 ``int-quantized`` checkpoint。当前的约束是：

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
- W8A8 CUTLASS 路径当前支持 Ampere / SM8x（SM80–SM89）。已验证目标是 Jetson Orin SM87；
  Hopper / SM90 暂不在支持范围内。

执行链路：

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

checkpoint 里的 INT8 权重通常是 ``[N, K]`` row-major。``process_weights_after_loading`` 会把它
转成 ``[K, N]`` column-major 视图并整理 ``weight_scale``，以满足 CUTLASS kernel 的接口约定。

LinearMethod 生命周期
----------------------------------------

每个 linear layer 都持有一个 ``quant_method``：

- 不量化时用 ``UnquantizedLinearMethod``，注册普通 ``weight`` 并调 ``F.linear``。
- 量化时由 ``QuantizationConfig.get_quant_method(layer, prefix)`` 返回具体方法。

典型生命周期：

1. 模型构造时，linear layer 调 ``quant_method.create_weights`` 注册参数。
2. ``model.load_weights`` 按参数名和 ``weight_loader`` 写进 checkpoint tensor。
3. 权重全部加载完，``ModelRunner`` 遍历模块调 ``process_weights_after_loading``。
4. forward 时 linear layer 委托 ``quant_method.apply`` 执行。

有了这条边界，新增量化方法时基本不用碰模型主逻辑，只要实现新的 config 和 scheme。

新增模型的建议流程
----------------------------------------

新增模型时建议按这个顺序来：

1. 在 ``pymllm/models/`` 加模型文件。
2. 在 ``pymllm/models/__init__.py`` 注册 HuggingFace architecture 字符串。
3. 实现最小 forward 接口：``forward(input_ids, positions, forward_batch)``。
4. 复用现有基础层，并确保 linear layer 接受 ``quant_method``。
5. 实现 ``load_weights``，处理好 checkpoint 前缀、stacked projection 和 tied embedding。
6. 补 registry、weight loading、forward timing 的单元测试。
7. 最后再做服务级 smoke test。

新增量化方法的建议流程
----------------------------------------

新增量化方法时保持三层结构：

1. ``QuantizationConfig``：解析 checkpoint 配置，决定某个 layer 是否量化。
2. ``LinearMethod``：承接 layer 生命周期。
3. ``Scheme``：处理具体格式的参数注册、post-load 转换和 kernel apply。

不要把 checkpoint 格式判断写进模型类，也不要把 runtime repack 藏在通用 ``weight_loader``
里。守住这条，模型结构、权重格式、kernel layout 三者的边界才不会糊在一起。
