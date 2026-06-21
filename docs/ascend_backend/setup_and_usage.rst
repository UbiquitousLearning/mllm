Ascend Setup and Usage
======================

总览
----

Ascend Backend 依赖 Ascend CANN、ATB 运行时和对应头文件/库文件。当前文档面向
``mllm/backends/ascend``、``examples/qwen_ascend`` 和 ``tests/ascend`` 的开发与验证流程。

环境要求
--------

运行或编译 Ascend Backend 前，需要确认：

- 已安装可用的 Ascend CANN 环境。
- 已安装 ATB，并能找到 ATB 头文件和库。
- 编译器、CMake、Python 依赖满足 mLLM 基础构建要求。
- 目标设备上可以正常访问 Ascend NPU。

使用 ACL 和 ATB 前，需要加载 Ascend Toolkit 和 ATB 的环境脚本：

.. code-block:: bash

   source <ASCEND_TOOLKIT_ROOT>/set_env.sh
   source <ATB_ROOT>/set_env.sh

其中 ``<ASCEND_TOOLKIT_ROOT>`` 是 Ascend Toolkit 的安装入口目录，``<ATB_ROOT>``
是 ATB 的安装入口目录。开发机上可以把这两行加入 ``~/.bashrc``，新开的交互式
shell 会自动加载；CI、非交互式脚本或新机器上仍建议在构建/运行前显式加载。

当前验证使用的版本：

.. list-table::
   :header-rows: 1

   * - 组件
     - 版本
   * - Ascend Toolkit / CANN
     - ``8.2.RC1``，组件版本号 ``8.2.0.0.201``
   * - ATB
     - ``8.2.RC1.B150``，平台 ``aarch64``

常用环境变量如下：

.. list-table::
   :header-rows: 1

   * - 变量
     - 说明
   * - ``ASCEND_HOME_PATH``
     - Ascend CANN 安装路径，测试目标会读取其 include/lib。
   * - ``ATB_HOME_PATH``
     - ATB 安装路径，测试目标会读取其 include/lib。

构建
----

当前 Ascend 开发路径使用仓库中的构建任务：

.. code-block:: bash

   pip install -r requirements.txt
   python task.py tasks/build_arm_ascend.yaml

该任务会打开 Ascend Backend，并同时启用 Arm Backend：

.. code-block:: text

   -DMLLM_BUILD_ASCEND_BACKEND=ON
   -DMLLM_BUILD_ARM_BACKEND=ON

如果手动配置 CMake，至少需要确保 ``MLLM_BUILD_ASCEND_BACKEND`` 被打开，并且 CANN/ATB
头文件与库路径能被 CMake 找到。

Qwen Ascend 示例
----------------

Qwen Ascend 示例目标为 ``mllm-qwen-ascend-runner``，入口文件位于
``examples/qwen_ascend/main.cpp``。示例会在加载权重前先执行 ``model.to(kAscend)``，
这样 Ascend Linear 算子可以直接读取模型文件中的 W8A8 量化参数。

默认模式是 QA generation：

.. code-block:: bash

   ./mllm-qwen-ascend-runner \
     -m /path/to/model.mllm \
     -c /path/to/config.json \
     -t /path/to/tokenizer.json \
     -p "请用一句话介绍你自己。" \
     -g 64

参数说明：

.. list-table::
   :header-rows: 1

   * - 参数
     - 说明
   * - ``-m`` / ``--model_path``
     - mLLM 模型文件路径。
   * - ``-c`` / ``--config_path``
     - Qwen Ascend 配置文件路径。
   * - ``-t`` / ``--tokenizer_path``
     - tokenizer json 路径，QA generation 必填。
   * - ``-p`` / ``--prompt``
     - 输入问题。
   * - ``-g`` / ``--max_new_tokens``
     - 最大生成 token 数。
   * - ``-mv`` / ``--model_version``
     - 模型文件版本，默认 ``v2``。

也可以运行 synthetic forward smoke test：

.. code-block:: bash

   ./mllm-qwen-ascend-runner \
     -m /path/to/model.mllm \
     -c /path/to/config.json \
     --forward_smoke_test \
     -s 8

该模式只构造简单 token 序列，适合快速确认模型 forward、权重加载和设备执行链路。

运行测试
--------

Ascend 单测目标为 ``Mllm-Test-AscendKernel``，测试代码位于 ``tests/ascend``。

测试覆盖内容包括：

- 基础 ATB 算子：Add、Sub、Mul、Linear、RMSNorm、Softmax、Transpose、Concat、Slice、SiLU。
- ACLNN 辅助算子：Cast、Abs、MaxDim、broadcast RealDiv/Mul。
- 模型相关算子：Embedding、RoPE、CausalMask、Attention、GQA。
- Graph 路径：GraphBuilder、Linear graph、Linear+Softmax graph、CausalMask plugin graph。
- W8A8 路径：activation quantization 到 ATB Linear W8A8 的端到端 pipeline。

常用开关
--------

.. list-table::
   :header-rows: 1

   * - 环境变量
     - 说明
   * - ``MLLM_ASCEND_QWEN_DECODER_GRAPH``
     - 控制 Qwen Ascend decoder graph，默认开启；设为 ``0`` 可回退 eager 路径。
   * - ``MLLM_ASCEND_QWEN_DECODER_GRAPH_SETUP_BUCKET``
     - decoder graph attention setup bucket 大小。
   * - ``MLLM_ASCEND_ENABLE_DYNAMIC_W8A8``
     - 打开 dynamic eager W8A8 调试路径，默认关闭。

注意事项
--------

- Dynamic W8A8 eager 路径仅用于精度和调试验证，不是默认推理路径。
- Decoder graph 是 Qwen Ascend 的默认优化路径；遇到 graph 问题时可通过
  ``MLLM_ASCEND_QWEN_DECODER_GRAPH=0`` 回退到 eager 路径定位。
