# Qwen NPU 示例代码详细解释

本文档逐行解释 `main.cpp` 中每行代码的具体逻辑，以及它们如何与其他文件结合。

## 1. 头文件包含（1-11行）

```cpp
#include <fmt/core.h>
#include <cstdint>
#include <mllm/mllm.hpp>
#include <mllm/utils/AnyValue.hpp>

#include "mllm/backends/qnn/passes/QNNGraphBuildPass.hpp"
#include "mllm/backends/qnn/passes/QNNGraphBuildPipeline.hpp"
#include "mllm/compile/PassManager.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/models/qwen_npu/tokenization_qwen.hpp"
#include "mllm/models/qwen_npu/modeling_qwen_npu.hpp"
```

### 详细说明：

- **`<fmt/core.h>`**: 格式化库，用于打印输出
- **`<cstdint>`**: 标准整数类型（int64_t等）
- **`<mllm/mllm.hpp>`**: MLLM核心库，包含：
  - `MLLM_MAIN` 宏定义（在 `mllm/mllm.hpp:401-412`）
  - `Tensor` 类
  - `Context` 管理
  
  - 信号处理等基础设施
- **`<mllm/utils/AnyValue.hpp>`**: 类型擦除的值容器，用于传递任意类型的参数
- **QNN相关头文件**: QNN（Qualcomm Neural Network）后端相关的Pass
- **模型相关头文件**: Qwen NPU模型的实现和分词器

## 2. MLLM_MAIN 宏（15行）

```cpp
MLLM_MAIN({
```

### 详细说明：

`MLLM_MAIN` 宏定义在 `mllm/mllm.hpp:401-412`：

```cpp
#define MLLM_MAIN(...)                                     \
  int main(int argc, char** argv) {                        \
    ::mllm::__setup_signal_handler();                      \
    ::mllm::initializeContext();                           \
    auto user_main = [&]() -> int {                        \
      __VA_ARGS__;                                         \
      return 0;                                            \
    };                                                     \
    int result = ::mllm::__mllm_exception_main(user_main); \
    ::mllm::shutdownContext();                             \
    return result;                                         \
  }
```

**展开后的逻辑：**
1. 设置信号处理器（SIGINT, SIGTERM等）
2. 初始化MLLM上下文（内存管理器、设备等）
3. 将用户代码包装在lambda中
4. 在异常处理中执行用户代码
5. 程序结束时清理上下文

## 3. 初始化QNN后端（16行）

```cpp
  mllm::initQnnBackend();
```

### 详细说明：

- **位置**: `mllm/backends/qnn/Register.cpp:18`
- **作用**: 
  - 注册QNN后端操作（ops）
  - 初始化QNN运行时环境
  - 使QNN相关的操作可以在MLLM中使用

## 4. 配置路径定义（18-19行）

```cpp
  const std::string config_path = "./config_1.8B_w8a16_qnn.json";
  const std::string model_path = "./qwen1.5-1.8b-chat-rot-qnn.mllm";
```

### 详细说明：

- **config_path**: 模型配置文件，包含：
  - 模型架构参数（hidden_size, num_layers等）
  - 量化配置（w8a16表示8bit权重，16bit激活）
  - QNN特定配置
- **model_path**: 模型权重文件（.mllm格式）

## 5. 创建分词器（21行）

```cpp
  auto qwen_tokenizer = mllm::models::qwen_npu::QwenTokenizer("./tokenizer.json", "./qwen_merges.txt");
```

### 详细说明：

- **位置**: `mllm/models/qwen_npu/tokenization_qwen.hpp`
- **构造函数参数**:
  - `tokenizer.json`: BPE（Byte Pair Encoding）词汇表
  - `qwen_merges.txt`: BPE合并规则
- **作用**: 将文本转换为token IDs，或将token IDs转换回文本

### 内部实现（tokenization_qwen.hpp:285-309）：

```cpp
ARGenerationOutputPast convertMessage(const QwenMessage& message) {
  // 1. 应用消息模板
  auto applied_string = QwenMessage::message_template;
  size_t pos = applied_string.find("{{{prompt}}}");
  applied_string.replace(pos, 12, message.prompt);
  
  // 2. 分词
  auto sequence_str = tokenize(applied_string);
  
  // 3. 查找词汇表，转换为ID
  std::vector<int64_t> ids;
  for (const auto& str : sequence_str) { 
    ids.emplace_back(bpe_._lookup_vocab(str)); 
  }
  
  // 4. 创建Tensor
  Tensor sequence = Tensor::empty({1, (int32_t)ids.size()}, kInt64, kCPU)
                      .alloc();
  auto ptr = sequence.ptr<int64_t>();
  for (size_t i = 0; i < ids.size(); ++i) { ptr[i] = ids[i]; }
  
  return {{"sequence", sequence}};
}
```

## 6. 模型文件版本（23行）

```cpp
  mllm::ModelFileVersion file_version = mllm::ModelFileVersion::kV1;
```

### 详细说明：

- 指定模型文件格式版本
- 不同版本可能有不同的序列化格式
- 用于 `mllm::load()` 函数正确解析模型文件

## 7. 创建模型配置和实例（25-26行）

```cpp
  auto cfg = mllm::models::qwen_npu::QwenNPUConfig(config_path);
  auto model = mllm::models::qwen_npu::QwenForCausalLM("", cfg);
```

### 详细说明：

#### 7.1 QwenNPUConfig（25行）

- **位置**: `mllm/models/qwen_npu/configuration_qwen_npu.hpp`
- **作用**: 从JSON配置文件加载模型参数
- **包含的参数**:
  - `vocab_size`: 词汇表大小
  - `hidden_size`: 隐藏层维度
  - `num_attention_heads`: 注意力头数
  - `num_key_value_heads`: KV缓存头数（GQA）
  - `num_hidden_layers`: Transformer层数
  - `intermediate_size`: MLP中间层维度
  - `max_position_embeddings`: 最大位置编码
  - `rope_theta`: RoPE旋转角度
  - `linear_impl_type`: 线性层实现类型（QNN特定）

#### 7.2 QwenForCausalLM（26行）

- **位置**: `mllm/models/qwen_npu/modeling_qwen_npu.hpp:445-454`
- **继承关系**:
  - `nn::Module`: 神经网络模块基类
  - `ARGeneration`: 自回归生成接口
- **构造函数逻辑**:

```cpp
explicit QwenForCausalLM(const std::string& name, const QwenNPUConfig& cfg) 
    : cfg(cfg), nn::Module(name) {
  // 注册主模型（Transformer堆叠）
  model = reg<QwenText>("model", cfg);
  
  // 注册语言模型头（如果未共享权重）
  if (!cfg.tie_word_embeddings) {
    lm_head_ = reg<nn::Linear>("lm_head", cfg.hidden_size, cfg.vocab_size, 
                               false, cfg.linear_impl_type);
  }
  tie_word_embeddings_ = cfg.tie_word_embeddings;
}
```

**`reg<>()` 函数**:
- 注册子模块到当前模块
- 返回子模块的引用
- 子模块会被添加到模块树中，用于参数加载和计算图构建

## 8. 加载模型参数（28-29行）

```cpp
  auto param = mllm::load(model_path, file_version);
  model.load(param);
```

### 详细说明：

#### 8.1 mllm::load()（28行）

- **作用**: 从.mllm文件加载参数
- **返回**: `ParameterFile::ptr_t`，包含所有模型权重
- **内部流程**:
  1. 打开模型文件
  2. 根据file_version解析文件格式
  3. 读取所有张量数据（权重、偏置等）
  4. 返回参数容器

#### 8.2 model.load()（29行）

- **位置**: `nn::Module::load()`（继承自Module基类）
- **作用**: 将参数加载到模型结构中
- **匹配逻辑**:
  - 根据模块名称匹配参数
  - 递归加载子模块参数
  - 将权重张量复制到对应的模块中

## 9. 创建Trace输入占位符（31行）

```cpp
  mllm::models::ARGenerationOutputPast inputs{{"sequence", mllm::Tensor::empty({1, 32}, mllm::kInt64, mllm::kCPU).alloc()}};
```

### 详细说明：

#### 9.1 ARGenerationOutputPast

- **定义**: `mllm/models/ARGeneration.hpp:17`
```cpp
using ARGenerationOutputPast = std::unordered_map<std::string, Tensor>;
```
- **作用**: 模型输入/输出的键值对容器
- **常用键**:
  - `"sequence"`: 输入token序列
  - `"position_ids"`: 位置编码（可选）
  - 其他模型特定的输入

#### 9.2 Tensor::empty()

- **位置**: `mllm/core/Tensor.cpp:70-74`
```cpp
static Tensor empty(const std::vector<int32_t>& shape, DataTypes dtype, DeviceTypes device) {
  auto storage = TensorStorage::create(shape, dtype, device);
  auto impl = TensorViewImpl::create(shape, storage);
  return Tensor(impl);
}
```
- **参数**:
  - `{1, 32}`: shape，batch_size=1, seq_len=32
  - `kInt64`: 数据类型，64位整数
  - `kCPU`: 设备类型，CPU内存
- **注意**: `empty()` **不分配内存**，只创建Tensor对象

#### 9.3 .alloc()

- **位置**: `mllm/core/Tensor.cpp:63-66`
```cpp
Tensor& alloc() {
  Context::instance().memoryManager()->alloc(impl_->storage());
  return *this;
}
```
- **作用**: 
  - 通过内存管理器分配实际内存
  - 返回Tensor引用（支持链式调用）
- **内存布局**: 分配 `1 * 32 * sizeof(int64_t) = 256` 字节

## 10. Trace构建计算图（33行）

```cpp
  auto irs = model.trace(inputs, {});
```

### 详细说明：

#### 10.1 trace()方法

- **位置**: `mllm/models/qwen_npu/modeling_qwen_npu.hpp:514-563`
- **作用**: 构建计算图的中间表示（IR）
- **输入**:
  - `inputs`: 占位符输入（用于确定形状）
  - `args`: 额外参数（这里为空）

#### 10.2 trace()内部流程（实际实现）：

```cpp
IROutput trace(const ARGenerationOutputPast& input, const ARGenerationArgs& args) override {
  ir::IRContext::ptr_t llm_ir = nullptr;
  
  // 1. 开始trace：启用操作记录模式
  ir::lowlevel::traceStart();
  
  // 2. 获取输入序列
  auto sequence = input.at("sequence");
  
  // 3. Trace embedding层
  // 在trace模式下，embedding操作会被记录到IR中
  auto input_embeddings = model.embedding_(sequence);
  
  // 4. 暂停trace：停止记录操作
  // 接下来的操作（如创建position_ids）不需要被trace
  ir::lowlevel::traceYield();
  
  // 5. 准备RoPE嵌入（不在trace中）
  auto batch_size = sequence.shape()[0];
  auto seq_len = sequence.shape()[1];
  auto position_ids = Tensor::empty({batch_size, seq_len}, kInt64, kCPU);
  auto llm_embedding_sin = Tensor::empty({...}, kFloat32, kCPU);
  auto llm_embedding_cos = Tensor::empty({...}, kFloat32, kCPU);
  
  // 6. 继续trace：恢复操作记录
  ir::lowlevel::traceContinue();
  
  // 7. Trace模型主体（Transformer层）
  // traceModule会记录整个模块的计算图
  auto hidden_states = ir::lowlevel::traceModule(
      model, input_embeddings, llm_embedding_sin, llm_embedding_cos)[0];
  
  // 8. 截取最后一个位置
  auto S = hidden_states.shape()[1];
  hidden_states = hidden_states[{kAll, {S - 1}, kAll}];
  
  // 9. Trace语言模型头
  Tensor logits;
  if (!tie_word_embeddings_) { 
    logits = lm_head_(hidden_states); 
  }
  
  // 10. 停止trace并获取IR
  llm_ir = ir::lowlevel::traceStop();
  
  return {{"model", llm_ir}};
}
```

#### 10.2.1 trace函数说明：

- **`traceStart()`**: 
  - 启用trace模式
  - 后续的操作会被记录到IR中
  - 操作不会实际执行，只记录计算图结构

- **`traceYield()`**: 
  - 暂停trace
  - 接下来的操作不会被记录
  - 用于执行一些辅助操作（如创建辅助张量）

- **`traceContinue()`**: 
  - 恢复trace
  - 继续记录操作

- **`traceModule()`**: 
  - 专门用于trace模块（Module）
  - 会递归trace模块的所有子操作
  - 返回模块的输出

- **`traceStop()`**: 
  - 停止trace
  - 构建最终的IR
  - 返回IRContext指针

#### 10.3 IR（Intermediate Representation）

- **结构**: 计算图的节点和边
- **节点**: 操作（Op），如MatMul、Add、Softmax等
- **边**: 张量（Tensor）的流动
- **用途**: 
  - 图优化
  - 后端代码生成
  - 静态分析

#### 10.4 返回值

- **类型**: `IROutput = std::unordered_map<std::string, ir::IRContext::ptr_t>`
- **内容**: `{{"model", ir_context}}`
- **注意**: 此时KV Cache可能被更新，需要后续清理

## 11. QNN Graph Rewrite Pass（35-39行）

```cpp
  // QNN Graph Rewrite Pass
  mllm::ir::PassManager rewritePM(irs["model"]);
  rewritePM.reg(mllm::qnn::createQNNGraphIOTensorPass());
  rewritePM.reg(mllm::qnn::createQNNOpNamingPass());
  rewritePM.run();
```

### 详细说明：

#### 11.1 PassManager

- **位置**: `mllm/compile/PassManager.hpp`
- **作用**: 管理IR转换Pass的执行
- **构造函数**: 接收IR上下文

#### 11.2 Pass注册和执行

- **`reg()`**: 注册Pass到执行队列
- **`run()`**: 按顺序执行所有Pass

#### 11.3 QNNGraphIOTensorPass

- **作用**: 
  - 识别输入/输出张量
  - 为QNN图准备IO张量
  - 处理形状信息

#### 11.4 QNNOpNamingPass

- **作用**:
  - 为QNN操作生成唯一名称
  - 确保操作名称符合QNN要求
  - 便于调试和日志记录

## 12. 输出IR到文件（42行）

```cpp
  mllm::redirect("qwen_npu.mir", [&]() { mllm::print(irs["model"]); });
```

### 详细说明：

- **`mllm::redirect()`**: 重定向输出到文件
- **`mllm::print()`**: 打印IR的文本表示
- **用途**: 调试，查看优化后的计算图结构
- **文件内容**: MIR（MLLM IR）格式的计算图

## 13. QNN Graph Build Pass（44-47行）

```cpp
  // QNN Graph Build Pass
  mllm::ir::PassManager graphBuildPM(irs["model"]);
  graphBuildPM.reg(mllm::qnn::createQNNGraphBuildPass());
  graphBuildPM.run();
```

### 详细说明：

#### 13.1 QNNGraphBuildPass

- **位置**: `mllm/backends/qnn/passes/QNNGraphBuildPass.hpp`
- **作用**: 将MLLM IR转换为QNN图
- **转换过程**:
  1. **遍历IR节点**: 访问计算图中的每个操作节点
  2. **操作映射**: 将MLLM操作映射到QNN操作
     - MatMul → QNN MatMul
     - Add → QNN ElementWiseAdd
     - Softmax → QNN Softmax
     - 等等
  3. **创建QNN图**: 使用QNN API创建图结构
  4. **图优化**: 
     - 操作融合（如MatMul+Add → FullyConnected）
     - 量化处理（w8a16量化）
     - 内存优化
  5. **编译图**: 编译为QNN可执行图
- **结果**: 
  - 模型可以在QNN运行时执行
  - 图被编译并优化，准备在NPU上运行
  - 后续forward()调用会使用这个编译好的图

## 14. 清空KV Cache（50行）

```cpp
  // cache has been updated due to trace, clear cache
  model.model.clearKVCache();
```

### 详细说明：

- **原因**: trace过程中可能执行了前向传播，更新了KV Cache
- **作用**: 清空所有层的KV Cache，准备新的推理
- **位置**: `QwenText::clearKVCache()`，递归清空所有注意力层的缓存

## 15. 分词输入文本（52-53行）

```cpp
  auto raw_input_tokens = qwen_tokenizer.convertMessage({.prompt = "How are you?"})["sequence"];
  print(raw_input_tokens);
```

### 详细说明：

#### 15.1 convertMessage()

- **输入**: `QwenMessage` 结构，包含 `prompt` 字段
- **处理流程**:
  1. 应用消息模板（添加系统提示等）
  2. BPE分词
  3. 词汇表查找，转换为token IDs
  4. 创建Tensor并返回

#### 15.2 返回值

- **类型**: `ARGenerationOutputPast`
- **内容**: `{{"sequence", Tensor}}`
- **Tensor形状**: `[1, token_count]`，例如 `[1, 15]`

#### 15.3 print()

- **作用**: 打印Tensor内容（用于调试）
- **输出**: token IDs数组

## 16. 手动填充输入（55-59行）

```cpp
  // manually set input data as fill op is not supported in QNN
  auto ptr = inputs["sequence"].ptr<int64_t>();
  auto input_data = raw_input_tokens.ptr<int64_t>();
  for (int i = 0; i < raw_input_tokens.shape()[1]; ++i) { ptr[i] = input_data[i]; }
  for (int i = raw_input_tokens.shape()[1]; i < 32; ++i) { ptr[i] = -1; }
```

### 详细说明：

#### 16.1 为什么手动填充？

- **原因**: QNN后端不支持Fill操作
- **解决方案**: 在CPU上手动填充，然后传递给QNN

#### 16.2 ptr<int64_t>()

- **位置**: `mllm/core/Tensor.hpp`
- **作用**: 获取张量的原始指针
- **类型**: `int64_t*`
- **注意**: 必须确保张量已分配内存（已调用alloc()）

#### 16.3 填充逻辑

```cpp
// 1. 复制有效token
for (int i = 0; i < raw_input_tokens.shape()[1]; ++i) { 
  ptr[i] = input_data[i]; 
}

// 2. 填充padding（-1表示无效位置）
for (int i = raw_input_tokens.shape()[1]; i < 32; ++i) { 
  ptr[i] = -1; 
}
```

**结果**:
- 前15个位置：有效token IDs
- 后17个位置：-1（padding）

## 17. 前向推理（61行）

```cpp
  auto out = model.forward(inputs, {{"seq_len", mllm::AnyValue((int)raw_input_tokens.shape()[1])}})["sequence"];
```

### 详细说明：

#### 17.1 forward()方法

- **位置**: `mllm/models/qwen_npu/modeling_qwen_npu.hpp:456-512`
- **签名**: 
```cpp
ARGenerationOutputPast forward(
    const ARGenerationOutputPast& input, 
    const ARGenerationArgs& args
) override
```

#### 17.2 forward()内部流程：

```cpp
ARGenerationOutputPast forward(...) {
  // 1. 获取输入序列
  auto sequence = input.at("sequence");
  auto batch_size = sequence.shape()[0];  // 1
  auto seq_len = sequence.shape()[1];     // 32
  
  // 2. 获取真实序列长度
  auto real_seq = args.count("seq_len") 
      ? args.at("seq_len").get<int>()     // 15
      : seq_len;                          // 32（fallback）
  
  // 3. 生成位置编码
  Tensor position_ids = Tensor::empty({batch_size, seq_len}, kInt64, kCPU).alloc();
  // 填充 [0, 1, 2, ..., 31]
  
  // 4. 生成RoPE嵌入
  auto [llm_embedding_sin, llm_embedding_cos] = 
      makeRotaryPosEmbedding(position_ids, model.getBuffer("inv_freq"), 1.0f);
  
  // 5. 文本嵌入
  auto input_embeddings = model.embedding_(sequence);
  // shape: [1, 32, hidden_size]
  // 
  // 注意：QwenText中的embedding使用了QNN版本
  // (modeling_qwen_npu.hpp:415: embedding_.to(kQNN))
  // QNN版本的embedding会特殊处理padding token（-1）：
  // - 可能映射到特殊的embedding向量
  // - 或返回零向量
  // - 确保padding位置不影响计算
  
  // 6. Transformer前向传播
  auto hidden_states = model(input_embeddings, llm_embedding_sin, llm_embedding_cos)[0];
  // shape: [1, 32, hidden_size]
  
  // 7. 截取有效部分
  hidden_states = hidden_states[{kAll, {real_seq - 1}, kAll}];
  // shape: [1, 1, hidden_size]（只取最后一个有效位置）
  
  // 8. 语言模型头
  Tensor logits;
  if (!tie_word_embeddings_) {
    logits = lm_head_(hidden_states);
  } else {
    // 共享权重：使用embedding权重
    auto emb_w = model.embedding_.weight();
    logits = nn::functional::matmul(hidden_states, emb_w, false, true);
  }
  // shape: [1, 1, vocab_size]
  
  return {
      {"sequence", logits},
      {"position_ids", position_ids}
  };
}
```

#### 17.3 关键点：

1. **real_seq参数**: 告知模型真实序列长度，用于：
   - 截取输出（只取最后一个有效位置）
   - 可能影响attention mask（虽然当前实现可能未完全处理）

2. **位置编码**: 生成 `[0, 1, 2, ..., 31]`，即使有padding

3. **Embedding处理**: 
   - token ID `-1` 可能被映射为特殊embedding
   - 或返回零向量

4. **输出截取**: 
   ```cpp
   hidden_states[{kAll, {real_seq - 1}, kAll}]
   ```
   - `kAll`: 保留batch维度
   - `{real_seq - 1}`: 只取第real_seq-1个位置（最后一个有效位置）
   - `kAll`: 保留hidden维度

#### 17.4 返回值

- **类型**: `ARGenerationOutputPast`
- **内容**: 
  - `"sequence"`: logits，形状 `[1, 1, vocab_size]`
  - `"position_ids"`: 位置编码，形状 `[1, 32]`

## 18. 采样（63行）

```cpp
  auto sampled = model.sampleGreedy(out);
```

### 详细说明：

#### 18.1 sampleGreedy()

- **位置**: `mllm/models/ARGeneration.cpp`
- **作用**: 贪心采样，选择概率最高的token
- **实现**:

```cpp
int64_t ARGeneration::sampleGreedy(Tensor& logits) {
  // 1. 获取最后一个位置的logits
  auto last_logits = getLastLogits(logits);
  // shape: [vocab_size]
  
  // 2. 找到最大值索引
  int64_t max_idx = 0;
  float max_val = last_logits.ptr<float>()[0];
  for (int i = 1; i < vocab_size; ++i) {
    if (last_logits.ptr<float>()[i] > max_val) {
      max_val = last_logits.ptr<float>()[i];
      max_idx = i;
    }
  }
  
  return max_idx;
}
```

- **输入**: logits，形状 `[1, 1, vocab_size]`
- **输出**: token ID（int64_t）

## 19. 输出结果（64行）

```cpp
  std::wcout << "token: " << sampled << " " << qwen_tokenizer.detokenize(sampled) << "\n";
```

### 详细说明：

#### 19.1 detokenize()

- **位置**: `mllm/models/qwen_npu/tokenization_qwen.hpp`
- **作用**: 将token ID转换回文本
- **流程**:
  1. 查找词汇表，获取token字符串
  2. 合并BPE tokens
  3. 解码为UTF-8文本

#### 19.2 输出

- **格式**: `token: <ID> <text>`
- **示例**: `token: 1234 I'm`

## 20. 返回（66行）

```cpp
  return 0;
```

### 详细说明：

- 返回0表示程序成功执行
- `MLLM_MAIN`宏会捕获返回值并传递给系统

## 数据流总结

### 完整执行流程

```
┌─────────────────────────────────────────────────────────────┐
│ 1. 初始化阶段                                                │
├─────────────────────────────────────────────────────────────┤
│ initQnnBackend()                                             │
│   └─> 注册QNN后端操作                                        │
│   └─> 初始化QNN运行时环境                                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. 模型加载阶段                                              │
├─────────────────────────────────────────────────────────────┤
│ QwenNPUConfig(config_path)                                  │
│   └─> 从JSON读取模型配置                                     │
│   └─> 解析架构参数（hidden_size, num_layers等）             │
│   └─> 解析量化配置（w8a16）                                  │
│                                                              │
│ QwenForCausalLM("", cfg)                                     │
│   └─> 创建模型结构                                           │
│   └─> 注册子模块（QwenText, lm_head等）                     │
│                                                              │
│ mllm::load(model_path, file_version)                        │
│   └─> 打开.mllm文件                                         │
│   └─> 读取所有权重张量                                      │
│   └─> 返回ParameterFile                                     │
│                                                              │
│ model.load(param)                                            │
│   └─> 递归加载子模块参数                                     │
│   └─> 将权重复制到对应模块                                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. 图构建阶段（Trace）                                       │
├─────────────────────────────────────────────────────────────┤
│ inputs = {{"sequence", Tensor::empty({1, 32}, ...)}}        │
│   └─> 创建占位符输入（用于确定形状）                         │
│                                                              │
│ model.trace(inputs, {})                                     │
│   ├─> traceStart()                                          │
│   ├─> model.embedding_(sequence)  [记录到IR]                │
│   ├─> traceYield()                                          │
│   ├─> 创建position_ids, RoPE嵌入  [不记录]                  │
│   ├─> traceContinue()                                       │
│   ├─> traceModule(model, ...)  [记录整个模型]                │
│   ├─> lm_head_(hidden_states)  [记录到IR]                   │
│   └─> traceStop() → 返回IR                                  │
│                                                              │
│ PassManager: QNNGraphIOTensorPass                           │
│   └─> 识别输入/输出张量                                     │
│   └─> 为QNN图准备IO张量                                     │
│                                                              │
│ PassManager: QNNOpNamingPass                                │
│   └─> 为QNN操作生成唯一名称                                 │
│                                                              │
│ PassManager: QNNGraphBuildPass                              │
│   └─> 将MLLM IR转换为QNN图                                 │
│   └─> 操作映射（MatMul → QNN MatMul）                       │
│   └─> 图优化（融合、量化）                                   │
│   └─> 编译为QNN可执行图                                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. 推理准备阶段                                              │
├─────────────────────────────────────────────────────────────┤
│ model.model.clearKVCache()                                  │
│   └─> 清空所有注意力层的KV Cache                            │
│                                                              │
│ qwen_tokenizer.convertMessage({.prompt = "How are you?"})  │
│   ├─> 应用消息模板                                          │
│   ├─> BPE分词                                               │
│   ├─> 词汇表查找 → token IDs                                │
│   └─> 创建Tensor: [1, 15]                                  │
│                                                              │
│ 手动填充输入                                                 │
│   ├─> 复制有效token (0-14)                                  │
│   └─> 填充padding (-1) (15-31)                             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. 前向推理阶段                                              │
├─────────────────────────────────────────────────────────────┤
│ model.forward(inputs, {{"seq_len", 15}})                    │
│   ├─> 获取输入序列 [1, 32]                                  │
│   ├─> 生成position_ids [0, 1, 2, ..., 31]                  │
│   ├─> 生成RoPE嵌入 (sin, cos)                               │
│   ├─> model.embedding_(sequence)                            │
│   │   └─> QNN版本处理padding token (-1)                     │
│   │   └─> 输出: [1, 32, hidden_size]                       │
│   ├─> model(input_embeddings, sin, cos)                     │
│   │   ├─> 遍历所有Transformer层                             │
│   │   ├─> 每层: Attention + MLP                            │
│   │   └─> 输出: [1, 32, hidden_size]                        │
│   ├─> 截取最后一个有效位置                                  │
│   │   └─> hidden_states[{kAll, {14}, kAll}]                │
│   │   └─> 输出: [1, 1, hidden_size]                        │
│   ├─> lm_head_(hidden_states)                               │
│   │   └─> 输出: [1, 1, vocab_size]                         │
│   └─> 返回: {{"sequence", logits}, {"position_ids", ...}}  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. 采样和输出阶段                                            │
├─────────────────────────────────────────────────────────────┤
│ model.sampleGreedy(out)                                     │
│   ├─> 获取最后一个位置的logits                              │
│   ├─> 找到最大值索引                                         │
│   └─> 返回token ID (int64_t)                                │
│                                                              │
│ qwen_tokenizer.detokenize(sampled)                          │
│   ├─> 查找词汇表                                             │
│   ├─> 合并BPE tokens                                        │
│   └─> 解码为UTF-8文本                                       │
└─────────────────────────────────────────────────────────────┘
```

### 关键数据流

```
文本输入: "How are you?"
    ↓
分词器: convertMessage()
    ↓
Token IDs: [1234, 5678, 9012, ...]  (15个tokens)
    ↓
填充: [1234, 5678, ..., -1, -1, ...]  (32个位置)
    ↓
Embedding: [1, 32, hidden_size]
    ↓
Transformer层 × N: [1, 32, hidden_size]
    ↓
截取: [1, 1, hidden_size]  (只取最后一个有效位置)
    ↓
LM Head: [1, 1, vocab_size]
    ↓
采样: token ID (int64_t)
    ↓
Detokenize: "I'm"
```

## 关键数据结构

### Tensor
- **位置**: `mllm/core/Tensor.hpp`
- **组成**:
  - `TensorViewImpl`: 视图实现（形状、步长）
  - `TensorStorage`: 存储（实际数据）
- **生命周期**:
  1. `Tensor::empty()`: 创建对象（无内存）
  2. `.alloc()`: 分配内存
  3. 使用指针操作数据
  4. 自动析构释放内存
- **设备管理**:
  - 默认在CPU上创建
  - 可以通过`.to(device)`转换设备
  - QNN后端会自动处理设备转换

### 设备类型
- **`kCPU`**: CPU内存，用于：
  - 输入/输出张量
  - 辅助计算（如position_ids）
  - 不支持QNN的操作
- **`kQNN`**: QNN设备，用于：
  - 模型权重
  - 主要计算（在NPU上执行）
  - 需要QNN支持的操作
- **设备转换**:
  - 自动转换：操作会自动将输入转换到正确设备
  - 手动转换：`.to(device)`显式转换
  - 注意：QNN操作要求输入在QNN设备上

### ARGenerationOutputPast
- **类型**: `std::unordered_map<std::string, Tensor>`
- **用途**: 模型输入/输出的统一接口
- **键**: 模型特定的字符串标识符

### IRContext
- **位置**: `mllm/compile/ir/`
- **组成**: 计算图的节点和边
- **用途**: 图优化和代码生成

## 与其他文件的连接

1. **模型定义**: `modeling_qwen_npu.hpp`
   - `QwenForCausalLM`: 主模型类（继承ARGeneration）
     - `forward()`: 前向传播实现
     - `trace()`: 图构建实现
   - `QwenText`: Transformer堆叠（modeling_qwen_npu.hpp:403）
     - `decode_blocks_`: 多层QwenDecoder
     - `norm_`: RMSNorm层
     - `embedding_`: 词嵌入层（使用QNN版本处理padding）
     - `forward()`: 执行所有Transformer层
   - `QwenDecoder`: 单个Transformer层
     - 包含注意力层和MLP层
   - `QwenAttentionMatmul`: 注意力层
     - 包含QKV投影、RoPE、CausalMask、Softmax等
   - `QwenMLP`: MLP层
     - Gate、Up、Down投影，SiLU激活

2. **分词器**: `tokenization_qwen.hpp`
   - `QwenTokenizer`: 分词器类
   - BPE实现

3. **后端**: `mllm/backends/qnn/`
   - QNN操作实现
   - Pass实现
   - 运行时集成

4. **核心**: `mllm/core/`
   - `Tensor`: 张量实现
   - `Context`: 全局上下文
   - `MemoryManager`: 内存管理

5. **编译**: `mllm/compile/`
   - `PassManager`: Pass管理
   - `ir/`: IR定义和操作

