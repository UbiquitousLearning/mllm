# QNN Backend Qwen NPU Decoding 功能需求分析文档

## 1. 项目背景

### 1.1 目标
在 mllm_v2 框架上实现基于 QNN 加速的 Qwen3 4B 长文本推理功能。当前 QNN 已迁移到 v2 版本，但仅支持单个固定长度输入的 prefill 推理。需要实现自回归解码（decoding）功能，使模型能够连续生成文本。

### 1.2 当前状态
- **功能限制**：QNN 后端仅支持单 chunk 128 长度的 prefill 推理
- **问题现象**：`mllm-qwen-npu` 示例程序只输出单个 token 后即结束，无法进行连续生成
- **技术约束**：
  - QNN 端只允许单 chunk 128 长度
  - Decode 阶段的新 token 处理需要在 CPU 侧完成
  - QNN 负责输出 logits，CPU 负责采样和 token 管理

## 2. 功能需求

### 2.1 核心功能
在输入长度 < chunk_size（chunk_size = 128）的场景下，实现基于 QNN 后端的自回归解码：

1. **KV Cache 管理**
   - KV cache 默认长度为 1K（1024）
   - Prefill 阶段：real_seq 以内的真实输入 + (128 - real_seq) 的 padding
   - Decode 阶段：利用 padding 区域存放新生成的 token

2. **解码循环**
   - 循环调用 `forward` 生成下一个 token
   - 将新 token 写入 padding 区域（在 CPU buffer 中维护输入序列）
   - 累积 seq_len，直至满足终止条件

3. **终止条件**
   - 总长度达到 128（chunk_size）
   - 生成 EOS token（token ID: 151645）

### 2.2 预期效果
示例程序能够输出完整句子，而非仅单个 token。推理流程能够连续生成文本，直到达到最大长度或遇到 EOS token。

## 3. 技术实现方案

### 3.1 KV Cache 接口扩展

#### 3.1.1 接口设计原则
- **避免全局接口**：v1 中使用了大量全局接口，耦合性过大
- **在 modeling 中体现接口**：方便后续功能扩展
- **保持向后兼容**：确保新增接口不会破坏已有 trace/prefill 流程

#### 3.1.2 需要实现的接口层次

**层次 1：基类接口（aops::KVCacheOp）**
```cpp
// mllm/core/aops/KVCacheOp.hpp
class KVCacheOp : public BaseOp {
public:
  // 现有接口
  void setLayerIndex(int32_t layer_idx);
  virtual void clearCache();
  
  // 新增接口
  virtual void setCurrentSeqCnt(int32_t seq);
};
```

**层次 2：CPU 实现（CPUKVCacheOp）**
```cpp
// mllm/backends/cpu/ops/KVCacheOp.hpp
class CPUKVCacheOp final : public aops::KVCacheOp {
public:
  void setCurrentSeqCnt(int32_t seq) override;
  
private:
  nn::StaticCache cache_;  // 内部使用 StaticCache
};
```

**层次 3：Layer 接口（nn::KVCache）**
```cpp
// mllm/nn/layers/KVCache.hpp
class KVCache : public Layer {
public:
  void setCurrentSeqCnt(int32_t seq);
  // 现有接口：clearCache(), setLayerIndex()
};
```

**层次 4：Model 接口（QwenText/QwenForCausalLM）**
```cpp
// mllm/models/qwen_npu/modeling_qwen_npu.hpp
class QwenText : public nn::Module {
public:
  void setKVCacheSeqCnt(int32_t seq);  // 设置所有层的 KV cache 序列长度
  void clearKVCache();  // 现有接口
};

class QwenForCausalLM : public nn::Module, public ARGeneration {
public:
  void setKVCacheSeqCnt(int32_t seq);  // 委托给 model.setKVCacheSeqCnt()
};
```

#### 3.1.3 实现细节

**StaticCache::setCurrentSeqCnt 行为**
- 参考 `nn::StaticCache::setCurrentSeqCnt(int32_t seq)`
- 设置所有层的 `current_seq_cnt_[layer_idx] = seq`
- **关键**：不会覆盖已有 KV cache 数据，只是更新长度计数器

**CPUKVCacheOp::setCurrentSeqCnt 实现**
```cpp
void CPUKVCacheOp::setCurrentSeqCnt(int32_t seq) {
  cache_.setCurrentSeqCnt(seq);
}
```

**nn::KVCache::setCurrentSeqCnt 实现**
```cpp
void KVCache::setCurrentSeqCnt(int32_t seq) {
  std::static_pointer_cast<aops::KVCacheOp>(impl()->getInstancedOp())->setCurrentSeqCnt(seq);
}
```

### 3.2 解码循环实现

#### 3.2.1 在 main.cpp 中添加解码循环

**当前代码结构**（examples/qwen_npu/main.cpp）：
```cpp
// Prefill 阶段
auto out = model.forward(inputs, {{"seq_len", mllm::AnyValue((int)raw_input_tokens.shape()[1])}})["sequence"];
auto sampled = model.sampleGreedy(out);
std::wcout << "token: " << sampled << " " << qwen_tokenizer.detokenize(sampled) << "\n";
```

**需要添加的解码循环**（包含调试日志）：
```cpp
const int chunk_size = 128;
const int real_seq = raw_input_tokens.shape()[1];  // 实际输入长度
const int eos_token_id = 151645;

// Prefill 阶段（已有代码）
MLLM_INFO("=== Prefill Phase ===");
MLLM_INFO("Input sequence length: {}", real_seq);
auto prefill_output = model.forward(inputs, {{"seq_len", mllm::AnyValue(real_seq)}});
auto sampled = model.sampleGreedy(prefill_output["sequence"]);
MLLM_INFO("Prefill generated token: {} ({})", sampled, qwen_tokenizer.detokenize(sampled));
std::wcout << qwen_tokenizer.detokenize(sampled);

// 解码循环
int current_seq_len = real_seq;
auto& sequence_tensor = inputs["sequence"];
auto sequence_ptr = sequence_tensor.ptr<int64_t>();

// 将第一个生成的 token 写入 padding 区域
sequence_ptr[current_seq_len] = sampled;
current_seq_len++;

// 保存 prefill 返回的 position_ids，用于第一次 decode
ARGenerationOutputPast past = prefill_output;

MLLM_INFO("=== Decode Phase ===");
MLLM_INFO("Starting decode loop, initial seq_len: {}", current_seq_len);

// 循环生成直到达到 chunk_size 或遇到 EOS
int decode_step = 0;
while (current_seq_len < chunk_size) {
  decode_step++;
  MLLM_INFO("--- Decode Step {} ---", decode_step);
  MLLM_INFO("Current sequence length: {}", current_seq_len);
  
  // 更新 KV cache 序列长度
  model.setKVCacheSeqCnt(current_seq_len);
  
  // 验证 KV cache 状态（调试用）
  // 注意：需要通过 model.model 访问内部 KV cache
  // 这里假设可以通过某种方式访问，实际实现时可能需要添加辅助方法
  // MLLM_INFO("KV cache seq_cnt after update: {}", model.model.getKVCacheSeqCnt(0));
  
  // 准备输入：只包含当前要处理的 token（decode 阶段每次只处理 1 个 token）
  // 注意：需要传入上一次返回的 position_ids，forward 方法会自动递增
  auto decode_input = ARGenerationOutputPast{
    {"sequence", Tensor::empty({1, 1}, kInt64, kCPU).alloc()},
    {"position_ids", past["position_ids"]}  // 使用上一次返回的 position_ids
  };
  decode_input["sequence"].ptr<int64_t>()[0] = sequence_ptr[current_seq_len - 1];
  
  MLLM_INFO("Decode input token: {}", sequence_ptr[current_seq_len - 1]);
  
  // 调用 forward，传入当前序列长度
  // forward 方法会检测到 position_ids 存在且 seq_len == 1，自动递增位置
  auto decode_output = model.forward(decode_input, {{"seq_len", mllm::AnyValue(current_seq_len)}});
  
  // 采样下一个 token
  auto next_token = model.sampleGreedy(decode_output["sequence"]);
  MLLM_INFO("Generated token: {} ({})", next_token, qwen_tokenizer.detokenize(next_token));
  std::wcout << qwen_tokenizer.detokenize(next_token);
  
  // 检查终止条件
  if (next_token == eos_token_id) {
    MLLM_INFO("EOS token detected, stopping decode");
    break;
  }
  
  // 将新 token 写入序列
  sequence_ptr[current_seq_len] = next_token;
  current_seq_len++;
  MLLM_INFO("Updated sequence length: {}", current_seq_len);
  
  // 保存本次输出，用于下次循环（包含更新后的 position_ids）
  past = decode_output;
}

MLLM_INFO("=== Decode Complete ===");
MLLM_INFO("Total decode steps: {}", decode_step);
MLLM_INFO("Final sequence length: {}", current_seq_len);
MLLM_INFO("Remaining capacity: {}", chunk_size - current_seq_len);
std::wcout << "\n";
```

#### 3.2.2 关键实现细节

**输入序列管理**
- Prefill 阶段：使用完整的 128 长度 tensor，real_seq 之前是真实 token，之后是 padding（-1）
- Decode 阶段：每次 forward 只传入单个 token（形状 [1, 1]），但需要正确设置 seq_len 参数

**KV Cache 同步**
- 每次解码循环前调用 `model.setKVCacheSeqCnt(current_seq_len)`
- 确保 KV cache 知道当前已处理的序列长度
- 新 token 的 KV 会被追加到现有 cache 的末尾

**Position IDs 处理**
- Decode 阶段需要正确传递 position_ids
- 参考 `QwenForCausalLM::forward` 中的 position_ids 生成逻辑：
  - Prefill 阶段：自动生成 `[0, 1, 2, ..., seq_len-1]`
  - Decode 阶段：如果 input 中包含 position_ids，会自动递增最后一个位置
- 实现要点：
  - Prefill 返回的 output 中包含 position_ids
  - 第一次 decode 时，使用 prefill 返回的 position_ids
  - 后续 decode 时，使用上一次 forward 返回的 position_ids
  - forward 方法会自动检测 `seq_len == 1` 且存在 position_ids，然后递增位置

### 3.3 量化信息处理

#### 3.3.1 量化 Scale 的作用
- **GraphBuild 阶段**：量化 scale 用于构建 QNN 计算图
- **执行阶段**：量化 scale 仍然有效，但不会被使用（QNN 内部已固化）

#### 3.3.2 实现注意事项
- 量化 scale 只在 quantize 前需要显式 attach 到 input tensor
- 参考 `QNNCastTypeOp.cpp::QNNQuantizePattern` 中从输入 tensor 获取 quant scale 的操作
- 对于 scale 维持不变的算子（view, transpose），使用 `propagateQuantScale` 进行传递
- 对于 Linear 算子，scale 通过模型加载而来

**结论**：在 decode 循环中，不需要重新设置量化 scale，因为：
1. 量化参数已在 GraphBuild 时附加到 QNN tensor 中
2. 执行时 QNN 内部使用已固化的量化参数

## 4. 约束与关注点

### 4.1 技术约束
1. **QNN 限制**：QNN 端只允许单 chunk 128 长度；decode 只能在 CPU 侧处理新增 token
2. **KV Cache 管理**：需要确保新增的接口不会破坏已有 trace/prefill 流程
3. **内存管理**：避免在 decode 循环中创建新的 128 长度 KV cache，应复用现有 cache

### 4.2 实现注意事项
1. **Position IDs**：decode 阶段需要正确生成 position_ids，确保位置编码正确
2. **序列长度参数**：每次 forward 需要传入正确的 seq_len，告知模型当前实际序列长度
3. **Tensor 设备**：注意 QNN/CPU 之间的 tensor 转换，确保数据正确传递
4. **错误处理**：
   - 验证 `current_seq_len` 不超过 `chunk_size`（128）
   - 验证输入序列长度 `real_seq` 小于 `chunk_size`
   - 处理 `forward` 调用可能出现的异常
   - 验证 `setKVCacheSeqCnt` 的参数范围（0 <= seq <= chunk_size）
5. **边界情况**：
   - 输入长度为 0 或负数（应在调用前验证）
   - 输入长度等于或超过 chunk_size（应在调用前验证或拒绝）
   - KV cache 已满的情况（理论上不应发生，因为限制在 chunk_size 内）

### 4.3 调试与验证
1. **调试环境**：Android 设备、ADB
2. **验证方法**：
   - 检查输出是否连续生成多个 token
   - 验证是否在 EOS 或达到 128 长度时正确停止
   - 确认没有内存泄漏或崩溃

## 5. 待解决问题

### 5.1 Context 析构问题
**问题描述**：当前存在 SIGSEGV 崩溃，推测与析构顺序相关。

**解决方案**：需要在 context 析构中手动管理 backend 销毁顺序。

**实现位置**：待确认具体实现位置（可能在 QNNBackend 或 Context 相关代码中）

### 5.2 Position IDs 生成逻辑（已解决）
**解决方案**：decode 循环中需要显式传递 position_ids。

**实现方式**：
- Prefill 阶段返回的 output 中包含 position_ids
- 第一次 decode 时，使用 prefill 返回的 position_ids
- 后续 decode 时，使用上一次 forward 返回的 position_ids
- `QwenForCausalLM::forward` 方法会自动检测 `seq_len == 1` 且存在 position_ids，然后递增位置

**参考实现**：见 3.2.1 节解码循环代码示例。

## 6. 调试日志与测试验证

### 6.1 调试日志需求

为了验证 KV 缓存长度控制的正确性以及解码流程的正确性，需要在关键位置添加调试日志。

#### 6.1.1 日志位置与内容

**1. Prefill 阶段日志**
- 输入序列长度（real_seq）
- 生成的第一个 token ID 和文本

**2. Decode 循环日志（每次迭代）**
- 当前解码步骤编号
- 当前序列长度（current_seq_len）
- 输入 token ID（用于验证输入序列管理）
- 生成的 token ID 和文本
- KV cache 序列长度（验证 `setKVCacheSeqCnt` 是否正确设置）
- 终止原因（EOS 或达到最大长度）

**3. 解码完成日志**
- 总解码步数
- 最终序列长度
- 剩余容量（chunk_size - current_seq_len）

#### 6.1.2 日志实现方式

使用项目现有的日志宏 `MLLM_INFO`（定义在 `mllm/utils/Log.hpp`）：

```cpp
#include "mllm/utils/Log.hpp"

// 示例
MLLM_INFO("Current sequence length: {}", current_seq_len);
MLLM_INFO("Generated token: {} ({})", token_id, token_text);
```

**日志级别控制**：
- 默认日志级别为 `LogLevel::kInfo`，会显示所有 `MLLM_INFO` 日志
- 可以通过 `Logger::level()` 调整日志级别（如果需要减少日志输出）

#### 6.1.3 KV Cache 状态验证

为了验证 KV cache 状态，需要添加辅助方法获取当前序列长度：

**可选实现：在 Model 接口中添加查询方法**
```cpp
// mllm/models/qwen_npu/modeling_qwen_npu.hpp
class QwenText : public nn::Module {
public:
  void setKVCacheSeqCnt(int32_t seq);
  int32_t getKVCacheSeqCnt(int32_t layer_idx = 0) const;  // 新增：获取指定层的序列长度
  void clearKVCache();
};
```

**实现方式**：
```cpp
int32_t QwenText::getKVCacheSeqCnt(int32_t layer_idx) const {
  // 通过内部 KV cache 层获取序列长度
  // 需要访问 model 内部的 kv_cache_ 成员
  // 具体实现取决于内部结构
}
```

**注意**：如果添加查询方法比较复杂，也可以暂时在调试时通过其他方式验证（如直接访问内部 cache），或使用条件编译宏控制调试代码。

### 6.2 测试验证要点

#### 6.2.1 功能验证

1. **序列长度递增验证**
   - 验证 `current_seq_len` 从 `real_seq` 开始，每次循环递增 1
   - 验证最终长度不超过 `chunk_size`（128）

2. **KV Cache 同步验证**
   - 验证每次调用 `setKVCacheSeqCnt` 后，KV cache 的序列长度正确更新
   - 验证所有层的序列长度保持一致
   - 验证新 token 的 KV 被正确追加到 cache 末尾

3. **输入序列管理验证**
   - 验证新生成的 token 被正确写入 `sequence_tensor` 的 padding 区域
   - 验证每次 decode 时，输入 token 来自序列的正确位置（`sequence_ptr[current_seq_len - 1]`）

4. **终止条件验证**
   - 验证遇到 EOS token（151645）时正确停止
   - 验证达到 chunk_size（128）时正确停止
   - 验证终止后不再继续生成

5. **Position IDs 验证**
   - 验证 position_ids 在每次 decode 后正确递增
   - 验证 position_ids 与序列长度一致

#### 6.2.2 边界情况测试

1. **最小输入长度**
   - 测试 `real_seq = 1` 的情况
   - 验证能够正常进行 decode

2. **接近最大长度**
   - 测试 `real_seq = 127` 的情况（只能生成 1 个 token）
   - 验证在达到 128 时正确停止

3. **EOS 提前终止**
   - 测试在生成过程中遇到 EOS token
   - 验证提前终止后不再继续生成

4. **空输入处理**
   - 测试边界情况下的输入验证

#### 6.2.3 性能与稳定性验证

1. **内存泄漏检查**
   - 使用内存检测工具（如 Valgrind、AddressSanitizer）检查
   - 验证 decode 循环中不会创建不必要的临时对象

2. **崩溃检查**
   - 验证不会出现 SIGSEGV 或其他崩溃
   - 特别关注 Context 析构相关的崩溃（见 5.1 节）

3. **长时间运行稳定性**
   - 测试多次 decode 循环的稳定性
   - 验证 KV cache 不会溢出或损坏

### 6.3 调试日志示例输出

期望的日志输出格式：

```
[INFO] examples/qwen_npu/main.cpp:140 === Prefill Phase ===
[INFO] examples/qwen_npu/main.cpp:141 Input sequence length: 5
[INFO] examples/qwen_npu/main.cpp:144 Prefill generated token: 12345 (你好)
[INFO] examples/qwen_npu/main.cpp:156 === Decode Phase ===
[INFO] examples/qwen_npu/main.cpp:157 Starting decode loop, initial seq_len: 6
[INFO] examples/qwen_npu/main.cpp:162 --- Decode Step 1 ---
[INFO] examples/qwen_npu/main.cpp:163 Current sequence length: 6
[INFO] examples/qwen_npu/main.cpp:177 Decode input token: 12345
[INFO] examples/qwen_npu/main.cpp:186 Generated token: 67890 (世界)
[INFO] examples/qwen_npu/main.cpp:195 Updated sequence length: 7
[INFO] examples/qwen_npu/main.cpp:162 --- Decode Step 2 ---
[INFO] examples/qwen_npu/main.cpp:163 Current sequence length: 7
...
[INFO] examples/qwen_npu/main.cpp:200 === Decode Complete ===
[INFO] examples/qwen_npu/main.cpp:201 Total decode steps: 10
[INFO] examples/qwen_npu/main.cpp:202 Final sequence length: 15
[INFO] examples/qwen_npu/main.cpp:203 Remaining capacity: 113
```

### 6.4 调试日志的后续处理

**开发阶段**：
- 保留所有调试日志，便于问题定位和验证

**PR 提交阶段**：
- 根据项目规范，可以选择：
  - **方案 A**：保留日志，通过日志级别控制（推荐）
  - **方案 B**：注释掉调试日志，保留代码以便将来使用
  - **方案 C**：使用条件编译宏控制（如 `#ifdef MLLM_DEBUG_DECODING`）

**建议**：使用方案 A，通过日志级别控制。如果需要减少日志输出，可以在发布版本中设置更高的日志级别。

## 7. 实现步骤

### 阶段 1：KV Cache 接口扩展
1. 在 `aops::KVCacheOp` 中添加 `setCurrentSeqCnt` 虚方法
2. 在 `CPUKVCacheOp` 中实现该方法，调用 `cache_.setCurrentSeqCnt()`
3. 在 `nn::KVCache` 中添加 `setCurrentSeqCnt` 方法
4. 在 `QwenText` 和 `QwenForCausalLM` 中添加 `setKVCacheSeqCnt` 方法

### 阶段 2：解码循环实现与调试日志
1. 在 `main.cpp` 中添加解码循环代码（包含调试日志，见 3.2.1 节）
2. 实现输入序列管理（将新 token 写入 padding 区域）
3. 实现 KV cache 序列长度同步
4. 实现终止条件检查（EOS 或达到 128 长度）
5. 添加调试日志输出（见 6.1 节）

### 阶段 3：测试与验证
1. 编译并运行示例程序
2. 检查调试日志输出，验证 KV cache 状态和序列长度
3. 验证是否能够连续生成多个 token
4. 验证终止条件是否正确工作（EOS 和最大长度）
5. 验证边界情况（最小输入、接近最大长度等）
6. 检查是否有内存泄漏或崩溃
7. 验证长时间运行的稳定性

### 阶段 4：Context 析构修复（可选）
1. 定位 SIGSEGV 崩溃原因
2. 实现 backend 销毁顺序管理
3. 验证修复效果

## 8. 参考文档

- QNN Backend Design: `docs/qnn_backend/core_design.rst`
- QNN 量化文档: https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-10/quantization.html
- 模型量化基本概念: https://zhuanlan.zhihu.com/p/505570612

## 9. 相关代码文件

- `examples/qwen_npu/main.cpp` - 示例程序入口
- `mllm/backends/cpu/ops/KVCacheOp.{hpp,cpp}` - CPU KV Cache 操作实现
- `mllm/nn/layers/KVCache.{hpp,cpp}` - KV Cache Layer 接口
- `mllm/nn/lmcache/StaticCache.{hpp,cpp}` - 静态缓存实现（包含 `getCurrentSeqCnt` 方法）
- `mllm/models/qwen_npu/modeling_qwen_npu.hpp` - Qwen NPU 模型实现
- `mllm/core/aops/KVCacheOp.{hpp,cpp}` - KV Cache 操作基类
- `mllm/backends/qnn/op/QNNCastTypeOp.cpp` - QNN 量化实现参考
- `mllm/utils/Log.hpp` - 日志宏定义

