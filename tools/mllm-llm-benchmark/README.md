# MLLM LLM Benchmark Tool

## 概述

这是一个用于测试 MLLM 模型性能的基准测试工具，可以测量：
- **TTFT (Time To First Token)**: 首 token 延迟
- **Prefill Speed**: 预填充速度 (tokens/s)
- **Decode Speed**: 解码生成速度 (tokens/s)

## 编译

在 mllm_v2 项目根目录下编译：

```bash
mkdir -p build && cd build
cmake ..
make mllm-llm-benchmark
```

## 使用方法

### 基本用法

```bash
./mllm-llm-benchmark \
  -n qwen3-w4a32-kai \
  -m /path/to/model.mllm \
  -c /path/to/config.json \
  -t 4 \
  -pp 64,128,256 \
  -tg 100,200,300 \
  -cl 2048
```

### 参数说明

| 参数 | 长格式 | 说明 | 示例 |
|------|--------|------|------|
| `-n` | `--model_name` | 模型名称（用于选择正确的 benchmark 实现） | `qwen3-w4a32-kai` |
| `-m` | `--model_path` | 模型权重文件路径 | `/path/to/model.mllm` |
| `-c` | `--config_path` | 模型配置文件路径 | `/path/to/config.json` |
| `-t` | `--threads` | CPU 线程数 | `4` |
| `-pp` | `--prompt_length` | 提示词长度列表（逗号分隔） | `64,128,256` |
| `-tg` | `--test_generation_length` | 生成长度列表（逗号分隔，需与 pp 数量一致） | `100,200,300` |
| `-cl` | `--cache_length` | KV 缓存最大长度 | `2048` |

### 示例

#### 测试 Qwen3-0.6B 模型

```bash
./mllm-llm-benchmark \
  -n qwen3-w4a32-kai \
  -m ../models/Qwen3-0.6B-w4a32kai/model.mllm \
  -c ../models/Qwen3-0.6B-w4a32kai/config.json \
  -t 4 \
  -pp 64,128,256 \
  -tg 100,100,100 \
  -cl 2048
```

#### 快速测试（单个配置）

```bash
./mllm-llm-benchmark \
  -n qwen3-w4a32-kai \
  -m ../models/Qwen3-0.6B-w4a32kai/model.mllm \
  -c ../models/Qwen3-0.6B-w4a32kai/config.json \
  -t 8 \
  -pp 128 \
  -tg 128 \
  -cl 2048
```

## 输出示例

```
MLLM Build Version : abc123def456
ARCH               : ARM64
FP16               : true
...

Create Benchmark: qwen3-w4a32-kai
Model Info
========== Model Information ==========
Model Type         : Qwen3 W4A32 KAI
Hidden Size        : 1024
Num Layers         : 28
...
=======================================

Warmup Run
Warming up with 8 tokens prefill and 4 tokens generation...
Warmup completed

========================================
Starting Benchmark Tests
========================================

----------------------------------------
Test Configuration:
  Prompt Length (PP)    : 128
  Generation Length (TG): 128
----------------------------------------
  Run 1 of 3...
    TTFT         : 902.38605 ms
    Prefill Speed: 141.84618 tokens/s
    Decode Speed : 78.11022 tokens/s
    Cooling down for 5 seconds...
  Run 2 of 3...
    TTFT         : 911.94403 ms
    Prefill Speed: 140.3595 tokens/s
    Decode Speed : 77.60929 tokens/s
    Cooling down for 5 seconds...
  Run 3 of 3...
    TTFT         : 923.905 ms
    Prefill Speed: 138.54239 tokens/s
    Decode Speed : 76.48289 tokens/s

========== Average Results ==========
Configuration: PP= 128  TG= 128
Average TTFT         : 912.74506 ms
Average Prefill Speed: 140.24936 tokens/s
Average Decode Speed : 77.4008 tokens/s
=====================================


========================================
Benchmark Tests Completed
========================================
```

## 测试流程

每个测试配置会执行以下步骤：

1. **清理缓存** - 确保每次测试从干净状态开始
2. **运行 3 轮测试** - 每轮之间休眠 5 秒以避免过热
3. **计算平均值** - 对 3 轮结果求平均
4. **输出结果** - 显示 TTFT、Prefill Speed、Decode Speed

## 添加新模型支持

### 1. 创建新的 Benchmark 类

在 `models/` 目录下创建 `YourModel_Benchmark.hpp`：

```cpp
#include "BenchmarkTemplate.hpp"
#include <mllm/models/yourmodel/modeling_yourmodel.hpp>

class YourModel_Benchmark final : public BenchmarkTemplate {
 public:
  void init(const std::string& cfg_path, const std::string& model_path, int32_t cache_length) override {
    // 初始化你的模型
  }
  
  void printModelInfo() override {
    // 打印模型信息
  }
  
  void warmup() override {
    // 预热运行
  }
  
  void clear() override {
    // 清理 KV 缓存
  }
  
  BenchmarkTemplateResult run(int32_t pp, int32_t tg) override {
    // 运行测试并返回结果
  }
  
 private:
  std::unique_ptr<YourModelConfig> config_;
  std::unique_ptr<YourModel> model_;
};
```

### 2. 在 All.hpp 中注册

```cpp
#include "YourModel_Benchmark.hpp"

std::shared_ptr<BenchmarkTemplate> createBenchmark(const std::string& model_name) {
  auto normalized_model_name = tolower(model_name);
  
  // 添加你的模型判断
  if (normalized_model_name.find("yourmodel") != std::string::npos) {
    return std::make_shared<YourModel_Benchmark>();
  }
  
  // ... 其他模型
}
```

## 注意事项

- 确保有足够的内存用于模型加载和推理
- 每轮测试之间会休眠 5 秒，以避免设备过热影响结果
- TTFT 以毫秒为单位，速度以 tokens/s 为单位
- 每个配置会运行 3 轮并取平均值，以获得更稳定的结果

## 故障排查

### 错误：Model not initialized
- 检查模型路径是否正确
- 确认配置文件格式正确

### 错误：Benchmark not found
- 检查模型名称是否正确（使用 `-n` 参数）
- 确认 `All.hpp` 中有对应的模型注册

### 性能异常
- 检查 CPU 线程数设置（`-t` 参数）
- 确认没有其他程序占用系统资源
- 检查是否启用了正确的量化方案
