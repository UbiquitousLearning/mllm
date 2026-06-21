# MLLM LLM Benchmark Tool

## Overview

This is a benchmark tool for measuring MLLM model performance, including:
- **TTFT (Time To First Token)**: Time to first token latency
- **Prefill Speed**: Prefill speed (tokens/s)
- **Decode Speed**: Decode generation speed (tokens/s)

## Build

Build from the mllm_v2 project root directory:

```bash
mkdir -p build && cd build
cmake ..
make mllm-llm-benchmark
```

## Usage

### Basic Usage

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

### Parameters

| Parameter | Long Format | Description | Example |
|-----------|-------------|-------------|---------|
| `-n` | `--model_name` | Model name (used to select the correct benchmark implementation) | `qwen3-w4a32-kai` |
| `-m` | `--model_path` | Model weight file path | `/path/to/model.mllm` |
| `-c` | `--config_path` | Model configuration file path | `/path/to/config.json` |
| `-t` | `--threads` | Number of CPU threads | `4` |
| `-pp` | `--prompt_length` | Prompt length list (comma-separated) | `64,128,256` |
| `-tg` | `--test_generation_length` | Generation length list (comma-separated, must match pp count) | `100,200,300` |
| `-cl` | `--cache_length` | Maximum KV cache length | `2048` |

### Examples

#### Testing Qwen3-0.6B Model

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

#### Quick Test (Single Configuration)

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

## Output Example

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

## Test Workflow

Each test configuration executes the following steps:

1. **Clear Cache** - Ensures each test starts from a clean state
2. **Run 3 Test Rounds** - 5-second sleep between rounds to avoid overheating
3. **Calculate Average** - Average results from 3 rounds
4. **Output Results** - Display TTFT, Prefill Speed, Decode Speed

## Adding New Model Support

### 1. Create New Benchmark Class

Create `YourModel_Benchmark.hpp` in the `models/` directory:

```cpp
#include "BenchmarkTemplate.hpp"
#include <mllm/models/yourmodel/modeling_yourmodel.hpp>

class YourModel_Benchmark final : public BenchmarkTemplate {
 public:
  void init(const std::string& cfg_path, const std::string& model_path, int32_t cache_length) override {
    // Initialize your model
  }
  
  void printModelInfo() override {
    // Print model information
  }
  
  void warmup() override {
    // Warmup run
  }
  
  void clear() override {
    // Clear KV cache
  }
  
  BenchmarkTemplateResult run(int32_t pp, int32_t tg) override {
    // Run test and return results
  }
  
 private:
  std::unique_ptr<YourModelConfig> config_;
  std::unique_ptr<YourModel> model_;
};
```

### 2. Register in All.hpp

```cpp
#include "YourModel_Benchmark.hpp"

std::shared_ptr<BenchmarkTemplate> createBenchmark(const std::string& model_name) {
  auto normalized_model_name = tolower(model_name);
  
  // Add your model check
  if (normalized_model_name.find("yourmodel") != std::string::npos) {
    return std::make_shared<YourModel_Benchmark>();
  }
  
  // ... other models
}
```

## Notes

- Ensure sufficient memory for model loading and inference
- 5-second sleep between test rounds to avoid device overheating affecting results
- TTFT is in milliseconds, speed is in tokens/s
- Each configuration runs 3 rounds and averages results for more stable measurements

## Troubleshooting

### Error: Model not initialized
- Check if model path is correct
- Verify configuration file format is correct

### Error: Benchmark not found
- Check if model name is correct (using `-n` parameter)
- Confirm corresponding model registration exists in `All.hpp`

### Performance Anomalies
- Check CPU thread count setting (`-t` parameter)
- Confirm no other programs are consuming system resources
- Check if correct quantization scheme is enabled
