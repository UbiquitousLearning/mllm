// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <string>

/**
 * @brief Benchmark result structure
 */
struct BenchmarkTemplateResult {
  float ttft;            ///< Time To First Token in milliseconds
  float prefill_speed;   ///< Prefill phase speed in tokens/s
  float decode_speed;    ///< Decode phase speed in tokens/s
};

/**
 * @brief Base class for benchmark templates
 * 
 * All model benchmark implementations should inherit from this class and implement all virtual functions.
 */
class BenchmarkTemplate {
 public:
  virtual ~BenchmarkTemplate() = default;

  /**
   * @brief Initialize model
   * @param cfg_path Configuration file path
   * @param model_path Model weight file path
   * @param cache_length Maximum KV cache length
   */
  virtual void init(const std::string& cfg_path, const std::string& model_path, int32_t cache_length) = 0;

  /**
   * @brief Print model information
   * 
   * Should output model key parameters such as number of layers, hidden size, attention heads, etc.
   */
  virtual void printModelInfo() = 0;

  /**
   * @brief Warmup run
   * 
   * Run the model once with small-scale input to ensure the model enters a stable state.
   */
  virtual void warmup() = 0;

  /**
   * @brief Clear cache
   * 
   * Clear KV cache and performance counters to prepare for the next test.
   */
  virtual void clear() = 0;

  /**
   * @brief Run benchmark test
   * @param pp Prompt Length
   * @param tg Test Generation Length
   * @return Test results
   */
  virtual BenchmarkTemplateResult run(int32_t pp, int32_t tg) = 0;
};
