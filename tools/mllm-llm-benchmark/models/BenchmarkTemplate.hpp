// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <string>

/**
 * @brief 基准测试结果结构
 */
struct BenchmarkTemplateResult {
  float ttft;            ///< Time To First Token (首 token 延迟) in milliseconds
  float prefill_speed;   ///< Prefill 阶段速度 in tokens/s
  float decode_speed;    ///< Decode 阶段速度 in tokens/s
};

/**
 * @brief 基准测试模板基类
 * 
 * 所有模型的 benchmark 实现都应该继承此类并实现所有虚函数。
 */
class BenchmarkTemplate {
 public:
  virtual ~BenchmarkTemplate() = default;

  /**
   * @brief 初始化模型
   * @param cfg_path 配置文件路径
   * @param model_path 模型权重文件路径
   * @param cache_length KV 缓存最大长度
   */
  virtual void init(const std::string& cfg_path, const std::string& model_path, int32_t cache_length) = 0;

  /**
   * @brief 打印模型信息
   * 
   * 应该输出模型的关键参数，如层数、隐藏层大小、注意力头数等。
   */
  virtual void printModelInfo() = 0;

  /**
   * @brief 预热运行
   * 
   * 使用小规模输入运行一次模型，确保模型进入稳定状态。
   */
  virtual void warmup() = 0;

  /**
   * @brief 清理缓存
   * 
   * 清理 KV 缓存和性能计数器，为下一次测试做准备。
   */
  virtual void clear() = 0;

  /**
   * @brief 运行基准测试
   * @param pp Prompt Length (提示词长度)
   * @param tg Test Generation Length (生成长度)
   * @return 测试结果
   */
  virtual BenchmarkTemplateResult run(int32_t pp, int32_t tg) = 0;
};
