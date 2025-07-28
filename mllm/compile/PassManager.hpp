/**
 * @file PassManager.hpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-28
 *
 */
#pragma once

#include <vector>

#include "mllm/compile/ir/Node.hpp"
#include "mllm/compile/passes/Pass.hpp"

namespace mllm::ir {

class PassManager {
 public:
  enum Pattern {  // NOLINT
    GREEDY = 0,
  };

  PassManager() = delete;

  explicit PassManager(const IRContext::ptr_t& ctx);

  PassManager& reg(const Pass::ptr_t& pass);

  PassManager& reg(const std::vector<Pass::ptr_t>& pass);

  void clear();

  bool run(Pattern p = GREEDY);

 private:
  IRContext::ptr_t ctx_;
  std::vector<Pass::ptr_t> passes_;
};

}  // namespace mllm::ir