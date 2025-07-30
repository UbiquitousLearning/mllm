/**
 * @file PassManager.cpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-30
 *
 */
#include "mllm/compile/PassManager.hpp"

namespace mllm::ir {

PassManager::PassManager(const std::shared_ptr<IRContext>& ctx) : ctx_(ctx) {}

PassManager& PassManager::reg(const Pass::ptr_t& pass) {
  pass->setCtx(ctx_);
  passes_.emplace_back(pass);
  return *this;
}

PassManager& PassManager::reg(const std::vector<Pass::ptr_t>& pass) {
  for (auto& p : pass) {
    p->setCtx(ctx_);
    passes_.emplace_back(p);
  }
  return *this;
}

void PassManager::clear() { passes_.clear(); }

bool PassManager::run(Pattern p) {
  for (auto& pass : passes_) {
    switch (p) {
      case GREEDY: {
        uint8_t res = 0;
        do {
          res = pass->run(ctx_->topLevelOp());

          if (res & (size_t)0x02) { return false; }

        } while ((res >> (size_t)8) == 0x01);
        break;
      }
    }
  }
  return true;
}

}  // namespace mllm::ir