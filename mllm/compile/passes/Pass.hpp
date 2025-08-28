// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "mllm/compile/ir/Node.hpp"
#include "mllm/compile/passes/Pattern.hpp"

namespace mllm::ir {

enum PassReturnState : uint8_t {
  PASS_RET_SUCCESS = 0x01,
  PASS_RET_FAILURE = 0x02,
  PASS_RET_CONTINUE = 0x80,
};

class Pass {
 public:
  using ptr_t = std::shared_ptr<Pass>;

  virtual ~Pass() = default;

  Pass() = default;

  // Do not promise the T type is castable with external_data_.
  //
  // NOTE: Pass class will not free external_data_;
  template<typename T>
  T* getExternalData() {
    return (T*)external_data_;
  }

  template<typename T>
  void setExternalData(T* data) {
    external_data_ = (void*)data;
  }

  // the ret should be PassReturnState's binary expression's value
  // E.g.: PASS_RET_SUCCESS | PASS_RET_CONTINUE
  virtual uint8_t run(const node_ptr_t& op);

  virtual void setCtx(const std::shared_ptr<IRContext>& ctx);

  std::shared_ptr<IRContext> getCtx();

 private:
  std::shared_ptr<IRContext> ctx_ = nullptr;
  void* external_data_ = nullptr;
};

class PatternMatchPass : public Pass {
 public:
  PatternMatchPass() = default;

  ~PatternMatchPass() override = default;

  uint8_t run(const node_ptr_t& op) override;

  template<typename... Args>
  void regPattern() {
    (..., (_reg_one_pattern<Args>()));
  }

  void setCtx(const std::shared_ptr<IRContext>& ctx) override;

 protected:
  template<typename T>
  void _reg_one_pattern() {
    auto ins = T::create();
    patterns_.emplace_back(ins);
  }

  std::vector<Pattern::ptr_t> patterns_;
};

}  // namespace mllm::ir
