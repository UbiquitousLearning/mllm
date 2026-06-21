// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/compile/passes/Pass.hpp"

namespace mllm::ir {

struct ProgramModeConfigPassOptions {
  int32_t mode = 0;
};

class ProgramModeConfigPass final : public Pass {
 public:
  explicit ProgramModeConfigPass(const ProgramModeConfigPassOptions& options);

  uint8_t run(const node_ptr_t& op) override;

 private:
  ProgramModeConfigPassOptions options_;
};

Pass::ptr_t createProgramModeConfigPass(const ProgramModeConfigPassOptions& options);

}  // namespace mllm::ir
