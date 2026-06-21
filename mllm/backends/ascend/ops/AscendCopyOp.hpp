// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/CopyOp.hpp"

namespace mllm::ascend {

class AscendCopyOp final : public aops::CopyOp {
 public:
  explicit AscendCopyOp(const aops::CopyOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class AscendCopyOpFactory : public TypedOpFactory<OpTypes::kCopy, aops::CopyOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::CopyOpOptions& options) override {
    return std::make_shared<AscendCopyOp>(options);
  }
};

}  // namespace mllm::ascend
