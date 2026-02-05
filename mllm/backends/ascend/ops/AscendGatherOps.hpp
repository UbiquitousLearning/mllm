// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/GatherOp.hpp"
#include "mllm/core/OpTypes.hpp"

namespace mllm::ascend {

/// AscendGatherOp implements Gather using ACLNN aclnnEmbedding.
/// This is used instead of ATB GatherParam because ATB Gather is not supported on Ascend 310B.
class AscendGatherOp final : public aops::GatherOp {
 public:
  explicit AscendGatherOp(const aops::GatherOpOptions& options);

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class AscendGatherOpFactory final : public TypedOpFactory<OpTypes::kGather, aops::GatherOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::GatherOpOptions& options) override {
    return std::make_shared<AscendGatherOp>(options);
  }
};

}  // namespace mllm::ascend
