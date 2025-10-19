// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/core/aops/ParamOp.hpp"

namespace mllm::qnn {

// This is JUST a placeholder for QNN ParamOp
class QNNParamOp final : public aops::ParamOp {
 public:
  explicit QNNParamOp(const aops::ParamOpOptions& options);
};

class QNNParamOpFactory : public TypedOpFactory<OpTypes::kParam, aops::ParamOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::ParamOpOptions& options) override {
    return std::make_shared<QNNParamOp>(options);
  }
};

}  // namespace mllm::qnn
