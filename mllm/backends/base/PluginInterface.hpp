// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include "mllm/core/BaseOp.hpp"

namespace mllm::plugin::interface {

class CustomizedOp : BaseOp {};

template<typename CargoT>
class CustomizedOpFactory : protected TypedOpFactory<OpTypes::kDynamicOp_Start, CargoT> {
  inline std::shared_ptr<BaseOp> createOpImpl(const CargoT& cargo) override {
    MLLM_ERROR_EXIT(ExitCode::kCoreError, "CustomizedOpFactory::createOpImpl is not implemented");
    return nullptr;
  }
};

}  // namespace mllm::plugin::interface
