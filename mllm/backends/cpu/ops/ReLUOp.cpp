// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/cpu/ops/ReLUOp.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"

namespace mllm::cpu {

CPUReLUOp::CPUReLUOp(const aops::ReLUOpOptions& options) : aops::ReLUOp(options) {}

void CPUReLUOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  const auto& input = inputs[0];
  auto& output = outputs[0];

  auto dtype = input.dtype();
  switch (dtype) {
    case kFloat32: {
      auto input_ptr = input.ptr<float>();
      auto output_ptr = output.ptr<float>();
      for (int i = 0; i < input.numel(); ++i) { output_ptr[i] = input_ptr[i] > 0 ? input_ptr[i] : 0; }
      break;
    }
    default: NYI("ReLU not supported for data type: {}", nameOfType(dtype));
  }
}

}  // namespace mllm::cpu