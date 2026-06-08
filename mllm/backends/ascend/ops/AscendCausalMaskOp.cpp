// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/ascend/ops/AscendCausalMaskOp.hpp"

#include "mllm/backends/ascend/AscendCommon.hpp"
#include "mllm/backends/ascend/ops/AscendCausalMaskKernel.hpp"

namespace mllm::ascend {

AscendCausalMaskOp::AscendCausalMaskOp(const aops::CausalMaskOpOptions& options) : aops::CausalMaskOp(options) {}

void AscendCausalMaskOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  BaseOp::setup(inputs, outputs);
}

void AscendCausalMaskOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  MLLM_RT_ASSERT_EQ(inputs.size(), 1);
  MLLM_RT_ASSERT_EQ(outputs.size(), 1);

  atb::Tensor atb_x;
  atb::Tensor atb_y;
  fillAtbTensor(inputs[0], atb_x);
  fillAtbTensor(outputs[0], atb_y);
  auto st = executeAscendCausalMaskKernel(atb_x, atb_y, options_.sliding_window, options_.window_size);
  MLLM_ATB_CHECK(st);
  syncGlobalAtbStream();
}

}  // namespace mllm::ascend
