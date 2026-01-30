// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/ascend/ops/AscendLinearOp.hpp"

#include <acl/acl.h>
#include <atb/atb_infer.h>
#include <atb/types.h>
#include <atb/utils.h>
#include <atb/infer_op_params.h>

#include "mllm/utils/Common.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/backends/ascend/memory/AscendMemoryManager.hpp"
#include "mllm/backends/ascend/AscendCommon.hpp"

namespace mllm::ascend {

AscendLinearOp::AscendLinearOp(const aops::LinearOpOptions& options) : aops::LinearOp(options) {}

void AscendLinearOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  if (options().isRedirect()) {
    MLLM_RT_ASSERT(inputs.size() >= 1);
    const auto& input = inputs[0];
    const auto& weight = inputs.size() >= 2 ? inputs[1] : this->weight();
    auto out_shape = input.shape();
    out_shape[out_shape.size() - 1] = weight.shape()[0];  // out_channels
    outputs.emplace_back(Tensor::empty(out_shape, input.dtype(), input.device()));
    return;
  }
  aops::LinearOp::reshape(inputs, outputs);
}

void AscendLinearOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  BaseOp::setup(inputs, outputs);
}

void AscendLinearOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  MLLM_RT_ASSERT(inputs.size() >= 1 && inputs.size() <= 3);
  MLLM_RT_ASSERT_EQ(outputs.size(), 1);

  const Tensor* weight_ptr = nullptr;
  const Tensor* bias_ptr = nullptr;

  if (inputs.size() == 1) {
    weight_ptr = &weight();
    if (options().bias) { bias_ptr = &bias(); }
  } else if (inputs.size() == 2) {
    weight_ptr = &inputs[1];
  } else if (inputs.size() == 3) {
    weight_ptr = &inputs[1];
    bias_ptr = &inputs[2];
  }

  const auto& x = inputs[0];
  auto& y = outputs[0];

  // Validate that input tensors are FP16
  if (x.dtype() != MLLM_TYPE_F16) {
    MLLM_ERROR_EXIT(ExitCode::kAscendError,
                    "AscendLinearOp: Input tensor must be FP16, but got dtype={}",
                    static_cast<int>(x.dtype()));
  }
  if (weight_ptr->dtype() != MLLM_TYPE_F16) {
    MLLM_ERROR_EXIT(ExitCode::kAscendError,
                    "AscendLinearOp: Weight tensor must be FP16, but got dtype={}",
                    static_cast<int>(weight_ptr->dtype()));
  }
  if (bias_ptr != nullptr && bias_ptr->dtype() != MLLM_TYPE_F16) {
    MLLM_ERROR_EXIT(ExitCode::kAscendError,
                    "AscendLinearOp: Bias tensor must be FP16, but got dtype={}",
                    static_cast<int>(bias_ptr->dtype()));
  }

  // Validate bias dimensions: ATB Linear requires bias to be 2D [1, out_channels]
  if (bias_ptr != nullptr) {
    const auto& bias_shape = bias_ptr->shape();
    if (bias_shape.size() == 1) {
      MLLM_ERROR_EXIT(ExitCode::kAscendError,
                      "AscendLinearOp: Bias tensor must be 2D [1, out_channels], but got 1D shape with size={}. "
                      "Please reshape the bias tensor before passing to AscendLinearOp.",
                      bias_shape[0]);
    }
    if (bias_shape.size() != 2 || bias_shape[0] != 1) {
      MLLM_ERROR_EXIT(ExitCode::kAscendError,
                      "AscendLinearOp: Bias tensor must be 2D with shape [1, out_channels], but got shape=[{}, {}]",
                      bias_shape.size() >= 1 ? bias_shape[0] : 0,
                      bias_shape.size() >= 2 ? bias_shape[1] : 0);
    }
  }


  atb::infer::LinearParam linearParam;
  linearParam.transposeA = false;
  linearParam.transposeB = true;  // Set to true because weight is [out_channels, in_channels]
  linearParam.hasBias = (bias_ptr != nullptr);
  linearParam.outDataType = ACL_DT_UNDEFINED;
  linearParam.enAccum = false;
  linearParam.matmulType = atb::infer::LinearParam::MATMUL_UNDEFINED;
  linearParam.quantMode = atb::infer::LinearParam::QUANT_UNDEFINED;

  atb::Operation* op = nullptr;
  auto st = atb::CreateOperation(linearParam, &op);
  if (st != atb::NO_ERROR || op == nullptr) {
    MLLM_ERROR_EXIT(ExitCode::kAscendError, "ATB CreateOperation(Linear) failed, status={}", static_cast<int>(st));
  }

  atb::Context* atb_ctx = getGlobalAtbContext();

  atb::Tensor atb_x;
  atb::Tensor atb_weight;
  atb::Tensor atb_y;
  atb::Tensor atb_bias;

  fillAtbTensor(x, atb_x);
  fillAtbTensor(*weight_ptr, atb_weight);
  fillAtbTensor(y, atb_y);

  atb::SVector<atb::Tensor> inTensors;
  atb::SVector<atb::Tensor> outTensors;
  inTensors.push_back(atb_x);
  inTensors.push_back(atb_weight);

  if (bias_ptr != nullptr) {
    fillAtbTensor(*bias_ptr, atb_bias);
    inTensors.push_back(atb_bias);
  }

  outTensors.push_back(atb_y);

  atb::VariantPack vp;
  vp.inTensors = inTensors;
  vp.outTensors = outTensors;

  uint64_t workspaceSize = 0;
  st = op->Setup(vp, workspaceSize, atb_ctx);
  if (st != atb::NO_ERROR) {
    MLLM_ERROR_EXIT(ExitCode::kAscendError, "ATB LinearOp Setup failed, status={}", static_cast<int>(st));
  }

  void* workspace = nullptr;
  int workspace_block_id = -1;
  if (workspaceSize > 0) {
    auto& mem_mgr = getAscendMemoryManager();
    mem_mgr.allocateBlock(static_cast<uint32_t>(workspaceSize), workspace_block_id);
    mem_mgr.getBlockPtr(workspace_block_id, workspace);
  }

  {
    ASCEND_TIME_SCOPE("AscendLinearOp::forward");
    st = op->Execute(vp, reinterpret_cast<uint8_t*>(workspace), workspaceSize, atb_ctx);
  }

  if (st != atb::NO_ERROR) {
    MLLM_ERROR_EXIT(ExitCode::kAscendError, "ATB LinearOp Execute failed, status={}", static_cast<int>(st));
  }

  syncGlobalAtbStream();

  if (workspace_block_id != -1) {
    auto& mem_mgr = getAscendMemoryManager();
    mem_mgr.freeBlock(workspace_block_id);
  }

  atb::DestroyOperation(op);
}

}  // namespace mllm::ascend