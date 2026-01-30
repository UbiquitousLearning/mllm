// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/ascend/ops/AscendRMSNormOp.hpp"

#include <acl/acl.h>
#include <iostream>
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

AscendRMSNormOp::AscendRMSNormOp(const aops::RMSNormOpOptions& options) : aops::RMSNormOp(options) {}

void AscendRMSNormOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  BaseOp::setup(inputs, outputs);
}

void AscendRMSNormOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  //MLLM_RT_ASSERT(inputs.size() == 1 || inputs.size() == 2, "AscendRMSNormOp expects 1 or 2 inputs");
  MLLM_RT_ASSERT_EQ(outputs.size(), 1);

  const auto& x = inputs[0];
  const auto& weight = (inputs.size() == 2) ? inputs[1] : weight_;
  auto& y = outputs[0];

  const Tensor& weight_for_atb = weight;

  if (x.dtype() != y.dtype()) {
    NYI("AscendRMSNormOp currently requires x/y have same dtype");
  }
  if (x.numel() != y.numel()) {
    NYI("AscendRMSNormOp requires x/y have same numel");
  }

  atb::infer::RmsNormParam rmsNormParam;
  rmsNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
  rmsNormParam.normParam.quantType = atb::infer::QuantType::QUANT_UNQUANT;
  rmsNormParam.normParam.epsilon = options_.epsilon;

  atb::Operation* op = nullptr;
  auto st = atb::CreateOperation(rmsNormParam, &op);
  if (st != atb::NO_ERROR || op == nullptr) {
    MLLM_ERROR_EXIT(ExitCode::kAscendError, "ATB CreateOperation(RMS_NORM) failed, status={}", static_cast<int>(st));
  }

  atb::Context* atb_ctx = getGlobalAtbContext();

  atb::Tensor atb_x;
  atb::Tensor atb_weight;
  atb::Tensor atb_y;

  fillAtbTensor(x, atb_x);
  fillAtbTensor(weight_for_atb, atb_weight);
  fillAtbTensor(y, atb_y);

  atb::SVector<atb::Tensor> inTensors;
  atb::SVector<atb::Tensor> outTensors;
  inTensors.push_back(atb_x);
  inTensors.push_back(atb_weight);
  outTensors.push_back(atb_y);

  atb::VariantPack vp;
  vp.inTensors = inTensors;
  vp.outTensors = outTensors;

  uint64_t workspaceSize = 0;
  st = op->Setup(vp, workspaceSize, atb_ctx);
  if (st != atb::NO_ERROR) {
    MLLM_ERROR_EXIT(ExitCode::kAscendError, "ATB RMSNormOp Setup failed, status={}", static_cast<int>(st));
  }

  void* workspace = nullptr;
  int workspace_block_id = -1;
  if (workspaceSize > 0) {
    auto& mem_mgr = getAscendMemoryManager();
    mem_mgr.allocateBlock(static_cast<uint32_t>(workspaceSize), workspace_block_id);
    mem_mgr.getBlockPtr(workspace_block_id, workspace);
  }
  {
    ASCEND_TIME_SCOPE("AscendRMSNormOp::forward");
    st = op->Execute(vp, reinterpret_cast<uint8_t*>(workspace), workspaceSize, atb_ctx);
  }
  if (st != atb::NO_ERROR) {
    MLLM_ERROR_EXIT(ExitCode::kAscendError, "ATB RMSNormOp Execute failed, status={}", static_cast<int>(st));
  }

  syncGlobalAtbStream();

  if (workspace_block_id != -1) {
    auto& mem_mgr = getAscendMemoryManager();
    mem_mgr.freeBlock(workspace_block_id);
  }

  atb::DestroyOperation(op);
}

}  // namespace mllm::ascend