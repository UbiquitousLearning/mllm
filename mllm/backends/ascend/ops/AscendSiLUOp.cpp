// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/ascend/ops/AscendSiLUOp.hpp"

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

AscendSiLUOp::AscendSiLUOp(const aops::SiLUOpOptions& options) : aops::SiLUOp(options) {}

void AscendSiLUOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  BaseOp::setup(inputs, outputs);
}

void AscendSiLUOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  MLLM_RT_ASSERT_EQ(inputs.size(), 1);
  MLLM_RT_ASSERT_EQ(outputs.size(), 1);

  const auto& x = inputs[0];
  auto& y = outputs[0];

  if (x.dtype() != y.dtype()) {
    NYI("AscendSiLUOp currently requires x/y have same dtype");
  }
  if (x.numel() != y.numel()) {
    NYI("AscendSiLUOp requires x/y have same numel");
  }

  atb::infer::ActivationParam siluParam;
  siluParam.activationType = atb::infer::ACTIVATION_SWISH;

  atb::Operation* op = nullptr;
  auto st = atb::CreateOperation(siluParam, &op);
  if (st != atb::NO_ERROR || op == nullptr) {
    MLLM_ERROR_EXIT(ExitCode::kAscendError, "ATB CreateOperation(ACTIVATION_SWISH) failed, status={}", static_cast<int>(st));
  }

  atb::Context* atb_ctx = getGlobalAtbContext();

  atb::Tensor atb_x;
  atb::Tensor atb_y;

  fillAtbTensorDesc(x, atb_x.desc);
  fillAtbTensorDesc(y, atb_y.desc);

  atb_x.deviceData = reinterpret_cast<uint8_t*>(x.ptr<void>());
  atb_x.dataSize = x.bytes();
  atb_y.deviceData = reinterpret_cast<uint8_t*>(y.ptr<void>());
  atb_y.dataSize = y.bytes();

  atb::SVector<atb::Tensor> inTensors;
  atb::SVector<atb::Tensor> outTensors;
  inTensors.push_back(atb_x);
  outTensors.push_back(atb_y);

  atb::VariantPack vp;
  vp.inTensors = inTensors;
  vp.outTensors = outTensors;

  uint64_t workspaceSize = 0;
  st = op->Setup(vp, workspaceSize, atb_ctx);
  if (st != atb::NO_ERROR) {
    MLLM_ERROR_EXIT(ExitCode::kAscendError, "ATB SiLUOp Setup failed, status={}", static_cast<int>(st));
  }

  void* workspace = nullptr;
  int workspace_block_id = -1;
  if (workspaceSize > 0) {
    auto& mem_mgr = getAscendMemoryManager();
    mem_mgr.allocateBlock(static_cast<uint32_t>(workspaceSize), workspace_block_id);
    mem_mgr.getBlockPtr(workspace_block_id, workspace);
  }
  {
    ASCEND_TIME_SCOPE("AscendSiLUOp::forward");
    st = op->Execute(vp, reinterpret_cast<uint8_t*>(workspace), workspaceSize, atb_ctx);
  }
  if (st != atb::NO_ERROR) {
    MLLM_ERROR_EXIT(ExitCode::kAscendError, "ATB SiLUOp Execute failed, status={}", static_cast<int>(st));
  }


  syncGlobalAtbStream();

  if (workspace_block_id != -1) {
    auto& mem_mgr = getAscendMemoryManager();
    mem_mgr.freeBlock(workspace_block_id);
  }

  atb::DestroyOperation(op);
}

}  // namespace mllm::ascend
