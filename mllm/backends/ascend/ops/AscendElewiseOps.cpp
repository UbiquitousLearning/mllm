// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/ascend/ops/AscendElewiseOps.hpp"

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

AscendAddOp::AscendAddOp(const aops::AddOpOptions& options) : aops::AddOp(options) {}

void AscendAddOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  BaseOp::setup(inputs, outputs);
}

void AscendAddOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  MLLM_RT_ASSERT_EQ(inputs.size(), 2);
  MLLM_RT_ASSERT_EQ(outputs.size(), 1);

  const auto& x = inputs[0];
  const auto& y = inputs[1];
  auto& z = outputs[0];

  if (x.dtype() != y.dtype() || x.dtype() != z.dtype()) {
    NYI("AscendAddOp currently requires x/y/z have same dtype");
  }
  if (x.numel() != y.numel() || x.numel() != z.numel()) {
    NYI("AscendAddOp demo only supports no-broadcast case (numel equal)");
  }

  atb::infer::ElewiseParam addParam;
  addParam.elewiseType = atb::infer::ElewiseParam::ELEWISE_ADD;

  atb::Operation* op = nullptr;
  auto st = atb::CreateOperation(addParam, &op);
  if (st != atb::NO_ERROR || op == nullptr) {
    MLLM_ERROR_EXIT(ExitCode::kAscendError, "ATB CreateOperation(ELEWISE_ADD) failed, status={}", static_cast<int>(st));
  }

  atb::Context* atb_ctx = getGlobalAtbContext();

  atb::Tensor atb_x;
  atb::Tensor atb_y;
  atb::Tensor atb_z;

  fillAtbTensorDesc(x, atb_x.desc);
  fillAtbTensorDesc(y, atb_y.desc);
  fillAtbTensorDesc(z, atb_z.desc);

  atb_x.deviceData = reinterpret_cast<uint8_t*>(x.ptr<void>());
  atb_x.dataSize = x.bytes();
  atb_y.deviceData = reinterpret_cast<uint8_t*>(y.ptr<void>());
  atb_y.dataSize = y.bytes();
  atb_z.deviceData = reinterpret_cast<uint8_t*>(z.ptr<void>());
  atb_z.dataSize = z.bytes();

  atb::SVector<atb::Tensor> inTensors;
  atb::SVector<atb::Tensor> outTensors;
  inTensors.push_back(atb_x);
  inTensors.push_back(atb_y);
  outTensors.push_back(atb_z);

  atb::VariantPack vp;
  vp.inTensors = inTensors;
  vp.outTensors = outTensors;

  uint64_t workspaceSize = 0;
  st = op->Setup(vp, workspaceSize, atb_ctx);
  if (st != atb::NO_ERROR) {
    MLLM_ERROR_EXIT(ExitCode::kAscendError, "ATB AddOp Setup failed, status={}", static_cast<int>(st));
  }

  void* workspace = nullptr;
  int workspace_block_id = -1;
  if (workspaceSize > 0) {
    auto& mem_mgr = getAscendMemoryManager();
    mem_mgr.allocateBlock(static_cast<uint32_t>(workspaceSize), workspace_block_id);
    mem_mgr.getBlockPtr(workspace_block_id, workspace);
  }
  {
    ASCEND_TIME_SCOPE("AscendAddOp::forward");
    st = op->Execute(vp, reinterpret_cast<uint8_t*>(workspace), workspaceSize, atb_ctx);
  }
  if (st != atb::NO_ERROR) {
    MLLM_ERROR_EXIT(ExitCode::kAscendError, "ATB AddOp Execute failed, status={}", static_cast<int>(st));
  }

  
  syncGlobalAtbStream();

  if (workspace_block_id != -1) {
    auto& mem_mgr = getAscendMemoryManager();
    mem_mgr.freeBlock(workspace_block_id);
  }

  atb::DestroyOperation(op);
}

}  // namespace mllm::ascend