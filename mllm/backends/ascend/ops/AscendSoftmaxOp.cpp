// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/ascend/ops/AscendSoftmaxOp.hpp"

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

AscendSoftmaxOp::AscendSoftmaxOp(const aops::SoftmaxOpOptions& options) : aops::SoftmaxOp(options) {}

void AscendSoftmaxOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  BaseOp::setup(inputs, outputs);
}

void AscendSoftmaxOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  MLLM_RT_ASSERT_EQ(inputs.size(), 1);
  MLLM_RT_ASSERT_EQ(outputs.size(), 1);

  const auto& x = inputs[0];
  auto& y = outputs[0];

  // Validate that input tensors are FP16
  if (x.dtype() != MLLM_TYPE_F16) {
    MLLM_ERROR_EXIT(ExitCode::kAscendError,
                    "AscendSoftmaxOp: Input tensor must be FP16, but got dtype={}",
                    static_cast<int>(x.dtype()));
  }
  if (y.dtype() != MLLM_TYPE_F16) {
    MLLM_ERROR_EXIT(ExitCode::kAscendError,
                    "AscendSoftmaxOp: Output tensor must be FP16, but got dtype={}",
                    static_cast<int>(y.dtype()));
  }

  if (x.dtype() != y.dtype()) {
    NYI("AscendSoftmaxOp currently requires x/y have same dtype");
  }
  if (x.numel() != y.numel()) {
    NYI("AscendSoftmaxOp requires x/y have same numel");
  }

  // Configure Softmax parameters
  atb::infer::SoftmaxParam softmaxParam;

  // Convert axis to positive index if negative
  int axis = options_.axis;
  if (axis < 0) {
    axis = static_cast<int>(x.rank()) + axis;
  }

  // ATB expects axes as SVector<int64_t>
  softmaxParam.axes.push_back(static_cast<int64_t>(axis));

  // Create ATB operation
  atb::Operation* op = nullptr;
  auto st = atb::CreateOperation(softmaxParam, &op);
  if (st != atb::NO_ERROR || op == nullptr) {
    MLLM_ERROR_EXIT(ExitCode::kAscendError,
                    "ATB CreateOperation(Softmax) failed, status={}",
                    static_cast<int>(st));
  }

  // Get global ATB context
  atb::Context* atb_ctx = getGlobalAtbContext();

  // Prepare ATB tensors
  atb::Tensor atb_x;
  atb::Tensor atb_y;

  fillAtbTensor(x, atb_x);
  fillAtbTensor(y, atb_y);

  // Setup input/output tensors
  atb::SVector<atb::Tensor> inTensors;
  atb::SVector<atb::Tensor> outTensors;
  inTensors.push_back(atb_x);
  outTensors.push_back(atb_y);

  atb::VariantPack vp;
  vp.inTensors = inTensors;
  vp.outTensors = outTensors;

  // Setup operation (calculate required workspace size)
  uint64_t workspaceSize = 0;
  st = op->Setup(vp, workspaceSize, atb_ctx);
  if (st != atb::NO_ERROR) {
    MLLM_ERROR_EXIT(ExitCode::kAscendError,
                    "ATB SoftmaxOp Setup failed, status={}",
                    static_cast<int>(st));
  }

  // Allocate workspace if needed
  void* workspace = nullptr;
  int workspace_block_id = -1;
  if (workspaceSize > 0) {
    auto& mem_mgr = getAscendMemoryManager();
    mem_mgr.allocateBlock(static_cast<uint32_t>(workspaceSize), workspace_block_id);
    mem_mgr.getBlockPtr(workspace_block_id, workspace);
  }

  // Execute operation
  {
    ASCEND_TIME_SCOPE("AscendSoftmaxOp::forward");
    st = op->Execute(vp, reinterpret_cast<uint8_t*>(workspace), workspaceSize, atb_ctx);
  }
  if (st != atb::NO_ERROR) {
    MLLM_ERROR_EXIT(ExitCode::kAscendError,
                    "ATB SoftmaxOp Execute failed, status={}",
                    static_cast<int>(st));
  }

  // Synchronize stream
  syncGlobalAtbStream();

  // Free workspace
  if (workspace_block_id != -1) {
    auto& mem_mgr = getAscendMemoryManager();
    mem_mgr.freeBlock(workspace_block_id);
  }

  // Destroy operation
  atb::DestroyOperation(op);
}

}  // namespace mllm::ascend
