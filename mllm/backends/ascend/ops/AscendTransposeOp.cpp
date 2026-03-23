// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/ascend/ops/AscendTransposeOp.hpp"

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

AscendTransposeOp::AscendTransposeOp(const aops::TransposeOpOptions& options) : aops::TransposeOp(options) {}

void AscendTransposeOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  aops::TransposeOp::reshape(inputs, outputs);
}

void AscendTransposeOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  BaseOp::setup(inputs, outputs);
}

void AscendTransposeOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  MLLM_RT_ASSERT_EQ(inputs.size(), 1);
  MLLM_RT_ASSERT_EQ(outputs.size(), 1);

  const auto& x = inputs[0];
  auto& y = outputs[0];

  // Validate that input tensor is FP16
  if (x.dtype() != MLLM_TYPE_F16) {
    MLLM_ERROR_EXIT(ExitCode::kAscendError,
                    "AscendTransposeOp: Input tensor must be FP16, but got dtype={}",
                    static_cast<int>(x.dtype()));
  }

  // Build permutation vector from dim0 and dim1
  // For a tensor with ndim dimensions, we swap dim0 and dim1
  int ndim = static_cast<int>(x.shape().size());
  int dim0 = options().dim0;
  int dim1 = options().dim1;

  // Handle negative dimensions
  if (dim0 < 0) { dim0 += ndim; }
  if (dim1 < 0) { dim1 += ndim; }

  MLLM_RT_ASSERT(dim0 >= 0 && dim0 < ndim);
  MLLM_RT_ASSERT(dim1 >= 0 && dim1 < ndim);

  // Create ATB Transpose operation
  atb::infer::TransposeParam transposeParam;

  // Create permutation: identity permutation with dim0 and dim1 swapped
  // ATB uses SVector<int>, so we push elements one by one
  for (int i = 0; i < ndim; ++i) {
    transposeParam.perm.push_back(i);
  }
  std::swap(transposeParam.perm[dim0], transposeParam.perm[dim1]);

  atb::Operation* op = nullptr;
  auto st = atb::CreateOperation(transposeParam, &op);
  if (st != atb::NO_ERROR || op == nullptr) {
    MLLM_ERROR_EXIT(ExitCode::kAscendError, "ATB CreateOperation(Transpose) failed, status={}", static_cast<int>(st));
  }

  atb::Context* atb_ctx = getGlobalAtbContext();

  atb::Tensor atb_x;
  atb::Tensor atb_y;

  fillAtbTensor(x, atb_x);
  fillAtbTensor(y, atb_y);

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
    MLLM_ERROR_EXIT(ExitCode::kAscendError, "ATB TransposeOp Setup failed, status={}", static_cast<int>(st));
  }

  void* workspace = nullptr;
  int workspace_block_id = -1;
  if (workspaceSize > 0) {
    auto& mem_mgr = getAscendMemoryManager();
    mem_mgr.allocateBlock(static_cast<uint32_t>(workspaceSize), workspace_block_id);
    mem_mgr.getBlockPtr(workspace_block_id, workspace);
  }

  {
    //ASCEND_TIME_SCOPE("AscendTransposeOp::forward");
    st = op->Execute(vp, reinterpret_cast<uint8_t*>(workspace), workspaceSize, atb_ctx);
  }

  if (st != atb::NO_ERROR) {
    MLLM_ERROR_EXIT(ExitCode::kAscendError, "ATB TransposeOp Execute failed, status={}", static_cast<int>(st));
  }

  syncGlobalAtbStream();

  if (workspace_block_id != -1) {
    auto& mem_mgr = getAscendMemoryManager();
    mem_mgr.freeBlock(workspace_block_id);
  }

  atb::DestroyOperation(op);
}

}  // namespace mllm::ascend
