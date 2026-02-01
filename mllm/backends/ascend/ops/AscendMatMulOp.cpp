// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/ascend/ops/AscendMatMulOp.hpp"

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

AscendMatMulOp::AscendMatMulOp(const aops::MatMulOpOptions& options) : aops::MatMulOp(options) {}

void AscendMatMulOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  BaseOp::setup(inputs, outputs);
}

namespace {

// Helper to fill ATB tensor with custom shape (for reshape without copy)
void fillAtbTensorWithShape(const Tensor& t, atb::Tensor& atb_tensor, const std::vector<int64_t>& shape) {
  atb::TensorDesc desc;
  desc.dtype = ACL_FLOAT16;  // Ascend uses FP16
  desc.format = ACL_FORMAT_ND;

  desc.shape.dimNum = shape.size();
  for (size_t i = 0; i < shape.size(); ++i) {
    desc.shape.dims[i] = shape[i];
  }

  atb_tensor.desc = desc;
  atb_tensor.dataSize = atb::Utils::GetTensorSize(atb_tensor);
  atb_tensor.deviceData = reinterpret_cast<uint8_t*>(t.ptr<void>());
}

}  // namespace

void AscendMatMulOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  MLLM_RT_ASSERT_EQ(inputs.size(), 2);
  MLLM_RT_ASSERT_EQ(outputs.size(), 1);

  const auto& A = inputs[0];
  const auto& B = inputs[1];
  auto& C = outputs[0];

  // ATB Linear/MatMul only supports 2D/3D tensors.
  // For 4D tensors [B, H, S, D], we reshape to 3D [B*H, S, D], compute, then reshape back.
  const auto& a_shape = A.shape();
  const auto& b_shape = B.shape();
  const auto& c_shape = C.shape();

  bool is_4d = (a_shape.size() == 4);

  // Prepare shapes for ATB
  std::vector<int64_t> atb_a_shape, atb_b_shape, atb_c_shape;

  if (is_4d) {
    // Reshape [B, H, S, D] -> [B*H, S, D]
    int64_t batch_heads_a = static_cast<int64_t>(a_shape[0]) * static_cast<int64_t>(a_shape[1]);
    int64_t batch_heads_b = static_cast<int64_t>(b_shape[0]) * static_cast<int64_t>(b_shape[1]);
    int64_t batch_heads_c = static_cast<int64_t>(c_shape[0]) * static_cast<int64_t>(c_shape[1]);

    atb_a_shape = {batch_heads_a, static_cast<int64_t>(a_shape[2]), static_cast<int64_t>(a_shape[3])};
    atb_b_shape = {batch_heads_b, static_cast<int64_t>(b_shape[2]), static_cast<int64_t>(b_shape[3])};
    atb_c_shape = {batch_heads_c, static_cast<int64_t>(c_shape[2]), static_cast<int64_t>(c_shape[3])};
  } else {
    // 2D or 3D: use original shapes
    for (auto dim : a_shape) atb_a_shape.push_back(static_cast<int64_t>(dim));
    for (auto dim : b_shape) atb_b_shape.push_back(static_cast<int64_t>(dim));
    for (auto dim : c_shape) atb_c_shape.push_back(static_cast<int64_t>(dim));
  }

  // Create LinearParam for ATB (used for MatMul)
  atb::infer::LinearParam linearParam;
  linearParam.transposeA = options_.transpose_a;
  linearParam.transposeB = options_.transpose_b;
  linearParam.hasBias = false;
  linearParam.outDataType = ACL_DT_UNDEFINED;
  linearParam.enAccum = false;
  linearParam.matmulType = atb::infer::LinearParam::MATMUL_UNDEFINED;
  linearParam.quantMode = atb::infer::LinearParam::QUANT_UNDEFINED;

  atb::Operation* op = nullptr;
  auto st = atb::CreateOperation(linearParam, &op);
  if (st != atb::NO_ERROR || op == nullptr) {
    MLLM_ERROR_EXIT(ExitCode::kAscendError, "ATB CreateOperation(MatMul) failed, status={}", static_cast<int>(st));
  }

  atb::Context* atb_ctx = getGlobalAtbContext();

  atb::Tensor atb_A, atb_B, atb_C;
  fillAtbTensorWithShape(A, atb_A, atb_a_shape);
  fillAtbTensorWithShape(B, atb_B, atb_b_shape);
  fillAtbTensorWithShape(C, atb_C, atb_c_shape);

  atb::SVector<atb::Tensor> inTensors;
  atb::SVector<atb::Tensor> outTensors;
  inTensors.push_back(atb_A);
  inTensors.push_back(atb_B);
  outTensors.push_back(atb_C);

  atb::VariantPack vp;
  vp.inTensors = inTensors;
  vp.outTensors = outTensors;

  uint64_t workspaceSize = 0;
  st = op->Setup(vp, workspaceSize, atb_ctx);
  if (st != atb::NO_ERROR) {
    MLLM_ERROR_EXIT(ExitCode::kAscendError, "ATB MatMulOp Setup failed, status={}", static_cast<int>(st));
  }

  void* workspace = nullptr;
  int workspace_block_id = -1;
  if (workspaceSize > 0) {
    auto& mem_mgr = getAscendMemoryManager();
    mem_mgr.allocateBlock(static_cast<uint32_t>(workspaceSize), workspace_block_id);
    mem_mgr.getBlockPtr(workspace_block_id, workspace);
  }

  {
    ASCEND_TIME_SCOPE("AscendMatMulOp::forward");
    st = op->Execute(vp, reinterpret_cast<uint8_t*>(workspace), workspaceSize, atb_ctx);
  }

  if (st != atb::NO_ERROR) {
    MLLM_ERROR_EXIT(ExitCode::kAscendError, "ATB MatMulOp Execute failed, status={}", static_cast<int>(st));
  }

  syncGlobalAtbStream();

  if (workspace_block_id != -1) {
    auto& mem_mgr = getAscendMemoryManager();
    mem_mgr.freeBlock(workspace_block_id);
  }

  atb::DestroyOperation(op);
}

}  // namespace mllm::ascend