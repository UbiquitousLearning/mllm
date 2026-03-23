// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/ascend/ops/AscendRoPEOp.hpp"

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

AscendRoPEOp::AscendRoPEOp(const aops::RoPEOpOptions& options) : aops::RoPEOp(options) {}

void AscendRoPEOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  if (options_.isInplace()) {
    outputs.emplace_back(inputs[0]);
  } else {
    BaseOp::setup(inputs, outputs);
  }
}

void AscendRoPEOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  // MLLM RoPE interface: 3 inputs (x, sin, cos)
  // ATB RoPE interface: 5 inputs (Q, K, cos, sin, position_ids)
  //
  // Input x should be in [B, S, H, D] format (caller should transpose if needed)
  // ATB format: [B, S, H, D] (4D) or [ntokens, hiddenSize] (2D)
  MLLM_RT_ASSERT(inputs.size() >= 3);
  MLLM_RT_ASSERT_EQ(outputs.size(), 1);

  const auto& x = inputs[0];
  // MLLM RoPE canonical order is: [x, sin, cos]
  const auto& sin_in = inputs[1];
  const auto& cos_in = inputs[2];
  auto& y = outputs[0];

  // Validate that input tensors are FP16
  if (x.dtype() != MLLM_TYPE_F16) {
    MLLM_ERROR_EXIT(ExitCode::kAscendError,
                    "AscendRoPEOp: Input tensor x must be FP16, but got dtype={}",
                    static_cast<int>(x.dtype()));
  }
  if (sin_in.dtype() != MLLM_TYPE_F16) {
    MLLM_ERROR_EXIT(ExitCode::kAscendError,
                    "AscendRoPEOp: Input tensor sin must be FP16, but got dtype={}",
                    static_cast<int>(sin_in.dtype()));
  }
  if (cos_in.dtype() != MLLM_TYPE_F16) {
    MLLM_ERROR_EXIT(ExitCode::kAscendError,
                    "AscendRoPEOp: Input tensor cos must be FP16, but got dtype={}",
                    static_cast<int>(cos_in.dtype()));
  }
  if (y.dtype() != MLLM_TYPE_F16) {
    MLLM_ERROR_EXIT(ExitCode::kAscendError,
                    "AscendRoPEOp: Output tensor must be FP16, but got dtype={}",
                    static_cast<int>(y.dtype()));
  }

  // Get dimensions from input format [B, S, H, D]
  int64_t B = x.shape()[0];
  int64_t S = x.shape()[1];
  int64_t H = x.shape()[2];
  int64_t D = x.shape()[3];

  auto& mem_mgr = getAscendMemoryManager();
  atb::Context* atb_ctx = getGlobalAtbContext();

  // Create ATB RoPE operation
  atb::infer::RopeParam ropeParam;
  ropeParam.rotaryCoeff = 2;  // Half rotation
  ropeParam.cosFormat = 0;

  atb::Operation* op = nullptr;
  auto st = atb::CreateOperation(ropeParam, &op);
  if (st != atb::NO_ERROR || op == nullptr) {
    MLLM_ERROR_EXIT(ExitCode::kAscendError, "ATB CreateOperation(RoPE) failed, status={}", static_cast<int>(st));
  }

  // Use 2D format [ntokens, hiddenSize] where ntokens = B*S, hiddenSize = H*D
  int64_t ntokens = B * S;
  int64_t hiddenSize = H * D;

  // Prepare ATB tensors for Q
  atb::Tensor atb_q;
  atb_q.desc.dtype = ACL_FLOAT16;
  atb_q.desc.format = ACL_FORMAT_ND;
  atb_q.desc.shape.dimNum = 2;
  atb_q.desc.shape.dims[0] = ntokens;
  atb_q.desc.shape.dims[1] = hiddenSize;
  atb_q.deviceData = reinterpret_cast<uint8_t*>(x.ptr<void>());
  atb_q.dataSize = x.bytes();

  // K uses the same tensor as Q (we only care about Q output)
  atb::Tensor atb_k = atb_q;

  // cos/sin tensor handling:
  // ATB RoPE expects 2D [ntokens, D], where ntokens = B*S.
  // Supported input layouts:
  //   - [S, D]
  //   - [B, S, D]
  //   - [B, S, 1, D]
  const auto& cos_shape = cos_in.shape();
  const auto& sin_shape = sin_in.shape();
  bool has_batch_dim = false;
  int64_t cos_S = 0;
  int64_t cos_D = 0;
  if (cos_shape.size() == 2) {
    cos_S = cos_shape[0];
    cos_D = cos_shape[1];
  } else if (cos_shape.size() == 3) {
    MLLM_RT_ASSERT_EQ(cos_shape[0], B);
    cos_S = cos_shape[1];
    cos_D = cos_shape[2];
    has_batch_dim = true;
  } else if (cos_shape.size() == 4) {
    MLLM_RT_ASSERT_EQ(cos_shape[0], B);
    MLLM_RT_ASSERT_EQ(cos_shape[2], 1);
    cos_S = cos_shape[1];
    cos_D = cos_shape[3];
    has_batch_dim = true;
  } else {
    MLLM_ERROR_EXIT(ExitCode::kAscendError, "AscendRoPEOp: Unsupported cos rank={}, expected 2/3/4",
                    static_cast<int>(cos_shape.size()));
  }
  MLLM_RT_ASSERT_EQ(cos_S, S);
  MLLM_RT_ASSERT_EQ(sin_shape.size(), cos_shape.size());
  MLLM_RT_ASSERT_EQ(sin_shape.back(), cos_D);
  const bool need_expand = (B > 1 && !has_batch_dim);

  // Allocate expanded cos/sin tensors on device if B > 1
  int cos_expanded_block_id = -1;
  int sin_expanded_block_id = -1;
  void* cos_expanded_ptr = nullptr;
  void* sin_expanded_ptr = nullptr;
  size_t expanded_size = static_cast<size_t>(ntokens * cos_D) * sizeof(uint16_t);  // FP16

  if (need_expand) {
    // Need to expand cos/sin from [S, D] to [B*S, D]
    mem_mgr.allocateBlock(static_cast<uint32_t>(expanded_size), cos_expanded_block_id);
    mem_mgr.getBlockPtr(cos_expanded_block_id, cos_expanded_ptr);
    mem_mgr.allocateBlock(static_cast<uint32_t>(expanded_size), sin_expanded_block_id);
    mem_mgr.getBlockPtr(sin_expanded_block_id, sin_expanded_ptr);

    // Copy cos/sin data for each batch
    size_t single_batch_size = static_cast<size_t>(S * cos_D) * sizeof(uint16_t);
    for (int64_t b = 0; b < B; ++b) {
      aclrtMemcpy(reinterpret_cast<uint8_t*>(cos_expanded_ptr) + b * single_batch_size,
                  single_batch_size, cos_in.ptr<void>(), single_batch_size, ACL_MEMCPY_DEVICE_TO_DEVICE);
      aclrtMemcpy(reinterpret_cast<uint8_t*>(sin_expanded_ptr) + b * single_batch_size,
                  single_batch_size, sin_in.ptr<void>(), single_batch_size, ACL_MEMCPY_DEVICE_TO_DEVICE);
    }
  }

  const size_t rope_table_size = static_cast<size_t>(ntokens * cos_D) * sizeof(uint16_t);

  // cos tensor - ATB requires 2D [ntokens, headDim]
  atb::Tensor atb_cos;
  atb_cos.desc.dtype = ACL_FLOAT16;
  atb_cos.desc.format = ACL_FORMAT_ND;
  atb_cos.desc.shape.dimNum = 2;
  atb_cos.desc.shape.dims[0] = ntokens;
  atb_cos.desc.shape.dims[1] = cos_D;
  if (need_expand) {
    atb_cos.deviceData = reinterpret_cast<uint8_t*>(cos_expanded_ptr);
    atb_cos.dataSize = rope_table_size;
  } else {
    atb_cos.deviceData = reinterpret_cast<uint8_t*>(cos_in.ptr<void>());
    atb_cos.dataSize = rope_table_size;
  }

  // sin tensor - ATB requires 2D [ntokens, headDim]
  atb::Tensor atb_sin;
  atb_sin.desc.dtype = ACL_FLOAT16;
  atb_sin.desc.format = ACL_FORMAT_ND;
  atb_sin.desc.shape.dimNum = 2;
  atb_sin.desc.shape.dims[0] = ntokens;
  atb_sin.desc.shape.dims[1] = cos_D;
  if (need_expand) {
    atb_sin.deviceData = reinterpret_cast<uint8_t*>(sin_expanded_ptr);
    atb_sin.dataSize = rope_table_size;
  } else {
    atb_sin.deviceData = reinterpret_cast<uint8_t*>(sin_in.ptr<void>());
    atb_sin.dataSize = rope_table_size;
  }

  // Create position_ids tensor on device
  // For multi-batch case, we need position IDs for each token in each batch
  // Position IDs repeat for each batch: [0, 1, 2, ..., S-1, 0, 1, 2, ..., S-1, ...]
  int pos_ids_block_id = -1;
  void* pos_ids_ptr = nullptr;
  size_t pos_ids_size = static_cast<size_t>(ntokens) * sizeof(int32_t);
  mem_mgr.allocateBlock(static_cast<uint32_t>(pos_ids_size), pos_ids_block_id);
  mem_mgr.getBlockPtr(pos_ids_block_id, pos_ids_ptr);

  // Generate position_ids: for each batch, positions are 0, 1, 2, ..., S-1
  std::vector<int32_t> pos_ids_host(ntokens);
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t s = 0; s < S; ++s) {
      pos_ids_host[b * S + s] = static_cast<int32_t>(s);
    }
  }
  aclrtMemcpy(pos_ids_ptr, pos_ids_size, pos_ids_host.data(), pos_ids_size, ACL_MEMCPY_HOST_TO_DEVICE);

  atb::Tensor atb_pos_ids;
  atb_pos_ids.desc.dtype = ACL_INT32;
  atb_pos_ids.desc.format = ACL_FORMAT_ND;
  atb_pos_ids.desc.shape.dimNum = 1;
  atb_pos_ids.desc.shape.dims[0] = ntokens;
  atb_pos_ids.deviceData = reinterpret_cast<uint8_t*>(pos_ids_ptr);
  atb_pos_ids.dataSize = pos_ids_size;

  // Output tensor for Q - use 2D format [ntokens, hiddenSize]
  atb::Tensor atb_y;
  atb_y.desc.dtype = ACL_FLOAT16;
  atb_y.desc.format = ACL_FORMAT_ND;
  atb_y.desc.shape.dimNum = 2;
  atb_y.desc.shape.dims[0] = ntokens;
  atb_y.desc.shape.dims[1] = hiddenSize;
  atb_y.deviceData = reinterpret_cast<uint8_t*>(y.ptr<void>());
  atb_y.dataSize = y.bytes();

  // ATB RoPE requires 2 outputs: Q_out and K_out
  // We allocate a dummy K_out tensor since we only need Q_out
  int k_out_block_id = -1;
  void* k_out_ptr = nullptr;
  mem_mgr.allocateBlock(static_cast<uint32_t>(y.bytes()), k_out_block_id);
  mem_mgr.getBlockPtr(k_out_block_id, k_out_ptr);

  atb::Tensor atb_k_out;
  atb_k_out.desc.dtype = ACL_FLOAT16;
  atb_k_out.desc.format = ACL_FORMAT_ND;
  atb_k_out.desc.shape.dimNum = 2;
  atb_k_out.desc.shape.dims[0] = ntokens;
  atb_k_out.desc.shape.dims[1] = hiddenSize;
  atb_k_out.deviceData = reinterpret_cast<uint8_t*>(k_out_ptr);
  atb_k_out.dataSize = y.bytes();

  // ATB RoPE: 5 inputs [Q, K, cos, sin, pos_ids], 2 outputs [Q_out, K_out]
  atb::SVector<atb::Tensor> inTensors;
  atb::SVector<atb::Tensor> outTensors;
  inTensors.push_back(atb_q);
  inTensors.push_back(atb_k);
  inTensors.push_back(atb_cos);
  inTensors.push_back(atb_sin);
  inTensors.push_back(atb_pos_ids);
  outTensors.push_back(atb_y);
  outTensors.push_back(atb_k_out);

  atb::VariantPack vp;
  vp.inTensors = inTensors;
  vp.outTensors = outTensors;

  uint64_t workspaceSize = 0;
  st = op->Setup(vp, workspaceSize, atb_ctx);
  if (st != atb::NO_ERROR) {
    mem_mgr.freeBlock(pos_ids_block_id);
    mem_mgr.freeBlock(k_out_block_id);
    if (cos_expanded_block_id != -1) {
      mem_mgr.freeBlock(cos_expanded_block_id);
    }
    if (sin_expanded_block_id != -1) {
      mem_mgr.freeBlock(sin_expanded_block_id);
    }
    atb::DestroyOperation(op);
    MLLM_ERROR_EXIT(ExitCode::kAscendError, "ATB RoPEOp Setup failed, status={}", static_cast<int>(st));
  }

  void* workspace = nullptr;
  int workspace_block_id = -1;
  if (workspaceSize > 0) {
    mem_mgr.allocateBlock(static_cast<uint32_t>(workspaceSize), workspace_block_id);
    mem_mgr.getBlockPtr(workspace_block_id, workspace);
  }

  {
    //ASCEND_TIME_SCOPE("AscendRoPEOp::forward");
    st = op->Execute(vp, reinterpret_cast<uint8_t*>(workspace), workspaceSize, atb_ctx);
  }

  if (st != atb::NO_ERROR) {
    if (workspace_block_id != -1) {
      mem_mgr.freeBlock(workspace_block_id);
    }
    mem_mgr.freeBlock(pos_ids_block_id);
    mem_mgr.freeBlock(k_out_block_id);
    if (cos_expanded_block_id != -1) {
      mem_mgr.freeBlock(cos_expanded_block_id);
    }
    if (sin_expanded_block_id != -1) {
      mem_mgr.freeBlock(sin_expanded_block_id);
    }
    atb::DestroyOperation(op);
    MLLM_ERROR_EXIT(ExitCode::kAscendError, "ATB RoPEOp Execute failed, status={}", static_cast<int>(st));
  }

  syncGlobalAtbStream();

  // Cleanup
  atb::DestroyOperation(op);
  if (workspace_block_id != -1) {
    mem_mgr.freeBlock(workspace_block_id);
  }
  mem_mgr.freeBlock(pos_ids_block_id);
  mem_mgr.freeBlock(k_out_block_id);
  if (cos_expanded_block_id != -1) {
    mem_mgr.freeBlock(cos_expanded_block_id);
  }
  if (sin_expanded_block_id != -1) {
    mem_mgr.freeBlock(sin_expanded_block_id);
  }
}

}  // namespace mllm::ascend
