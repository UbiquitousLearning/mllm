// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/ascend/ops/AscendEmbeddingOp.hpp"

#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <aclnnop/aclnn_embedding.h>

#include "mllm/utils/Common.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/backends/ascend/memory/AscendMemoryManager.hpp"
#include "mllm/backends/ascend/AscendCommon.hpp"

namespace mllm::ascend {

namespace MLLM_ANONYMOUS_NAMESPACE {

// Helper function to convert MLLM dtype to ACL dtype
aclDataType toAclDataType(DataTypes dtype) {
  switch (dtype) {
    case DataTypes::kFloat32: return ACL_FLOAT;
    case DataTypes::kFloat16: return ACL_FLOAT16;
    case DataTypes::kInt32: return ACL_INT32;
    case DataTypes::kInt64: return ACL_INT64;
    case DataTypes::kInt8: return ACL_INT8;
    case DataTypes::kUInt8: return ACL_UINT8;
    default:
      MLLM_ERROR_EXIT(ExitCode::kAscendError, "Unsupported dtype for ACLNN: {}", static_cast<int>(dtype));
  }
}

// Helper function to calculate strides for contiguous tensor
std::vector<int64_t> calcStrides(const std::vector<int64_t>& dims) {
  int ndim = static_cast<int>(dims.size());
  std::vector<int64_t> strides(ndim, 1);
  for (int i = ndim - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * dims[i + 1];
  }
  return strides;
}

// Helper function to create aclTensor from MLLM Tensor
aclTensor* createAclTensor(const Tensor& tensor) {
  const auto& shape = tensor.shape();
  int ndim = static_cast<int>(shape.size());

  // Convert shape to int64_t
  std::vector<int64_t> dims(ndim);
  for (int i = 0; i < ndim; ++i) {
    dims[i] = static_cast<int64_t>(shape[i]);
  }

  // Calculate strides (row-major, contiguous)
  std::vector<int64_t> strides = calcStrides(dims);

  aclDataType dtype = toAclDataType(tensor.dtype());

  // Create aclTensor
  aclTensor* acl_tensor = aclCreateTensor(
      dims.data(), static_cast<uint64_t>(ndim),
      dtype,
      strides.data(),
      0,  // offset
      ACL_FORMAT_ND,
      dims.data(), static_cast<uint64_t>(ndim),
      tensor.ptr<void>()
  );

  return acl_tensor;
}

}  // namespace MLLM_ANONYMOUS_NAMESPACE

AscendEmbeddingOp::AscendEmbeddingOp(const aops::EmbeddingOpOptions& options) : aops::EmbeddingOp(options) {}

void AscendEmbeddingOp::load(const ParameterFile::ptr_t& ploader) {
  // Guard: during LayerImpl::to() the temp ploader may be empty.
  if (!ploader->has(getName() + ".weight")) { return; }

  // First call parent's load to get weight from file (on CPU)
  aops::EmbeddingOp::load(ploader);

  // Convert weight to FP16 and move to Ascend NPU
  if (!weight_.isNil()) {
    weight_ = convertTensorToAscendFP16(weight_);
  }
}

void AscendEmbeddingOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  aops::EmbeddingOp::reshape(inputs, outputs);
}

void AscendEmbeddingOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  BaseOp::setup(inputs, outputs);
}

void AscendEmbeddingOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  MLLM_RT_ASSERT_EQ(inputs.size(), 1);
  MLLM_RT_ASSERT_EQ(outputs.size(), 1);

  const auto& indices = inputs[0];  // Indices tensor [B, S]
  auto& output = outputs[0];        // Output tensor [B, S, hidden_size]

  // weight_ is the embedding table [vocab_size, hidden_size]

  // Create aclTensors
  aclTensor* acl_weight = createAclTensor(weight_);
  aclTensor* acl_indices = createAclTensor(indices);
  aclTensor* acl_output = createAclTensor(output);

  if (acl_weight == nullptr || acl_indices == nullptr || acl_output == nullptr) {
    MLLM_ERROR_EXIT(ExitCode::kAscendError, "AscendEmbeddingOp: Failed to create aclTensor");
  }

  // Get workspace size and executor using aclnnEmbedding
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;

  auto ret = aclnnEmbeddingGetWorkspaceSize(acl_weight, acl_indices, acl_output, &workspace_size, &executor);
  if (ret != 0) {
    aclDestroyTensor(acl_weight);
    aclDestroyTensor(acl_indices);
    aclDestroyTensor(acl_output);
    MLLM_ERROR_EXIT(ExitCode::kAscendError, "aclnnEmbeddingGetWorkspaceSize failed, error={}", ret);
  }

  // Allocate workspace
  void* workspace = nullptr;
  int workspace_block_id = -1;
  if (workspace_size > 0) {
    auto& mem_mgr = getAscendMemoryManager();
    mem_mgr.allocateBlock(static_cast<uint32_t>(workspace_size), workspace_block_id);
    mem_mgr.getBlockPtr(workspace_block_id, workspace);
  }

  // Execute embedding
  aclrtStream stream = getGlobalAtbStream();
  {
    //ASCEND_TIME_SCOPE("AscendEmbeddingOp::forward");
    ret = aclnnEmbedding(workspace, workspace_size, executor, stream);
  }

  // Synchronize before checking result and cleanup
  aclrtSynchronizeStream(stream);

  if (ret != 0) {
    if (workspace_block_id != -1) {
      auto& mem_mgr = getAscendMemoryManager();
      mem_mgr.freeBlock(workspace_block_id);
    }
    aclDestroyTensor(acl_weight);
    aclDestroyTensor(acl_indices);
    aclDestroyTensor(acl_output);
    MLLM_ERROR_EXIT(ExitCode::kAscendError, "aclnnEmbedding failed, error={}", ret);
  }

  // Cleanup
  if (workspace_block_id != -1) {
    auto& mem_mgr = getAscendMemoryManager();
    mem_mgr.freeBlock(workspace_block_id);
  }

  aclDestroyTensor(acl_weight);
  aclDestroyTensor(acl_indices);
  aclDestroyTensor(acl_output);
  // Note: Don't call aclDestroyAclOpExecutor - ACLNN manages executor lifecycle internally
}

}  // namespace mllm::ascend
