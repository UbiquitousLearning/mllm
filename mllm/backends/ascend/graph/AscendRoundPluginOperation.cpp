// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/ascend/graph/AscendRoundPluginOperation.hpp"

#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <aclnnop/aclnn_round.h>
#include <atb/types.h>

#include <vector>

#include "mllm/backends/ascend/AscendCommon.hpp"

namespace mllm::ascend {

namespace MLLM_ANONYMOUS_NAMESPACE {

constexpr uint32_t INPUT_NUM = 1;
constexpr uint32_t OUTPUT_NUM = 1;
constexpr uint32_t INPUT_INDEX = 0;
constexpr uint32_t OUTPUT_INDEX = 0;

struct RoundCacheState {
  aclTensor* src_tensor{nullptr};
  aclTensor* dst_tensor{nullptr};
  aclOpExecutor* executor{nullptr};
  uint64_t workspace_size{0};
  atb::TensorDesc src_desc{};
  atb::TensorDesc dst_desc{};
  void* bound_src_ptr{nullptr};
  void* bound_dst_ptr{nullptr};
  bool repeatable{false};
  std::vector<int64_t> src_dims;
  std::vector<int64_t> src_strides;
  std::vector<int64_t> dst_dims;
  std::vector<int64_t> dst_strides;
};

bool sameTensorDesc(const atb::TensorDesc& lhs, const atb::TensorDesc& rhs) {
  if (lhs.dtype != rhs.dtype || lhs.format != rhs.format || lhs.shape.dimNum != rhs.shape.dimNum) {
    return false;
  }
  for (uint32_t i = 0; i < lhs.shape.dimNum; ++i) {
    if (lhs.shape.dims[i] != rhs.shape.dims[i]) return false;
  }
  return true;
}

aclTensor* createAclTensor(const atb::Tensor& t,
                                  std::vector<int64_t>& dims,
                                  std::vector<int64_t>& strides) {
  const int nd = static_cast<int>(t.desc.shape.dimNum);
  dims.resize(nd);
  strides.resize(nd);
  int64_t stride = 1;
  for (int i = nd - 1; i >= 0; --i) {
    dims[i] = static_cast<int64_t>(t.desc.shape.dims[i]);
    strides[i] = stride;
    stride *= dims[i];
  }
  return aclCreateTensor(dims.data(),
                         nd,
                         t.desc.dtype,
                         strides.data(),
                         /*storageOffset=*/0,
                         ACL_FORMAT_ND,
                         dims.data(),
                         nd,
                         t.deviceData);
}

void destroyCacheState(RoundCacheState*& cache) {
  if (cache == nullptr) return;
  if (cache->executor != nullptr) {
    aclDestroyAclOpExecutor(cache->executor);
    cache->executor = nullptr;
  }
  if (cache->src_tensor != nullptr) {
    aclDestroyTensor(cache->src_tensor);
    cache->src_tensor = nullptr;
  }
  if (cache->dst_tensor != nullptr) {
    aclDestroyTensor(cache->dst_tensor);
    cache->dst_tensor = nullptr;
  }
  delete cache;
  cache = nullptr;
}

atb::Status updateTensorAddrs(RoundCacheState* cache, const atb::VariantPack& variantPack) {
  if (cache == nullptr || !cache->repeatable) return atb::ERROR_INTERNAL_ERROR;
  void* src = variantPack.inTensors.at(INPUT_INDEX).deviceData;
  void* dst = variantPack.outTensors.at(OUTPUT_INDEX).deviceData;
  if (src == nullptr || dst == nullptr) return atb::ERROR_INTERNAL_ERROR;

  if (cache->bound_src_ptr != src) {
    aclnnStatus ret = aclSetInputTensorAddr(cache->executor, 0, cache->src_tensor, src);
    if (ret != ACL_SUCCESS) return atb::ERROR_INTERNAL_ERROR;
    cache->bound_src_ptr = src;
  }
  if (cache->bound_dst_ptr != dst) {
    aclnnStatus ret = aclSetOutputTensorAddr(cache->executor, 0, cache->dst_tensor, dst);
    if (ret != ACL_SUCCESS) return atb::ERROR_INTERNAL_ERROR;
    cache->bound_dst_ptr = dst;
  }
  return atb::NO_ERROR;
}

}  // namespace MLLM_ANONYMOUS_NAMESPACE

AscendRoundPluginOperation::~AscendRoundPluginOperation() {
  auto* cache = static_cast<RoundCacheState*>(cache_state_);
  destroyCacheState(cache);
  cache_state_ = nullptr;
}

std::string AscendRoundPluginOperation::GetName() const {
  return "AscendRoundPluginOperation";
}

atb::Status AscendRoundPluginOperation::InferShape(
    const atb::SVector<atb::TensorDesc>& inTensorDescs,
    atb::SVector<atb::TensorDesc>& outTensorDescs) const {
  if (inTensorDescs.size() != INPUT_NUM || outTensorDescs.size() != OUTPUT_NUM) {
    return atb::ERROR_INVALID_TENSOR_NUM;
  }
  outTensorDescs.at(OUTPUT_INDEX) = inTensorDescs.at(INPUT_INDEX);
  return atb::NO_ERROR;
}

uint32_t AscendRoundPluginOperation::GetInputNum() const { return INPUT_NUM; }
uint32_t AscendRoundPluginOperation::GetOutputNum() const { return OUTPUT_NUM; }

atb::Status AscendRoundPluginOperation::Setup(const atb::VariantPack& variantPack,
                                              uint64_t& workspace_size,
                                              atb::Context* context) {
  (void)context;
  if (variantPack.inTensors.size() != INPUT_NUM || variantPack.outTensors.size() != OUTPUT_NUM) {
    return atb::ERROR_INVALID_TENSOR_NUM;
  }

  auto* cache = static_cast<RoundCacheState*>(cache_state_);
  const auto& src_desc = variantPack.inTensors.at(INPUT_INDEX).desc;
  const auto& dst_desc = variantPack.outTensors.at(OUTPUT_INDEX).desc;
  if (cache != nullptr && cache->repeatable &&
      sameTensorDesc(cache->src_desc, src_desc) &&
      sameTensorDesc(cache->dst_desc, dst_desc)) {
    workspace_size = cache->workspace_size;
    return atb::NO_ERROR;
  }

  destroyCacheState(cache);
  cache_state_ = nullptr;

  auto* new_cache = new RoundCacheState();
  new_cache->src_desc = src_desc;
  new_cache->dst_desc = dst_desc;
  new_cache->bound_src_ptr = variantPack.inTensors.at(INPUT_INDEX).deviceData;
  new_cache->bound_dst_ptr = variantPack.outTensors.at(OUTPUT_INDEX).deviceData;
  new_cache->src_tensor = createAclTensor(variantPack.inTensors.at(INPUT_INDEX),
                                          new_cache->src_dims,
                                          new_cache->src_strides);
  new_cache->dst_tensor = createAclTensor(variantPack.outTensors.at(OUTPUT_INDEX),
                                          new_cache->dst_dims,
                                          new_cache->dst_strides);
  if (new_cache->src_tensor == nullptr || new_cache->dst_tensor == nullptr) {
    destroyCacheState(new_cache);
    return atb::ERROR_INTERNAL_ERROR;
  }

  aclnnStatus ret = aclnnRoundGetWorkspaceSize(new_cache->src_tensor,
                                               new_cache->dst_tensor,
                                               &new_cache->workspace_size,
                                               &new_cache->executor);
  if (ret != ACL_SUCCESS || new_cache->executor == nullptr) {
    destroyCacheState(new_cache);
    return atb::ERROR_INTERNAL_ERROR;
  }
  if (aclSetAclOpExecutorRepeatable(new_cache->executor) != ACL_SUCCESS) {
    destroyCacheState(new_cache);
    return atb::ERROR_INTERNAL_ERROR;
  }
  new_cache->repeatable = true;

  workspace_size = new_cache->workspace_size;
  cache_state_ = new_cache;
  return atb::NO_ERROR;
}

atb::Status AscendRoundPluginOperation::Execute(const atb::VariantPack& variantPack,
                                                uint8_t* workspace,
                                                uint64_t workspace_size,
                                                atb::Context* context) {
  (void)context;
  (void)workspace_size;
  if (variantPack.inTensors.size() != INPUT_NUM || variantPack.outTensors.size() != OUTPUT_NUM) {
    return atb::ERROR_INVALID_TENSOR_NUM;
  }
  auto* cache = static_cast<RoundCacheState*>(cache_state_);
  if (cache == nullptr || cache->executor == nullptr) {
    return atb::ERROR_INTERNAL_ERROR;
  }
  if (!sameTensorDesc(cache->src_desc, variantPack.inTensors.at(INPUT_INDEX).desc) ||
      !sameTensorDesc(cache->dst_desc, variantPack.outTensors.at(OUTPUT_INDEX).desc)) {
    return atb::ERROR_INTERNAL_ERROR;
  }

  auto st = updateTensorAddrs(cache, variantPack);
  if (st != atb::NO_ERROR) return st;

  aclError ret = aclnnRound(workspace,
                            cache->workspace_size,
                            cache->executor,
                            getGlobalAtbStream());
  return (ret == ACL_SUCCESS) ? atb::NO_ERROR : atb::ERROR_INTERNAL_ERROR;
}

atb::Operation* createRoundPluginGraphOp() {
  return new AscendRoundPluginOperation();
}

}  // namespace mllm::ascend
