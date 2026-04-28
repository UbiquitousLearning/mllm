// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/ascend/graph/AscendAttentionWithKVCachePluginOperation.hpp"

#include <acl/acl.h>
#include <atb/infer_op_params.h>
#include <atb/utils.h>

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include "mllm/backends/ascend/AscendCommon.hpp"
#include "mllm/backends/ascend/graph/AscendCausalMaskTensorPluginOperation.hpp"
#include "mllm/backends/ascend/graph/AscendGraphBuilder.hpp"

namespace mllm::ascend {

namespace MLLM_ANONYMOUS_NAMESPACE {

constexpr uint32_t INPUT_NUM = 7;
constexpr uint32_t OUTPUT_NUM = 1;
constexpr uint32_t QUERY_INPUT_INDEX = 0;
constexpr uint32_t KEY_NEW_INPUT_INDEX = 1;
constexpr uint32_t VALUE_NEW_INPUT_INDEX = 2;
constexpr uint32_t KEY_CACHE_INPUT_INDEX = 3;
constexpr uint32_t VALUE_CACHE_INPUT_INDEX = 4;
constexpr uint32_t CURRENT_SEQ_LEN_INPUT_INDEX = 5;
constexpr uint32_t ATTN_SCALE_INPUT_INDEX = 6;
constexpr uint32_t OUTPUT_INDEX = 0;

struct WorkspacePlan {
  uint64_t key_history_bytes{0};
  uint64_t value_history_bytes{0};
  uint64_t subgraph_workspace_bytes{0};

  [[nodiscard]] uint64_t totalBytes() const {
    return key_history_bytes + value_history_bytes + subgraph_workspace_bytes;
  }
};

inline uint64_t alignUp(uint64_t value, uint64_t alignment = 512) {
  return ((value + alignment - 1) / alignment) * alignment;
}

inline atb::Tensor makeTensorWithShape(aclDataType dtype,
                                       const std::vector<int64_t>& shape,
                                       uint8_t* device_data = nullptr) {
  atb::Tensor tensor;
  tensor.desc.dtype = dtype;
  tensor.desc.format = ACL_FORMAT_ND;
  tensor.desc.shape.dimNum = static_cast<uint64_t>(shape.size());
  for (size_t i = 0; i < shape.size(); ++i) {
    tensor.desc.shape.dims[i] = shape[i];
  }
  tensor.deviceData = device_data;
  tensor.dataSize = atb::Utils::GetTensorSize(tensor);
  return tensor;
}

inline atb::Tensor makeTensorView(const atb::Tensor& base,
                                  const std::vector<int64_t>& shape,
                                  uint8_t* device_data = nullptr) {
  auto* base_device_data = static_cast<uint8_t*>(base.deviceData);
  return makeTensorWithShape(base.desc.dtype, shape, device_data != nullptr ? device_data : base_device_data);
}

inline atb::Status readCurrentSeqLen(const atb::Tensor& seq_len_tensor, int32_t& current_seq_len) {
  if (seq_len_tensor.desc.dtype != ACL_INT32 || seq_len_tensor.dataSize < sizeof(int32_t) || seq_len_tensor.deviceData == nullptr) {
    return atb::ERROR_INVALID_TENSOR_DTYPE;
  }
  auto ret = aclrtMemcpy(&current_seq_len,
                         sizeof(int32_t),
                         seq_len_tensor.deviceData,
                         sizeof(int32_t),
                         ACL_MEMCPY_DEVICE_TO_HOST);
  return ret == ACL_SUCCESS ? atb::NO_ERROR : atb::ERROR_RT_FAIL;
}

inline atb::Status appendToBaseCache(const atb::Tensor& src_new,
                                     const atb::Tensor& dst_full_cache,
                                     int32_t current_seq_len,
                                     int32_t max_cache_length) {
  const auto& src_shape = src_new.desc.shape;
  const auto& dst_shape = dst_full_cache.desc.shape;
  if (src_shape.dimNum != 4 || dst_shape.dimNum != 4) {
    return atb::ERROR_INVALID_TENSOR_DIM_NUM;
  }

  const int64_t batch = src_shape.dims[0];
  const int64_t heads = src_shape.dims[1];
  const int64_t append_seq_len = src_shape.dims[2];
  const int64_t head_dim = src_shape.dims[3];
  if (batch != 1
      || dst_shape.dims[0] != batch
      || dst_shape.dims[1] != heads
      || dst_shape.dims[2] != max_cache_length
      || dst_shape.dims[3] != head_dim
      || current_seq_len + append_seq_len > max_cache_length) {
    return atb::ERROR_INVALID_TENSOR_NUM;
  }

  const size_t elem_size = aclDataTypeSize(src_new.desc.dtype);
  const size_t src_head_stride = static_cast<size_t>(append_seq_len) * head_dim * elem_size;
  const size_t dst_head_stride = static_cast<size_t>(max_cache_length) * head_dim * elem_size;
  const size_t seq_offset = static_cast<size_t>(current_seq_len) * head_dim * elem_size;

  const auto* src_base = static_cast<const uint8_t*>(src_new.deviceData);
  auto* dst_base = static_cast<uint8_t*>(dst_full_cache.deviceData);
  auto stream = getGlobalAtbStream();
  for (int64_t h = 0; h < heads; ++h) {
    auto ret = aclrtMemcpyAsync(dst_base + h * dst_head_stride + seq_offset,
                                src_head_stride,
                                src_base + h * src_head_stride,
                                src_head_stride,
                                ACL_MEMCPY_DEVICE_TO_DEVICE,
                                stream);
    if (ret != ACL_SUCCESS) {
      return atb::ERROR_RT_FAIL;
    }
  }
  return atb::NO_ERROR;
}

inline atb::Status copyCachePrefixToContiguous(const atb::Tensor& src_full_cache,
                                               int32_t total_seq,
                                               int32_t repeat_times,
                                               atb::Tensor& dst_history_contiguous) {
  const auto& src_shape = src_full_cache.desc.shape;
  const auto& dst_shape = dst_history_contiguous.desc.shape;
  if (src_shape.dimNum != 4 || dst_shape.dimNum != 3) {
    return atb::ERROR_INVALID_TENSOR_DIM_NUM;
  }

  const int64_t batch = src_shape.dims[0];
  const int64_t src_heads = src_shape.dims[1];
  const int64_t max_cache_length = src_shape.dims[2];
  const int64_t head_dim = src_shape.dims[3];
  const int64_t dst_heads = src_heads * repeat_times;
  if (batch != 1
      || dst_shape.dims[0] != dst_heads
      || dst_shape.dims[1] < total_seq
      || dst_shape.dims[2] != head_dim
      || total_seq > max_cache_length) {
    return atb::ERROR_INVALID_TENSOR_NUM;
  }

  const size_t elem_size = aclDataTypeSize(src_full_cache.desc.dtype);
  const size_t src_head_stride = static_cast<size_t>(max_cache_length) * head_dim * elem_size;
  const size_t dst_head_stride = static_cast<size_t>(dst_shape.dims[1]) * head_dim * elem_size;
  const size_t copy_bytes = static_cast<size_t>(total_seq) * head_dim * elem_size;

  const auto* src_base = static_cast<const uint8_t*>(src_full_cache.deviceData);
  auto* dst_base = static_cast<uint8_t*>(dst_history_contiguous.deviceData);
  auto stream = getGlobalAtbStream();
  for (int64_t h = 0; h < src_heads; ++h) {
    const auto* src_ptr = src_base + h * src_head_stride;
    for (int32_t r = 0; r < repeat_times; ++r) {
      auto* dst_ptr = dst_base + (h * repeat_times + r) * dst_head_stride;
      auto ret = aclrtMemcpyAsync(dst_ptr,
                                  copy_bytes,
                                  src_ptr,
                                  copy_bytes,
                                  ACL_MEMCPY_DEVICE_TO_DEVICE,
                                  stream);
      if (ret != ACL_SUCCESS) {
        return atb::ERROR_RT_FAIL;
      }
    }
  }
  return atb::NO_ERROR;
}

inline void fillVariantPack(atb::VariantPack& pack,
                            const std::initializer_list<atb::Tensor>& inputs,
                            const std::initializer_list<atb::Tensor>& outputs) {
  pack.inTensors.clear();
  pack.outTensors.clear();
  for (const auto& tensor : inputs) {
    pack.inTensors.push_back(tensor);
  }
  for (const auto& tensor : outputs) {
    pack.outTensors.push_back(tensor);
  }
}

inline atb::Operation* createMatMulOp(bool transpose_a, bool transpose_b) {
  atb::infer::LinearParam param;
  param.transposeA = transpose_a;
  param.transposeB = transpose_b;
  param.hasBias = false;
  param.outDataType = ACL_DT_UNDEFINED;
  param.enAccum = false;
  param.matmulType = atb::infer::LinearParam::MATMUL_UNDEFINED;
  param.quantMode = atb::infer::LinearParam::QUANT_UNDEFINED;

  atb::Operation* op = nullptr;
  MLLM_ATB_CHECK(atb::CreateOperation(param, &op));
  return op;
}

inline atb::Operation* createMulOp() {
  atb::infer::ElewiseParam param;
  param.elewiseType = atb::infer::ElewiseParam::ELEWISE_MUL;

  atb::Operation* op = nullptr;
  MLLM_ATB_CHECK(atb::CreateOperation(param, &op));
  return op;
}

inline atb::Operation* createAddOp() {
  atb::infer::ElewiseParam param;
  param.elewiseType = atb::infer::ElewiseParam::ELEWISE_ADD;

  atb::Operation* op = nullptr;
  MLLM_ATB_CHECK(atb::CreateOperation(param, &op));
  return op;
}

inline atb::Operation* createSoftmaxOp(int axis) {
  atb::infer::SoftmaxParam param;
  param.axes.push_back(axis);

  atb::Operation* op = nullptr;
  MLLM_ATB_CHECK(atb::CreateOperation(param, &op));
  return op;
}

}  // namespace MLLM_ANONYMOUS_NAMESPACE

AscendAttentionWithKVCachePluginOperation::AscendAttentionWithKVCachePluginOperation(int32_t num_attention_heads,
                                                                                      int32_t num_key_value_heads,
                                                                                      int32_t head_dim,
                                                                                      int32_t max_cache_length,
                                                                                      bool sliding_window,
                                                                                      int32_t window_size,
                                                                                      std::string name_suffix,
                                                                                      int32_t setup_bucket_size)
    : num_attention_heads_(num_attention_heads),
      num_key_value_heads_(num_key_value_heads),
      num_key_value_groups_(num_attention_heads / num_key_value_heads),
      head_dim_(head_dim),
      max_cache_length_(max_cache_length),
      sliding_window_(sliding_window),
      window_size_(window_size),
      name_suffix_(std::move(name_suffix)),
      setup_bucket_size_(setup_bucket_size) {
  buildAttentionSubgraph();
}

AscendAttentionWithKVCachePluginOperation::~AscendAttentionWithKVCachePluginOperation() {
  if (prefill_subgraph_op_ != nullptr) {
    atb::DestroyOperation(prefill_subgraph_op_);
    prefill_subgraph_op_ = nullptr;
  }
  if (decode_subgraph_op_ != nullptr) {
    atb::DestroyOperation(decode_subgraph_op_);
    decode_subgraph_op_ = nullptr;
  }
  // The built graph keeps references to its child operations and releases them
  // during graph destruction. Clearing the raw pointers here avoids accidental
  // double-destroy during plugin teardown.
  matmul_qk_op_ = nullptr;
  scale_mul_op_ = nullptr;
  mask_tensor_op_ = nullptr;
  mask_add_op_ = nullptr;
  softmax_op_ = nullptr;
  matmul_av_op_ = nullptr;
  decode_matmul_qk_op_ = nullptr;
  decode_scale_mul_op_ = nullptr;
  decode_mask_tensor_op_ = nullptr;
  decode_mask_add_op_ = nullptr;
  decode_softmax_op_ = nullptr;
  decode_matmul_av_op_ = nullptr;
}

int32_t AscendAttentionWithKVCachePluginOperation::bucketedTotalSeq(int32_t total_seq) const {
  if (setup_bucket_size_ <= 0) {
    return total_seq;
  }
  const int32_t rounded = ((total_seq + setup_bucket_size_ - 1) / setup_bucket_size_) * setup_bucket_size_;
  return std::min(max_cache_length_, rounded);
}

void AscendAttentionWithKVCachePluginOperation::buildAttentionSubgraph() {
  AscendGraphBuilder builder;
  builder.beginGraph(
      "AscendAttentionWithKVCacheSubGraph" + name_suffix_,
      {"query_3d", "key_history_3d", "value_history_3d", "attn_scale", "current_seq_len"},
      {"output_3d"},
      [](const atb::SVector<atb::TensorDesc>& inTensorDescs,
         atb::SVector<atb::TensorDesc>& outTensorDescs) -> atb::Status {
        if (inTensorDescs.empty() || outTensorDescs.empty()) {
          return atb::NO_ERROR;
        }
        outTensorDescs.at(OUTPUT_INDEX) = inTensorDescs.at(QUERY_INPUT_INDEX);
        return atb::NO_ERROR;
      });

  matmul_qk_op_ = createMatMulOp(false, true);
  scale_mul_op_ = createMulOp();
  mask_tensor_op_ = createCausalMaskTensorPluginGraphOp(sliding_window_, window_size_);
  mask_add_op_ = createAddOp();
  softmax_op_ = createSoftmaxOp(3);
  matmul_av_op_ = createMatMulOp(false, false);

  builder.addOperation(matmul_qk_op_, {"query_3d", "key_history_3d"}, {"scores_3d"});
  builder.reshape(
      "scores_3d",
      [this](const atb::Dims& oldShape, atb::Dims& newShape) {
        newShape.dimNum = 4;
        newShape.dims[0] = 1;
        newShape.dims[1] = num_attention_heads_;
        newShape.dims[2] = oldShape.dims[1] / num_key_value_groups_;
        newShape.dims[3] = oldShape.dims[2];
      },
      "scores_4d");
  builder.addOperation(scale_mul_op_, {"scores_4d", "attn_scale"}, {"scaled_scores_4d"});
  builder.addOperation(mask_tensor_op_, {"scaled_scores_4d", "current_seq_len"}, {"mask_tensor_4d"});
  builder.addOperation(mask_add_op_, {"scaled_scores_4d", "mask_tensor_4d"}, {"masked_scores_4d"});
  builder.addOperation(softmax_op_, {"masked_scores_4d"}, {"probs_4d"});
  builder.reshape(
      "probs_4d",
      [this](const atb::Dims& oldShape, atb::Dims& newShape) {
        newShape.dimNum = 3;
        newShape.dims[0] = num_key_value_heads_;
        newShape.dims[1] = num_key_value_groups_ * oldShape.dims[2];
        newShape.dims[2] = oldShape.dims[3];
      },
      "probs_3d");
  builder.addOperation(matmul_av_op_, {"probs_3d", "value_history_3d"}, {"output_3d"});

  prefill_subgraph_op_ = builder.build();

  AscendGraphBuilder decode_builder;
  decode_builder.beginGraph(
      "AscendAttentionWithKVCacheDecodeSubGraph" + name_suffix_,
      {"query_3d", "key_history_3d", "value_history_3d", "attn_scale", "current_seq_len"},
      {"output_3d"},
      [](const atb::SVector<atb::TensorDesc>& inTensorDescs,
         atb::SVector<atb::TensorDesc>& outTensorDescs) -> atb::Status {
        if (inTensorDescs.empty() || outTensorDescs.empty()) {
          return atb::NO_ERROR;
        }
        outTensorDescs.at(OUTPUT_INDEX) = inTensorDescs.at(QUERY_INPUT_INDEX);
        return atb::NO_ERROR;
      });

  decode_matmul_qk_op_ = createMatMulOp(false, true);
  decode_scale_mul_op_ = createMulOp();
  if (setup_bucket_size_ > 0) {
    decode_mask_tensor_op_ = createCausalMaskTensorPluginGraphOp(sliding_window_, window_size_);
    decode_mask_add_op_ = createAddOp();
  }
  decode_softmax_op_ = createSoftmaxOp(3);
  decode_matmul_av_op_ = createMatMulOp(false, false);

  decode_builder.addOperation(decode_matmul_qk_op_, {"query_3d", "key_history_3d"}, {"scores_3d"});
  decode_builder.reshape(
      "scores_3d",
      [this](const atb::Dims& oldShape, atb::Dims& newShape) {
        newShape.dimNum = 4;
        newShape.dims[0] = 1;
        newShape.dims[1] = num_attention_heads_;
        newShape.dims[2] = oldShape.dims[1] / num_key_value_groups_;
        newShape.dims[3] = oldShape.dims[2];
      },
      "scores_4d");
  decode_builder.addOperation(decode_scale_mul_op_, {"scores_4d", "attn_scale"}, {"scaled_scores_4d"});
  if (setup_bucket_size_ > 0) {
    decode_builder.addOperation(decode_mask_tensor_op_, {"scaled_scores_4d", "current_seq_len"}, {"decode_mask_tensor_4d"});
    decode_builder.addOperation(decode_mask_add_op_, {"scaled_scores_4d", "decode_mask_tensor_4d"}, {"decode_masked_scores_4d"});
    decode_builder.addOperation(decode_softmax_op_, {"decode_masked_scores_4d"}, {"probs_4d"});
  } else {
    decode_builder.addOperation(decode_softmax_op_, {"scaled_scores_4d"}, {"probs_4d"});
  }
  decode_builder.reshape(
      "probs_4d",
      [this](const atb::Dims& oldShape, atb::Dims& newShape) {
        newShape.dimNum = 3;
        newShape.dims[0] = num_key_value_heads_;
        newShape.dims[1] = num_key_value_groups_ * oldShape.dims[2];
        newShape.dims[2] = oldShape.dims[3];
      },
      "probs_3d");
  decode_builder.addOperation(decode_matmul_av_op_, {"probs_3d", "value_history_3d"}, {"output_3d"});

  decode_subgraph_op_ = decode_builder.build();
}

std::string AscendAttentionWithKVCachePluginOperation::GetName() const {
  return "AscendAttentionWithKVCachePluginOperation" + name_suffix_;
}

uint32_t AscendAttentionWithKVCachePluginOperation::GetInputNum() const {
  return INPUT_NUM;
}

uint32_t AscendAttentionWithKVCachePluginOperation::GetOutputNum() const {
  return OUTPUT_NUM;
}

atb::Status AscendAttentionWithKVCachePluginOperation::InferShape(
    const atb::SVector<atb::TensorDesc>& inTensorDescs,
    atb::SVector<atb::TensorDesc>& outTensorDescs) const {
  if (inTensorDescs.size() != GetInputNum() || outTensorDescs.size() != OUTPUT_NUM) {
    return atb::ERROR_INVALID_TENSOR_NUM;
  }
  outTensorDescs.at(OUTPUT_INDEX) = inTensorDescs.at(QUERY_INPUT_INDEX);
  return atb::NO_ERROR;
}

atb::Status AscendAttentionWithKVCachePluginOperation::Setup(const atb::VariantPack& variantPack,
                                                             uint64_t& workspace_size,
                                                             atb::Context* context) {
  if (variantPack.inTensors.size() != GetInputNum() || variantPack.outTensors.size() != OUTPUT_NUM) {
    return atb::ERROR_INVALID_TENSOR_NUM;
  }
  if (context == nullptr || prefill_subgraph_op_ == nullptr || decode_subgraph_op_ == nullptr) {
    return atb::ERROR_INVALID_TENSOR_NUM;
  }

  int32_t current_seq_len = 0;
  auto st = readCurrentSeqLen(variantPack.inTensors.at(CURRENT_SEQ_LEN_INPUT_INDEX), current_seq_len);
  if (st != atb::NO_ERROR) {
    return st;
  }

  const auto& query = variantPack.inTensors.at(QUERY_INPUT_INDEX);
  const int64_t batch = query.desc.shape.dims[0];
  const int64_t q_heads = query.desc.shape.dims[1];
  const int64_t seq_q = query.desc.shape.dims[2];
  const int64_t total_seq = current_seq_len + seq_q;
  const int64_t bucket_total_seq = bucketedTotalSeq(static_cast<int32_t>(total_seq));
  if (batch != 1 || q_heads != num_attention_heads_ || query.desc.shape.dims[3] != head_dim_
      || total_seq > max_cache_length_ || bucket_total_seq > max_cache_length_) {
    return atb::ERROR_INVALID_TENSOR_NUM;
  }

  const auto history_shape_3d = std::vector<int64_t>{batch * num_key_value_heads_, bucket_total_seq, head_dim_};
  WorkspacePlan plan;
  plan.key_history_bytes = alignUp(makeTensorWithShape(query.desc.dtype, history_shape_3d).dataSize);
  plan.value_history_bytes = alignUp(makeTensorWithShape(query.desc.dtype, history_shape_3d).dataSize);

  atb::VariantPack subgraph_pack;
  fillVariantPack(
      subgraph_pack,
      {makeTensorView(query, {batch * num_key_value_heads_, num_key_value_groups_ * seq_q, head_dim_}),
       makeTensorWithShape(query.desc.dtype, history_shape_3d),
       makeTensorWithShape(query.desc.dtype, history_shape_3d),
       variantPack.inTensors.at(ATTN_SCALE_INPUT_INDEX),
       variantPack.inTensors.at(CURRENT_SEQ_LEN_INPUT_INDEX)},
      {makeTensorWithShape(query.desc.dtype,
                           {batch * num_key_value_heads_, num_key_value_groups_ * seq_q, head_dim_})});

  atb::Operation* selected_subgraph = seq_q > 1 ? prefill_subgraph_op_ : decode_subgraph_op_;
  uint64_t subgraph_workspace = 0;
  st = selected_subgraph->Setup(subgraph_pack, subgraph_workspace, context);
  if (st != atb::NO_ERROR) {
    return st;
  }
  if (seq_q > 1) {
    prefill_subgraph_workspace_bytes_ = alignUp(subgraph_workspace);
    plan.subgraph_workspace_bytes = prefill_subgraph_workspace_bytes_;
  } else {
    decode_subgraph_workspace_bytes_ = alignUp(subgraph_workspace);
    plan.subgraph_workspace_bytes = decode_subgraph_workspace_bytes_;
  }

  workspace_size = plan.totalBytes();
  return atb::NO_ERROR;
}

atb::Status AscendAttentionWithKVCachePluginOperation::Execute(const atb::VariantPack& variantPack,
                                                               uint8_t* workspace,
                                                               uint64_t workspace_size,
                                                               atb::Context* context) {
  if (variantPack.inTensors.size() != GetInputNum() || variantPack.outTensors.size() != OUTPUT_NUM) {
    return atb::ERROR_INVALID_TENSOR_NUM;
  }
  if (context == nullptr || prefill_subgraph_op_ == nullptr || decode_subgraph_op_ == nullptr) {
    return atb::ERROR_INVALID_TENSOR_NUM;
  }

  int32_t current_seq_len = 0;
  auto st = readCurrentSeqLen(variantPack.inTensors.at(CURRENT_SEQ_LEN_INPUT_INDEX), current_seq_len);
  if (st != atb::NO_ERROR) {
    return st;
  }

  const auto& query = variantPack.inTensors.at(QUERY_INPUT_INDEX);
  const auto& key_new = variantPack.inTensors.at(KEY_NEW_INPUT_INDEX);
  const auto& value_new = variantPack.inTensors.at(VALUE_NEW_INPUT_INDEX);
  const auto& k_cache = variantPack.inTensors.at(KEY_CACHE_INPUT_INDEX);
  const auto& v_cache = variantPack.inTensors.at(VALUE_CACHE_INPUT_INDEX);
  const auto& attn_scale = variantPack.inTensors.at(ATTN_SCALE_INPUT_INDEX);
  const auto& output = variantPack.outTensors.at(OUTPUT_INDEX);

  const int64_t batch = query.desc.shape.dims[0];
  const int64_t q_heads = query.desc.shape.dims[1];
  const int64_t seq_q = query.desc.shape.dims[2];
  const int64_t total_seq = current_seq_len + seq_q;
  const int64_t bucket_total_seq = bucketedTotalSeq(static_cast<int32_t>(total_seq));
  if (batch != 1 || q_heads != num_attention_heads_ || query.desc.shape.dims[3] != head_dim_
      || total_seq > max_cache_length_ || bucket_total_seq > max_cache_length_) {
    return atb::ERROR_INVALID_TENSOR_NUM;
  }

  st = appendToBaseCache(key_new, k_cache, current_seq_len, max_cache_length_);
  if (st != atb::NO_ERROR) {
    return st;
  }
  st = appendToBaseCache(value_new, v_cache, current_seq_len, max_cache_length_);
  if (st != atb::NO_ERROR) {
    return st;
  }

  const auto history_shape_3d = std::vector<int64_t>{batch * num_key_value_heads_, bucket_total_seq, head_dim_};
  WorkspacePlan plan;
  plan.key_history_bytes = alignUp(makeTensorWithShape(query.desc.dtype, history_shape_3d).dataSize);
  plan.value_history_bytes = alignUp(makeTensorWithShape(query.desc.dtype, history_shape_3d).dataSize);
  plan.subgraph_workspace_bytes = workspace_size >= (plan.key_history_bytes + plan.value_history_bytes)
      ? workspace_size - (plan.key_history_bytes + plan.value_history_bytes)
      : 0;
  if (workspace == nullptr && workspace_size > 0) {
    return atb::ERROR_INVALID_TENSOR_NUM;
  }

  uint8_t* key_history_ptr = workspace;
  uint8_t* value_history_ptr = key_history_ptr + plan.key_history_bytes;
  uint8_t* subgraph_workspace = value_history_ptr + plan.value_history_bytes;

  atb::Tensor key_hist_3d = makeTensorWithShape(query.desc.dtype, history_shape_3d, key_history_ptr);
  atb::Tensor value_hist_3d = makeTensorWithShape(query.desc.dtype, history_shape_3d, value_history_ptr);

  st = copyCachePrefixToContiguous(k_cache, total_seq, 1, key_hist_3d);
  if (st != atb::NO_ERROR) {
    return st;
  }
  st = copyCachePrefixToContiguous(v_cache, total_seq, 1, value_hist_3d);
  if (st != atb::NO_ERROR) {
    return st;
  }

  atb::VariantPack subgraph_pack;
  fillVariantPack(
      subgraph_pack,
      {makeTensorView(query, {batch * num_key_value_heads_, num_key_value_groups_ * seq_q, head_dim_}),
       key_hist_3d,
       value_hist_3d,
       attn_scale,
       variantPack.inTensors.at(CURRENT_SEQ_LEN_INPUT_INDEX)},
      {makeTensorView(output, {batch * num_key_value_heads_, num_key_value_groups_ * seq_q, head_dim_})});

  auto* selected_subgraph = seq_q > 1 ? prefill_subgraph_op_ : decode_subgraph_op_;
  return selected_subgraph->Execute(
      subgraph_pack,
      subgraph_workspace,
      plan.subgraph_workspace_bytes,
      context);
}

atb::Operation* createAttentionWithKVCachePluginGraphOp(int32_t num_attention_heads,
                                                        int32_t num_key_value_heads,
                                                        int32_t head_dim,
                                                        int32_t max_cache_length,
                                                        bool sliding_window,
                                                        int32_t window_size,
                                                        std::string name_suffix,
                                                        int32_t setup_bucket_size) {
  return new AscendAttentionWithKVCachePluginOperation(num_attention_heads,
                                                       num_key_value_heads,
                                                       head_dim,
                                                       max_cache_length,
                                                       sliding_window,
                                                       window_size,
                                                       std::move(name_suffix),
                                                       setup_bucket_size);
}

}  // namespace mllm::ascend
