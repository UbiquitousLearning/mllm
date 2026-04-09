// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/ascend/graph/AscendCausalMaskTensorPluginOperation.hpp"

#include <acl/acl.h>
#include <half/half.hpp>

#include <algorithm>
#include <cstdint>

#include "mllm/backends/ascend/AscendCommon.hpp"

namespace mllm::ascend {

namespace {

template <typename T>
void fillMaskBuffer(T* ptr,
                    int64_t batch,
                    int64_t heads,
                    int64_t seq_q,
                    int64_t seq_kv,
                    bool sliding_window,
                    int32_t window_size,
                    T mask_val) {
  const size_t total_elems = static_cast<size_t>(batch * heads * seq_q * seq_kv);
  std::fill(ptr, ptr + total_elems, static_cast<T>(0));

  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t h = 0; h < heads; ++h) {
      for (int64_t s_q = 0; s_q < seq_q; ++s_q) {
        const int64_t base_idx = b * heads * seq_q * seq_kv + h * seq_q * seq_kv + s_q * seq_kv;
        const int64_t current_kv_pos = seq_kv - seq_q + s_q;

        if (!sliding_window) {
          for (int64_t s_kv = current_kv_pos + 1; s_kv < seq_kv; ++s_kv) {
            ptr[base_idx + s_kv] = mask_val;
          }
          continue;
        }

        const int64_t window_start = std::max<int64_t>(0, current_kv_pos - window_size + 1);
        for (int64_t s_kv = 0; s_kv < window_start; ++s_kv) {
          ptr[base_idx + s_kv] = mask_val;
        }
        for (int64_t s_kv = current_kv_pos + 1; s_kv < seq_kv; ++s_kv) {
          ptr[base_idx + s_kv] = mask_val;
        }
      }
    }
  }
}

}  // namespace

AscendCausalMaskTensorPluginOperation::AscendCausalMaskTensorPluginOperation(bool sliding_window, int32_t window_size)
    : sliding_window_(sliding_window), window_size_(window_size) {}

AscendCausalMaskTensorPluginOperation::~AscendCausalMaskTensorPluginOperation() {
  if (host_mask_buffer_ != nullptr) {
    auto ret = aclrtFreeHost(host_mask_buffer_);
    MLLM_ACL_CHECK(ret);
    host_mask_buffer_ = nullptr;
    host_mask_buffer_bytes_ = 0;
  }
}

std::string AscendCausalMaskTensorPluginOperation::GetName() const {
  return "AscendCausalMaskTensorPluginOperation";
}

atb::Status AscendCausalMaskTensorPluginOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs,
                                                              atb::SVector<atb::TensorDesc>& outTensorDescs) const {
  if (inTensorDescs.size() != 1 || outTensorDescs.size() != 1) {
    return atb::ERROR_INVALID_TENSOR_NUM;
  }
  outTensorDescs.at(0) = inTensorDescs.at(0);
  return atb::NO_ERROR;
}

uint32_t AscendCausalMaskTensorPluginOperation::GetInputNum() const {
  return 1;
}

uint32_t AscendCausalMaskTensorPluginOperation::GetOutputNum() const {
  return 1;
}

atb::Status AscendCausalMaskTensorPluginOperation::Setup(const atb::VariantPack& variantPack,
                                                         uint64_t& workspaceSize,
                                                         atb::Context* context) {
  (void)context;
  if (variantPack.inTensors.size() != 1 || variantPack.outTensors.size() != 1) {
    return atb::ERROR_INVALID_TENSOR_NUM;
  }
  workspaceSize = 0;
  return atb::NO_ERROR;
}

atb::Status AscendCausalMaskTensorPluginOperation::ensureHostMaskBuffer(uint64_t required_bytes) {
  if (required_bytes <= host_mask_buffer_bytes_) {
    return atb::NO_ERROR;
  }

  if (host_mask_buffer_ != nullptr) {
    auto ret = aclrtFreeHost(host_mask_buffer_);
    if (ret != ACL_SUCCESS) {
      return atb::ERROR_RT_FAIL;
    }
    host_mask_buffer_ = nullptr;
    host_mask_buffer_bytes_ = 0;
  }

  auto ret = aclrtMallocHost(&host_mask_buffer_, required_bytes);
  if (ret != ACL_SUCCESS) {
    return atb::ERROR_RT_FAIL;
  }
  host_mask_buffer_bytes_ = required_bytes;
  return atb::NO_ERROR;
}

atb::Status AscendCausalMaskTensorPluginOperation::Execute(const atb::VariantPack& variantPack,
                                                           uint8_t* workspace,
                                                           uint64_t workspaceSize,
                                                           atb::Context* context) {
  (void)workspace;
  (void)workspaceSize;
  (void)context;

  if (variantPack.inTensors.size() != 1 || variantPack.outTensors.size() != 1) {
    return atb::ERROR_INVALID_TENSOR_NUM;
  }

  const auto& input = variantPack.inTensors.at(0);
  auto output = variantPack.outTensors.at(0);
  if (input.desc.shape.dimNum != 4 || output.desc.shape.dimNum != 4) {
    return atb::ERROR_INVALID_TENSOR_DIM_NUM;
  }
  if (input.desc.dtype != output.desc.dtype) {
    return atb::ERROR_INVALID_TENSOR_DTYPE;
  }
  if (output.deviceData == nullptr) {
    return atb::ERROR_INVALID_TENSOR_ADDR;
  }

  const int64_t batch = input.desc.shape.dims[0];
  const int64_t heads = input.desc.shape.dims[1];
  const int64_t seq_q = input.desc.shape.dims[2];
  const int64_t seq_kv = input.desc.shape.dims[3];
  const size_t elem_size = aclDataTypeSize(input.desc.dtype);
  const uint64_t total_bytes = static_cast<uint64_t>(batch * heads * seq_q * seq_kv) * elem_size;

  auto st = ensureHostMaskBuffer(total_bytes);
  if (st != atb::NO_ERROR) {
    return st;
  }

  if (input.desc.dtype == ACL_FLOAT16) {
    fillMaskBuffer(reinterpret_cast<half_float::half*>(host_mask_buffer_),
                   batch,
                   heads,
                   seq_q,
                   seq_kv,
                   sliding_window_,
                   window_size_,
                   half_float::half(-65500.0f));
  } else if (input.desc.dtype == ACL_FLOAT) {
    fillMaskBuffer(reinterpret_cast<float*>(host_mask_buffer_),
                   batch,
                   heads,
                   seq_q,
                   seq_kv,
                   sliding_window_,
                   window_size_,
                   -1e10f);
  } else {
    return atb::ERROR_INVALID_TENSOR_DTYPE;
  }

  if (total_bytes == 0) {
    return atb::NO_ERROR;
  }

  auto ret = aclrtMemcpyAsync(output.deviceData,
                              total_bytes,
                              host_mask_buffer_,
                              total_bytes,
                              ACL_MEMCPY_HOST_TO_DEVICE,
                              getGlobalAtbStream());
  return ret == ACL_SUCCESS ? atb::NO_ERROR : atb::ERROR_RT_FAIL;
}

atb::Operation* createCausalMaskTensorPluginGraphOp(bool sliding_window, int32_t window_size) {
  return new AscendCausalMaskTensorPluginOperation(sliding_window, window_size);
}

}  // namespace mllm::ascend
