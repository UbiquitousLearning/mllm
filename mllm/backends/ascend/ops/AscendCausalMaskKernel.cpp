// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/ascend/ops/AscendCausalMaskKernel.hpp"

#include <acl/acl.h>
#include <half/half.hpp>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "mllm/backends/ascend/AscendCommon.hpp"

namespace mllm::ascend {

namespace {

template <typename T>
void applyStandardCausalMask(std::vector<T>& host_data,
                             int64_t B,
                             int64_t H,
                             int64_t S,
                             int64_t D,
                             T mask_val) {
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t h = 0; h < H; ++h) {
      T* row_ptr = host_data.data() + (b * H * S * D) + (h * S * D);
      for (int64_t s = 0; s < S; ++s) {
        const int64_t row_offset = s * D;
        const int64_t fill_start = D - S + s + 1;
        for (int64_t d = fill_start; d < D; ++d) {
          row_ptr[row_offset + d] = mask_val;
        }
      }
    }
  }
}

template <typename T>
void applySlidingWindowCausalMask(std::vector<T>& host_data,
                                  int64_t B,
                                  int64_t H,
                                  int64_t S,
                                  int64_t D,
                                  int32_t window_size,
                                  T mask_val) {
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t h = 0; h < H; ++h) {
      T* row_ptr = host_data.data() + (b * H * S * D) + (h * S * D);
      for (int64_t s = 0; s < S; ++s) {
        const int64_t row_offset = s * D;
        const int64_t copy_start_idx = std::max<int64_t>(0, s - window_size + 1);
        for (int64_t d = 0; d < copy_start_idx; ++d) {
          row_ptr[row_offset + d] = mask_val;
        }
        for (int64_t d = s + 1; d < D; ++d) {
          row_ptr[row_offset + d] = mask_val;
        }
      }
    }
  }
}

template <typename T>
atb::Status executeTypedCausalMask(const atb::Tensor& input,
                                   atb::Tensor& output,
                                   bool sliding_window,
                                   int32_t window_size,
                                   T mask_val) {
  const auto& shape = input.desc.shape;
  const int64_t B = shape.dims[0];
  const int64_t H = shape.dims[1];
  const int64_t S = shape.dims[2];
  const int64_t D = shape.dims[3];

  const size_t numel = static_cast<size_t>(B * H * S * D);
  std::vector<T> host_data(numel);

  auto ret = aclrtMemcpy(host_data.data(),
                         input.dataSize,
                         input.deviceData,
                         input.dataSize,
                         ACL_MEMCPY_DEVICE_TO_HOST);
  if (ret != ACL_SUCCESS) {
    return atb::ERROR_RT_FAIL;
  }

  if (!sliding_window) {
    applyStandardCausalMask(host_data, B, H, S, D, mask_val);
  } else {
    applySlidingWindowCausalMask(host_data, B, H, S, D, window_size, mask_val);
  }

  ret = aclrtMemcpy(output.deviceData,
                    output.dataSize,
                    host_data.data(),
                    input.dataSize,
                    ACL_MEMCPY_HOST_TO_DEVICE);
  if (ret != ACL_SUCCESS) {
    return atb::ERROR_RT_FAIL;
  }
  return atb::NO_ERROR;
}

}  // namespace

atb::Status executeAscendCausalMaskKernel(const atb::Tensor& input,
                                          atb::Tensor& output,
                                          bool sliding_window,
                                          int32_t window_size) {
  if (input.deviceData == nullptr || output.deviceData == nullptr) {
    return atb::ERROR_INVALID_TENSOR_ADDR;
  }
  if (input.desc.shape.dimNum != 4 || output.desc.shape.dimNum != 4) {
    return atb::ERROR_INVALID_TENSOR_DIM_NUM;
  }
  if (input.desc.dtype != output.desc.dtype) {
    return atb::ERROR_INVALID_TENSOR_DTYPE;
  }
  if (input.dataSize != output.dataSize) {
    return atb::ERROR_INVALID_TENSOR_SIZE;
  }

  const int64_t S = input.desc.shape.dims[2];
  const int64_t D = input.desc.shape.dims[3];
  if (S <= 0 || D <= 0 || D < S) {
    return atb::ERROR_INVALID_TENSOR_DIM;
  }
  if (sliding_window && window_size <= 0) {
    return atb::ERROR_INVALID_PARAM;
  }
  if (S == 1) {
    auto ret = aclrtMemcpy(output.deviceData,
                           output.dataSize,
                           input.deviceData,
                           input.dataSize,
                           ACL_MEMCPY_DEVICE_TO_DEVICE);
    return ret == ACL_SUCCESS ? atb::NO_ERROR : atb::ERROR_RT_FAIL;
  }

  switch (input.desc.dtype) {
    case ACL_FLOAT16:
      return executeTypedCausalMask(input, output, sliding_window, window_size, half_float::half(-65500.0f));
    case ACL_FLOAT:
      return executeTypedCausalMask(input, output, sliding_window, window_size, -1e10f);
    default:
      return atb::ERROR_INVALID_TENSOR_DTYPE;
  }
}

}  // namespace mllm::ascend
