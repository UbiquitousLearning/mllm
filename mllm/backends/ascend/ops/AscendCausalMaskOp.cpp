// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/ascend/ops/AscendCausalMaskOp.hpp"

#include <acl/acl.h>
#include <half/half.hpp>
#include <cstring>
#include <vector>
#include <algorithm>

#include "mllm/utils/Common.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/backends/ascend/AscendCommon.hpp"

namespace mllm::ascend {

AscendCausalMaskOp::AscendCausalMaskOp(const aops::CausalMaskOpOptions& options) : aops::CausalMaskOp(options) {}

void AscendCausalMaskOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  BaseOp::setup(inputs, outputs);
}

void AscendCausalMaskOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  MLLM_RT_ASSERT_EQ(inputs.size(), 1);
  MLLM_RT_ASSERT_EQ(outputs.size(), 1);

  const auto& x = inputs[0];
  auto& y = outputs[0];

  const auto& shape = x.shape();
  MLLM_RT_ASSERT_EQ(shape.size(), 4);

  const int64_t B = shape[0];  // batch size
  const int64_t H = shape[1];  // num heads
  const int64_t S = shape[2];  // query sequence length
  const int64_t D = shape[3];  // key sequence length

  const size_t numel = x.numel();
  const size_t bytes = x.bytes();
  const DataTypes dtype = x.dtype();

  if (dtype == kFloat16) {
    // Copy input from device to host
    std::vector<half_float::half> host_data(numel);
    auto ret = aclrtMemcpy(host_data.data(), bytes, x.ptr<void>(), bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    MLLM_ACL_CHECK(ret);

    // Apply causal mask on host
    const half_float::half mask_val(-65500.0f);

    if (S == 1) {
      // When sequence length is 1, no masking needed - just copy
      // Data is already in host_data, will be copied to output
    } else if (!options_.sliding_window) {
      // Standard causal mask
      for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < H; ++h) {
          half_float::half* row_ptr = host_data.data() + (b * H * S * D) + (h * S * D);
          for (int64_t s = 0; s < S; ++s) {
            // For each query position s, mask positions > s + (D - S) in key dimension
            // This handles the case where D > S (e.g., with KV cache)
            const int64_t row_offset = s * D;
            const int64_t copy_count = D - S + s + 1;
            const int64_t fill_start = copy_count;
            const int64_t fill_count = D - fill_start;

            // Values before fill_start are kept (lower triangular)
            // Values from fill_start to D are masked (upper triangular)
            for (int64_t d = fill_start; d < D; ++d) {
              row_ptr[row_offset + d] = mask_val;
            }
          }
        }
      }
    } else {
      // Sliding window causal mask
      const int window_size = options_.window_size;
      for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < H; ++h) {
          half_float::half* row_ptr = host_data.data() + (b * H * S * D) + (h * S * D);
          for (int64_t s = 0; s < S; ++s) {
            const int64_t row_offset = s * D;
            const int64_t copy_start_idx = std::max(static_cast<int64_t>(0), s - window_size + 1);

            // Mask prefix (before window)
            for (int64_t d = 0; d < copy_start_idx; ++d) {
              row_ptr[row_offset + d] = mask_val;
            }
            // Mask suffix (future tokens)
            for (int64_t d = s + 1; d < D; ++d) {
              row_ptr[row_offset + d] = mask_val;
            }
          }
        }
      }
    }

    // Copy result back to device
    ret = aclrtMemcpy(y.ptr<void>(), bytes, host_data.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    MLLM_ACL_CHECK(ret);

  } else if (dtype == kFloat32) {
    // Copy input from device to host
    std::vector<float> host_data(numel);
    auto ret = aclrtMemcpy(host_data.data(), bytes, x.ptr<void>(), bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    MLLM_ACL_CHECK(ret);

    // Apply causal mask on host
    const float mask_val = -1e10f;

    if (S == 1) {
      // When sequence length is 1, no masking needed
    } else if (!options_.sliding_window) {
      // Standard causal mask
      for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < H; ++h) {
          float* row_ptr = host_data.data() + (b * H * S * D) + (h * S * D);
          for (int64_t s = 0; s < S; ++s) {
            const int64_t row_offset = s * D;
            const int64_t copy_count = D - S + s + 1;
            const int64_t fill_start = copy_count;

            for (int64_t d = fill_start; d < D; ++d) {
              row_ptr[row_offset + d] = mask_val;
            }
          }
        }
      }
    } else {
      // Sliding window causal mask
      const int window_size = options_.window_size;
      for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < H; ++h) {
          float* row_ptr = host_data.data() + (b * H * S * D) + (h * S * D);
          for (int64_t s = 0; s < S; ++s) {
            const int64_t row_offset = s * D;
            const int64_t copy_start_idx = std::max(static_cast<int64_t>(0), s - window_size + 1);

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

    // Copy result back to device
    ret = aclrtMemcpy(y.ptr<void>(), bytes, host_data.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    MLLM_ACL_CHECK(ret);

  } else {
    MLLM_ERROR_EXIT(ExitCode::kAscendError, "AscendCausalMaskOp: Unsupported dtype {}",
                    static_cast<int>(dtype));
  }

  syncGlobalAtbStream();
}

}  // namespace mllm::ascend
