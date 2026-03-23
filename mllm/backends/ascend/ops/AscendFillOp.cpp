// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/ascend/ops/AscendFillOp.hpp"

#include <acl/acl.h>
#include <half/half.hpp>
#include <cstring>
#include <random>

#include "mllm/utils/Common.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/backends/ascend/AscendCommon.hpp"

namespace mllm::ascend {

AscendFillOp::AscendFillOp(const aops::FillOpOptions& options) : aops::FillOp(options) {}

void AscendFillOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  BaseOp::setup(inputs, outputs);
}

void AscendFillOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  MLLM_RT_ASSERT_EQ(inputs.size(), 1);
  MLLM_RT_ASSERT_EQ(outputs.size(), 1);

  auto& output = outputs[0];
  const size_t numel = output.numel();
  const DataTypes dtype = output.dtype();

  // Handle FP16 case (most common for Ascend)
  if (dtype == kFloat16) {
    std::vector<half_float::half> fp16_data(numel);

    switch (options_.type) {
      case aops::FillOpTypes::kZeros: {
        half_float::half zero_val(0.0f);
        std::fill(fp16_data.begin(), fp16_data.end(), zero_val);
        break;
      }
      case aops::FillOpTypes::kOnes: {
        half_float::half one_val(1.0f);
        std::fill(fp16_data.begin(), fp16_data.end(), one_val);
        break;
      }
      case aops::FillOpTypes::kSpecific: {
        half_float::half specific_val(static_cast<float>(options_.value));
        std::fill(fp16_data.begin(), fp16_data.end(), specific_val);
        break;
      }
      case aops::FillOpTypes::kArange: {
        float start = static_cast<float>(options_.start);
        float step = static_cast<float>(options_.step);
        for (size_t i = 0; i < numel; ++i) {
          fp16_data[i] = half_float::half(start + static_cast<float>(i) * step);
        }
        break;
      }
      case aops::FillOpTypes::kRandom: {
        std::mt19937 gen(static_cast<unsigned int>(options_.seed));
        float range_start = static_cast<float>(options_.start);
        float range_end = static_cast<float>(options_.end);
        std::uniform_real_distribution<float> dist(range_start, range_end);
        for (size_t i = 0; i < numel; ++i) {
          fp16_data[i] = half_float::half(dist(gen));
        }
        break;
      }
      default:
        MLLM_ERROR_EXIT(ExitCode::kAscendError, "AscendFillOp: Unsupported fill type {}",
                        static_cast<int>(options_.type));
    }

    // Copy to Ascend device
    void* dst_data = output.ptr<void>();
    const size_t bytes = output.bytes();
    auto ret = aclrtMemcpy(dst_data, bytes, fp16_data.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
      MLLM_ACL_CHECK(ret);
    }
  }
  // Handle Int64 case (for indices)
  else if (dtype == kInt64) {
    std::vector<int64_t> int64_data(numel);

    switch (options_.type) {
      case aops::FillOpTypes::kZeros:
        std::fill(int64_data.begin(), int64_data.end(), 0);
        break;
      case aops::FillOpTypes::kOnes:
        std::fill(int64_data.begin(), int64_data.end(), 1);
        break;
      case aops::FillOpTypes::kSpecific:
        std::fill(int64_data.begin(), int64_data.end(), static_cast<int64_t>(options_.value));
        break;
      case aops::FillOpTypes::kArange: {
        int64_t start = static_cast<int64_t>(options_.start);
        int64_t step = static_cast<int64_t>(options_.step);
        for (size_t i = 0; i < numel; ++i) {
          int64_data[i] = start + static_cast<int64_t>(i) * step;
        }
        break;
      }
      default:
        MLLM_ERROR_EXIT(ExitCode::kAscendError, "AscendFillOp: Unsupported fill type {} for Int64",
                        static_cast<int>(options_.type));
    }

    void* dst_data = output.ptr<void>();
    const size_t bytes = output.bytes();
    auto ret = aclrtMemcpy(dst_data, bytes, int64_data.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
      MLLM_ACL_CHECK(ret);
    }
  }
  // Handle Float32 case
  else if (dtype == kFloat32) {
    std::vector<float> fp32_data(numel);

    switch (options_.type) {
      case aops::FillOpTypes::kZeros:
        std::fill(fp32_data.begin(), fp32_data.end(), 0.0f);
        break;
      case aops::FillOpTypes::kOnes:
        std::fill(fp32_data.begin(), fp32_data.end(), 1.0f);
        break;
      case aops::FillOpTypes::kSpecific:
        std::fill(fp32_data.begin(), fp32_data.end(), static_cast<float>(options_.value));
        break;
      case aops::FillOpTypes::kArange: {
        float start = static_cast<float>(options_.start);
        float step = static_cast<float>(options_.step);
        for (size_t i = 0; i < numel; ++i) {
          fp32_data[i] = start + static_cast<float>(i) * step;
        }
        break;
      }
      case aops::FillOpTypes::kRandom: {
        std::mt19937 gen(static_cast<unsigned int>(options_.seed));
        float range_start = static_cast<float>(options_.start);
        float range_end = static_cast<float>(options_.end);
        std::uniform_real_distribution<float> dist(range_start, range_end);
        for (size_t i = 0; i < numel; ++i) {
          fp32_data[i] = dist(gen);
        }
        break;
      }
      default:
        MLLM_ERROR_EXIT(ExitCode::kAscendError, "AscendFillOp: Unsupported fill type {} for Float32",
                        static_cast<int>(options_.type));
    }

    void* dst_data = output.ptr<void>();
    const size_t bytes = output.bytes();
    auto ret = aclrtMemcpy(dst_data, bytes, fp32_data.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
      MLLM_ACL_CHECK(ret);
    }
  }
  else {
    MLLM_ERROR_EXIT(ExitCode::kAscendError, "AscendFillOp: Unsupported dtype {}",
                    static_cast<int>(dtype));
  }

  syncGlobalAtbStream();
}

}  // namespace mllm::ascend
