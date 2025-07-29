/**
 * @file FillOp.cpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-27
 *
 */
#include "mllm/backends/cpu/ops/FillOp.hpp"
#include "mllm/backends/cpu/kernels/Kernels.hpp"
#include "mllm/utils/PlatformRTHelper.hpp"

namespace mllm::cpu {

CPUFillOp::CPUFillOp(const aops::FillOpOptions& options) : aops::FillOp(options) {}

void CPUFillOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& dst = outputs[0];

  // For android
  auto threads = options_.getThreads();
  if constexpr (isAndroid() && (isARM() || isARM64())) { threads = 0; }

  switch (options_.type) {
    case aops::FillOpTypes::kZeros: {
      switch (dst.dtype()) {
        case kFloat32: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
          x86::fill_zeros(dst.ptr<mllm_fp32_t>(), dst.numel(), threads);
#endif
          break;
        }
        default: {
          NYI("FillOp::forward[zeros] not implemented for this data type");
          break;
        }
      }
      break;
    }
    case aops::FillOpTypes::kOnes: {
      switch (dst.dtype()) {
        case kFloat32: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
          x86::fill_ones(dst.ptr<mllm_fp32_t>(), dst.numel(), threads);
#endif
          break;
        }
        default: {
          NYI("FillOp::forward[ones] not implemented for this data type");
          break;
        }
      }
      break;
    }
    case aops::FillOpTypes::kArange: {
      switch (dst.dtype()) {
        case kFloat32: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
          x86::fill_arange(dst.ptr<mllm_fp32_t>(), dst.numel(), options_.start, options_.end, options_.step, threads);
#endif
          break;
        }
        default: {
          NYI("FillOp::forward[arange] not implemented for this data type");
        }
      }
      break;
    }
    case aops::FillOpTypes::kRandom: {
      switch (dst.dtype()) {
        case kFloat32: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
          x86::fill_random(dst.ptr<mllm_fp32_t>(), dst.numel(), options_.start, options_.end, options_.seed, threads);
#endif
          break;
        }
        default: {
          NYI("FillOp::forward[random] not implemented for this data type")
        }
      }
      break;
    }
    case aops::FillOpTypes::kSpecific: {
      switch (dst.dtype()) {
        case kFloat32: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
          x86::fill_specific_value(dst.ptr<mllm_fp32_t>(), dst.numel(), options_.value, threads);
#endif
          break;
        }
        default: {
          NYI("FillOp::forward[specific] not implemented for this data type");
          break;
        }
      }
      break;
    }
    default: {
      NYI("FillOp::forward[type] not implemented for this type");
      break;
    }
  }
}

}  // namespace mllm::cpu
