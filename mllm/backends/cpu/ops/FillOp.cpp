// Copyright (c) MLLM Team.
// Licensed under the MIT License.

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
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::fill_zeros(dst.ptr<mllm_fp32_t>(), dst.numel(), threads);
#endif
          break;
        }
        case kFloat16: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
          // TODO
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::fill_zeros_fp16(dst.ptr<mllm_fp16_t>(), dst.numel(), threads);
#endif
          break;
        }
        case kInt64: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
          // TODO
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::fill_zeros_anytype<mllm_int64_t>(dst.ptr<mllm_int64_t>(), dst.numel(), threads);
#endif
          break;
        }
        case kInt32: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
          // TODO
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::fill_zeros_anytype<mllm_int32_t>(dst.ptr<mllm_int32_t>(), dst.numel(), threads);
#endif
          break;
        }
        case kInt16: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
          // TODO
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::fill_zeros_anytype<mllm_int16_t>(dst.ptr<mllm_int16_t>(), dst.numel(), threads);
#endif
          break;
        }
        case kInt8: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
          // TODO
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::fill_zeros_anytype<mllm_int8_t>(dst.ptr<mllm_int8_t>(), dst.numel(), threads);
#endif
          break;
        }
        case kUInt64: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
          // TODO
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::fill_zeros_anytype<mllm_uint64_t>(dst.ptr<mllm_uint64_t>(), dst.numel(), threads);
#endif
          break;
        }
        case kUInt32: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
          // TODO
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::fill_zeros_anytype<mllm_uint32_t>(dst.ptr<mllm_uint32_t>(), dst.numel(), threads);
#endif
          break;
        }
        case kUInt16: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
          // TODO
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::fill_zeros_anytype<mllm_uint16_t>(dst.ptr<mllm_uint16_t>(), dst.numel(), threads);
#endif
          break;
        }
        case kUInt8: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
          // TODO
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::fill_zeros_anytype<mllm_uint8_t>(dst.ptr<mllm_uint8_t>(), dst.numel(), threads);
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
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::fill_ones(dst.ptr<mllm_fp32_t>(), dst.numel(), threads);
#endif
          break;
        }
        case kFloat16: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
          // TODO
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::fill_ones_fp16(dst.ptr<mllm_fp16_t>(), dst.numel(), threads);
#endif
          break;
        }
        case kInt64: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
          // TODO
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::fill_ones_anytype<mllm_int64_t>(dst.ptr<mllm_int64_t>(), dst.numel(), threads);
#endif
          break;
        }
        case kInt32: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
          // TODO
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::fill_ones_anytype<mllm_int32_t>(dst.ptr<mllm_int32_t>(), dst.numel(), threads);
#endif
          break;
        }
        case kInt16: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
          // TODO
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::fill_ones_anytype<mllm_int16_t>(dst.ptr<mllm_int16_t>(), dst.numel(), threads);
#endif
          break;
        }
        case kInt8: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
          // TODO
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::fill_ones_anytype<mllm_int8_t>(dst.ptr<mllm_int8_t>(), dst.numel(), threads);
#endif
          break;
        }
        case kUInt64: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
          // TODO
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::fill_ones_anytype<mllm_uint64_t>(dst.ptr<mllm_uint64_t>(), dst.numel(), threads);
#endif
          break;
        }
        case kUInt32: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
          // TODO
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::fill_ones_anytype<mllm_uint32_t>(dst.ptr<mllm_uint32_t>(), dst.numel(), threads);
#endif
          break;
        }
        case kUInt16: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
          // TODO
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::fill_ones_anytype<mllm_uint16_t>(dst.ptr<mllm_uint16_t>(), dst.numel(), threads);
#endif
          break;
        }
        case kUInt8: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
          // TODO
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::fill_ones_anytype<mllm_uint8_t>(dst.ptr<mllm_uint8_t>(), dst.numel(), threads);
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
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::fill_arange(dst.ptr<mllm_fp32_t>(), dst.numel(), options_.start, options_.end, options_.step, threads);
#endif
          break;
        }
        case kFloat16: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
          // TODO
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::fill_arange_fp16(dst.ptr<mllm_fp16_t>(), dst.numel(), options_.start, options_.end, options_.step, threads);
#endif
          break;
        }
        case kInt64: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
          // TODO
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::fill_arange_anytype<mllm_int64_t>(dst.ptr<mllm_int64_t>(), dst.numel(), options_.start, options_.end,
                                                 options_.step, threads);
#endif
          break;
        }
        case kInt32: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
          // TODO
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::fill_arange_anytype<mllm_int32_t>(dst.ptr<mllm_int32_t>(), dst.numel(), options_.start, options_.end,
                                                 options_.step, threads);
#endif
          break;
        }
        case kInt16: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
          // TODO
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::fill_arange_anytype<mllm_int16_t>(dst.ptr<mllm_int16_t>(), dst.numel(), options_.start, options_.end,
                                                 options_.step, threads);
#endif
          break;
        }
        case kInt8: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
          // TODO
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::fill_arange_anytype<mllm_int8_t>(dst.ptr<mllm_int8_t>(), dst.numel(), options_.start, options_.end,
                                                options_.step, threads);
#endif
          break;
        }
        case kUInt64: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
          // TODO
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::fill_arange_anytype<mllm_uint64_t>(dst.ptr<mllm_uint64_t>(), dst.numel(), options_.start, options_.end,
                                                  options_.step, threads);
#endif
          break;
        }
        case kUInt32: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
          // TODO
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::fill_arange_anytype<mllm_uint32_t>(dst.ptr<mllm_uint32_t>(), dst.numel(), options_.start, options_.end,
                                                  options_.step, threads);
#endif
          break;
        }
        case kUInt16: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
          // TODO
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::fill_arange_anytype<mllm_uint16_t>(dst.ptr<mllm_uint16_t>(), dst.numel(), options_.start, options_.end,
                                                  options_.step, threads);
#endif
          break;
        }
        case kUInt8: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
          // TODO
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::fill_arange_anytype<mllm_uint8_t>(dst.ptr<mllm_uint8_t>(), dst.numel(), options_.start, options_.end,
                                                 options_.step, threads);
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
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::fill_random(dst.ptr<mllm_fp32_t>(), dst.numel(), options_.start, options_.end, options_.seed, threads);
#endif
          break;
        }
        case kFloat16: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
          // TODO
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::fill_random_fp16(dst.ptr<mllm_fp16_t>(), dst.numel(), options_.start, options_.end, options_.seed, threads);
#endif
          break;
        }
        case kInt64: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
          // TODO
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::fill_random_anytype(dst.ptr<mllm_int64_t>(), dst.numel(), options_.start, options_.end, options_.seed, threads);
#endif
          break;
        }
        case kInt32: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
          // TODO
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::fill_random_anytype(dst.ptr<mllm_int32_t>(), dst.numel(), options_.start, options_.end, options_.seed, threads);
#endif
          break;
        }
        case kInt16: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
          // TODO
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::fill_random_anytype(dst.ptr<mllm_int16_t>(), dst.numel(), options_.start, options_.end, options_.seed, threads);
#endif
          break;
        }
        case kInt8: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
          // TODO
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::fill_random_anytype(dst.ptr<mllm_int8_t>(), dst.numel(), options_.start, options_.end, options_.seed, threads);
#endif
          break;
        }
        case kUInt64: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
          // TODO
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::fill_random_anytype(dst.ptr<mllm_uint64_t>(), dst.numel(), options_.start, options_.end, options_.seed, threads);
#endif
          break;
        }
        case kUInt32: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
          // TODO
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::fill_random_anytype(dst.ptr<mllm_uint32_t>(), dst.numel(), options_.start, options_.end, options_.seed, threads);
#endif
          break;
        }
        case kUInt16: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
          // TODO
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::fill_random_anytype(dst.ptr<mllm_uint16_t>(), dst.numel(), options_.start, options_.end, options_.seed, threads);
#endif
          break;
        }
        case kUInt8: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
          // TODO
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::fill_random_anytype(dst.ptr<mllm_uint8_t>(), dst.numel(), options_.start, options_.end, options_.seed, threads);
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
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::fill_specific_value(dst.ptr<mllm_fp32_t>(), dst.numel(), options_.value, threads);
#endif
          break;
        }
        case kFloat16: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
          // TODO
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::fill_specific_value_fp16(dst.ptr<mllm_fp16_t>(), dst.numel(), options_.value, threads);
#endif
          break;
        }
        case kInt64: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
          // TODO
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::fill_specific_value_anytype(dst.ptr<mllm_int64_t>(), dst.numel(), options_.value, threads);
#endif
          break;
        }
        case kInt32: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
          // TODO
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::fill_specific_value_anytype(dst.ptr<mllm_int32_t>(), dst.numel(), options_.value, threads);
#endif
          break;
        }
        case kInt16: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
          // TODO
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::fill_specific_value_anytype(dst.ptr<mllm_int16_t>(), dst.numel(), options_.value, threads);
#endif
          break;
        }
        case kInt8: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
          // TODO
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::fill_specific_value_anytype(dst.ptr<mllm_int8_t>(), dst.numel(), options_.value, threads);
#endif
          break;
        }
        case kUInt64: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
          // TODO
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::fill_specific_value_anytype(dst.ptr<mllm_uint64_t>(), dst.numel(), options_.value, threads);
#endif
          break;
        }
        case kUInt32: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
          // TODO
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::fill_specific_value_anytype(dst.ptr<mllm_uint32_t>(), dst.numel(), options_.value, threads);
#endif
          break;
        }
        case kUInt16: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
          // TODO
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::fill_specific_value_anytype(dst.ptr<mllm_uint16_t>(), dst.numel(), options_.value, threads);
#endif
          break;
        }
        case kUInt8: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
          // TODO
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::fill_specific_value_anytype(dst.ptr<mllm_uint8_t>(), dst.numel(), options_.value, threads);
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
