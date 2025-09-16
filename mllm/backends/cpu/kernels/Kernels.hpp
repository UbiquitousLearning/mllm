// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <variant>

#include "mllm/utils/CPUArchHelper.hpp"

#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
#include "mllm/backends/cpu/kernels/x86/fill.hpp"                // IWYU pragma: export
#include "/root/mllm/mllm/backends/cpu/kernels/x86/silu.hpp"     // IWYU pragma: export
#include "/root/mllm/mllm/backends/cpu/kernels/x86/softmax.hpp"  // IWYU pragma: export
#include "/root/mllm/mllm/backends/cpu/kernels/x86/rmsnorm.hpp"  // IWYU pragma: export
#include "/root/mllm/mllm/backends/cpu/kernels/x86/gelu.hpp"     // IWYU pragma: export
#endif

#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
#include "mllm/backends/cpu/kernels/arm/fill.hpp"                       // IWYU pragma: export
#include "mllm/backends/cpu/kernels/arm/elementwise.hpp"                // IWYU pragma: export
#include "mllm/backends/cpu/kernels/arm/reduce.hpp"                     // IWYU pragma: export
#include "mllm/backends/cpu/kernels/arm/transpose.hpp"                  // IWYU pragma: export
#include "mllm/backends/cpu/kernels/arm/permute.hpp"                    // IWYU pragma: export
#include "mllm/backends/cpu/kernels/arm/silu.hpp"                       // IWYU pragma: export
#include "mllm/backends/cpu/kernels/arm/cast_types.hpp"                 // IWYU pragma: export
#include "mllm/backends/cpu/kernels/arm/layernorm.hpp"                  // IWYU pragma: export
#include "mllm/backends/cpu/kernels/arm/softmax.hpp"                    // IWYU pragma: export
#include "mllm/backends/cpu/kernels/arm/rmsnorm.hpp"                    // IWYU pragma: export
#include "mllm/backends/cpu/kernels/arm/gelu.hpp"                       // IWYU pragma: export
#include "mllm/backends/cpu/kernels/arm/conv3d.hpp"                     // IWYU pragma: export
#include "mllm/backends/cpu/kernels/arm/linear/kai.hpp"                 // IWYU pragma: export
#include "mllm/backends/cpu/kernels/arm/relu.hpp"                       // IWYU pragma: export
#include "mllm/backends/cpu/kernels/arm/mllm_blas/mllm_blas_sgemm.hpp"  // IWYU pragma: export
#else
#include "mllm/backends/cpu/kernels/common/gelu-inl.hpp"     // IWYU pragma: export
#include "mllm/backends/cpu/kernels/common/permute-inl.hpp"  // IWYU pragma: export
#endif

// Platform free Kernels.

// NOTE: common/blas.hpp should be include after all kernels. That because in apple platform.
// Tensor::nil()'s nil keyword has been defined in apple's system head.
#include "mllm/backends/cpu/kernels/common/ggml/matmul.hpp"          // IWYU pragma: export
#include "mllm/backends/cpu/kernels/common/fa2/fwd_bshd.hpp"         // IWYU pragma: export
#include "mllm/backends/cpu/kernels/common/paged_attn/fwd_bshd.hpp"  // IWYU pragma: export
#include "mllm/backends/cpu/kernels/common/blas.hpp"                 // IWYU pragma: export

// TODO
// Provide all quantize methods in one function or class here.
// 1. GGUF Quant
// 2. U1-U7 Bitspack Quant
// 3. KAI Quant
// 4. Other quantization method.
namespace mllm::cpu::quantize {

//===----------------------------------------------------------------------===//
// Quantize methods.
//===----------------------------------------------------------------------===//
enum class QuantizeMethod : int32_t {
  kGGUF,
  kBitsPack,
  kKAI,
  kMllmNormal,
};

struct QuantizeKernelGGUFConfig {};

struct QuantizeKernelBitsPackConfig {};

struct QuantizeKernelKAIConfig {};

struct QuantizeKernelConfig {
  QuantizeMethod type;
  std::variant<QuantizeKernelGGUFConfig, QuantizeKernelBitsPackConfig, QuantizeKernelKAIConfig> config;
};

namespace details {

template<QuantizeMethod __QUANTIZE_METHOD>
struct QuantizeKernelImpl {
  static inline void process(const QuantizeKernelConfig& cfg, void* inputs_ptr, void* outputs_ptr, void* zp_ptr,
                             void* scale_ptr, void* extra_workspace) {};
};

///< GGUF
template<>
struct QuantizeKernelImpl<QuantizeMethod::kGGUF> {
  static inline void process(const QuantizeKernelConfig& cfg, void* inputs_ptr, void* outputs_ptr, void* zp_ptr,
                             void* scale_ptr, void* extra_workspace) {
    // TODO
  };
};

///< BitsPack
template<>
struct QuantizeKernelImpl<QuantizeMethod::kBitsPack> {
  static inline void process(const QuantizeKernelConfig& cfg, void* inputs_ptr, void* outputs_ptr, void* zp_ptr,
                             void* scale_ptr, void* extra_workspace) {
    // TODO
  };
};

///< KAI
template<>
struct QuantizeKernelImpl<QuantizeMethod::kKAI> {
  static inline void process(const QuantizeKernelConfig& cfg, void* inputs_ptr, void* outputs_ptr, void* zp_ptr,
                             void* scale_ptr, void* extra_workspace) {
    // TODO
  };
};

///< MllmNormal
template<>
struct QuantizeKernelImpl<QuantizeMethod::kMllmNormal> {
  static inline void process(const QuantizeKernelConfig& cfg, void* inputs_ptr, void* outputs_ptr, void* zp_ptr,
                             void* scale_ptr, void* extra_workspace) {
    // TODO
  };
};
}  // namespace details

inline void anyQuantize(const QuantizeKernelConfig& cfg, void* inputs_ptr, void* outputs_ptr, void* zp_ptr, void* scale_ptr,
                        void* extra_workspace) {
#define CASE(__type_name__)                                                                                              \
  case QuantizeMethod::__type_name__: {                                                                                  \
    details::QuantizeKernelImpl<QuantizeMethod::__type_name__>::process(cfg, inputs_ptr, outputs_ptr, zp_ptr, scale_ptr, \
                                                                        extra_workspace);                                \
    break;                                                                                                               \
  }

  switch (cfg.type) {
    CASE(kGGUF)
    CASE(kBitsPack)
    CASE(kKAI)
    CASE(kMllmNormal)
  }

#undef CASE
}

//===----------------------------------------------------------------------===//
// DeQuantize methods.
//===----------------------------------------------------------------------===//
enum class DeQuantizeMethod : int32_t {
  kGGUF,
  kBitsPack,
  kKAI,
  kMllmNormal,
};

struct DeQuantizeKernelGGUFConfig {};

struct DeQuantizeKernelBitsPackConfig {};

struct DeQuantizeKernelKAIConfig {};

struct DeQuantizeKernelConfig {
  DeQuantizeMethod type;
  std::variant<DeQuantizeKernelGGUFConfig, DeQuantizeKernelBitsPackConfig, DeQuantizeKernelKAIConfig> config;
};

namespace details {

template<DeQuantizeMethod T>
struct DeQuantizeKernelImpl {
  static inline void process(const DeQuantizeKernelConfig& cfg, void* inputs_ptr, void* zp_ptr, void* scale_ptr,
                             void* outputs_ptr, void* extra_workspace) {};
};

///< GGUF
template<>
struct DeQuantizeKernelImpl<DeQuantizeMethod::kGGUF> {
  static inline void process(const DeQuantizeKernelConfig& cfg, void* inputs_ptr, void* zp_ptr, void* scale_ptr,
                             void* outputs_ptr, void* extra_workspace) {
    // TODO
  };
};

///< BitsPack
template<>
struct DeQuantizeKernelImpl<DeQuantizeMethod::kBitsPack> {
  static inline void process(const DeQuantizeKernelConfig& cfg, void* inputs_ptr, void* zp_ptr, void* scale_ptr,
                             void* outputs_ptr, void* extra_workspace) {
    // TODO
  };
};

///< KAI
template<>
struct DeQuantizeKernelImpl<DeQuantizeMethod::kKAI> {
  static inline void process(const DeQuantizeKernelConfig& cfg, void* inputs_ptr, void* zp_ptr, void* scale_ptr,
                             void* outputs_ptr, void* extra_workspace) {
    // TODO
  };
};

///< MllmNormal
template<>
struct DeQuantizeKernelImpl<DeQuantizeMethod::kMllmNormal> {
  static inline void process(const DeQuantizeKernelConfig& cfg, void* inputs_ptr, void* zp_ptr, void* scale_ptr,
                             void* outputs_ptr, void* extra_workspace) {
    // TODO
  };
};
}  // namespace details

inline void anyDeQuantize(const DeQuantizeKernelConfig& cfg, void* inputs_ptr, void* zp_ptr, void* scale_ptr, void* outputs_ptr,
                          void* extra_workspace) {
#define CASE(__type_name__)                                                                                                  \
  case DeQuantizeMethod::__type_name__: {                                                                                    \
    details::DeQuantizeKernelImpl<DeQuantizeMethod::__type_name__>::process(cfg, inputs_ptr, zp_ptr, scale_ptr, outputs_ptr, \
                                                                            extra_workspace);                                \
    break;                                                                                                                   \
  }

  switch (cfg.type) {
    CASE(kGGUF)
    CASE(kBitsPack)
    CASE(kKAI)
    CASE(kMllmNormal)
  }

#undef CASE
}

}  // namespace mllm::cpu::quantize
