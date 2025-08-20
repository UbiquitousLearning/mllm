// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/utils/CPUArchHelper.hpp"
#include "mllm/backends/cpu/kernels/common/fa2/fwd_bshd.hpp"  // IWYU pragma: export

#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
#include "mllm/backends/cpu/kernels/x86/fill.hpp"  // IWYU pragma: export
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
#include "mllm/backends/cpu/kernels/arm/mllm_blas/mllm_blas_sgemm.hpp"  // IWYU pragma: export
#endif

#include "mllm/backends/cpu/kernels/common/blas.hpp"  // IWYU pragma: export
