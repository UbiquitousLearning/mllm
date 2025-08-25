// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/utils/Log.hpp"
#include "mllm/utils/Dbg.hpp"  // IWYU pragma: export

#if defined(_MSC_VER)
#define MLLM_FORCE_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define MLLM_FORCE_INLINE __attribute__((always_inline)) inline
#else
#define MLLM_FORCE_INLINE inline
#endif

#define MLLM_ANONYMOUS_NAMESPACE

#define MLLM_EMPTY_SCOPE

#define MLLM_ENABLE_RT_ASSERT 1

namespace mllm {

enum class ExitCode : int32_t {  // NOLINT
  kSuccess = 0,
  kCoreError,
  kAssert,
  kSliceOB,  // slice out of bound
  kMemory,
  kCudaError,
  kQnnError,
  kOpenCLError,
  kIOError,
  kShapeError,
};

// mllm runtime assert
#if (MLLM_ENABLE_RT_ASSERT)
#define MLLM_RT_ASSERT(statement) \
  if (!(statement)) { MLLM_ASSERT_EXIT(::mllm::ExitCode::kAssert, "{}", #statement); }

#define MLLM_RT_ASSERT_EQ(statement1, statement2) \
  if ((statement1) != (statement2)) { MLLM_ASSERT_EXIT(::mllm::ExitCode::kAssert, "{} != {}", #statement1, #statement2); }
#else
#define MLLM_RT_ASSERT(statement)

#define MLLM_RT_ASSERT_EQ(statement1, statement2)
#endif

#define NYI(...)                                                           \
  fmt::print(fg(fmt::color::green_yellow) | fmt::emphasis::bold, "[NYI]"); \
  fmt::print(" {}:{} {}\n", __FILE__, __LINE__, fmt::format(__VA_ARGS__));

}  // namespace mllm
