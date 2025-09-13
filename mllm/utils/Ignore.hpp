// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <type_traits>

namespace mllm {
template<typename T>
inline void ignore(T&&) noexcept {}
}  // namespace mllm

#define IGNORE(expr) (void)(::mllm::ignore(expr))
