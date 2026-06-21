// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <type_traits>

// We need this in module.hpp for GNU Toolchain.
template<typename T>
struct always_false : std::false_type {};
