/**
 * @file JITRunner.hpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-29
 *
 */
#pragma once

#include <memory>
#include <unordered_map>
#include <functional>
#include <vector>

#include "mllm/compile/ir/Node.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/RTTIHelper.hpp"

namespace mllm::jit {

class JITRunner;

}  // namespace mllm::jit
