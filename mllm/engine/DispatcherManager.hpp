/**
 * @file DispatcherManager.hpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-25
 *
 */
#pragma once

#include <future>
#include <memory>

#include "mllm/engine/Task.hpp"
#include "mllm/engine/Dispatcher.hpp"
#include "mllm/core/DeviceTypes.hpp"
#include "mllm/utils/SymbolTable.hpp"

namespace mllm {

class DispatcherManager {
 public:
  using ptr_t = std::shared_ptr<DispatcherManager>;
};

}  // namespace mllm
