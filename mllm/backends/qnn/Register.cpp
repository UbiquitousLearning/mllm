// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/DeviceTypes.hpp"
#include "mllm/engine/Context.hpp"
#include "mllm/mllm.hpp"
#include "mllm/backends/qnn/QNNBackend.hpp"
#include "mllm/backends/qnn/QNNDispatcher.hpp"
#include "CustomLayers.hpp"

namespace mllm {

// export initQnnBackend function to initialize QNN backend
void initQnnBackend() {
  MLLM_RT_ASSERT(isQnnAvailable());
  auto& ctx = Context::instance();

  // 1. Register backend
  auto backend = std::make_shared<qnn::QNNBackend>();
  ctx.registerBackend(backend);

  // 2. Initialize memory manager
  ctx.memoryManager()->registerAllocator(kQNN, backend->allocator(),
                                         {
                                             .really_large_tensor_threshold = 0,
                                             .using_buddy_mem_pool = false,
                                         });
  // 3. Initialize dispatcher manager
  ctx.dispatcherManager()->registerDispatcher(
      createQNNDispatcher(ctx.dispatcherManager()->getExecutor(), qnn::QNNDispatcherOptions()));

  // register QNN custom ops
  Context::instance().registerCustomizedOp(kQNN, "DequantizeAdd",
                                           std::shared_ptr<BaseOpFactory>((BaseOpFactory*)(new qnn::DequantizeAddFactory())));
}
}  // namespace mllm
