// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <memory>
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/DeviceTypes.hpp"
#include "mllm/engine/Context.hpp"
#include "mllm/mllm.hpp"
#include "mllm/backends/ascend/AscendBackend.hpp"
#include "mllm/backends/ascend/AscendDispatcher.hpp"
#include "mllm/backends/ascend/memory/AscendMemoryManager.hpp"

namespace mllm {

void initAscendBackend() {
  auto& ctx = Context::instance();

  // 1. Create memory pool
  size_t pool_size = 100 * 1024 * 1024;  // 100MB, can be adjusted as needed
  ascend::getAscendMemoryManager().createMemoryPool(pool_size);
  MLLM_INFO("Ascend memory pool initialized");

  // 2. Register backend
  auto backend = std::make_shared<ascend::AscendBackend>();
  ctx.registerBackend(backend);

  // 3. Register allocator
  ctx.memoryManager()->registerAllocator(kAscend, backend->allocator(), MemoryManagerOptions());

  // 4. Register dispatcher
  auto dispatcher = ascend::createAscendDispatcher(ctx.dispatcherManager()->getExecutor(), 
                                                   ascend::AscendDispatcherOptions{});
  ctx.dispatcherManager()->registerDispatcher(dispatcher);
  MLLM_INFO("Ascend dispatcher registered");

  // 5. Register custom ops 
  // ctx.registerCustomizedOp(kAscend, "CustomOpName", 
  //                          std::make_shared<CustomOpFactory>());
}

}  // namespace mllm
