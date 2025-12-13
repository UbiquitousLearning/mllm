// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <acl/acl.h>

#include "AscendMemoryManager.hpp"
#include "mllm/backends/ascend/AscendCommon.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/utils/Log.hpp"

namespace mllm::ascend {

AscendMemoryManager::AscendMemoryManager() = default;

AscendMemoryManager &getAscendMemoryManager() {
    static AscendMemoryManager instance;
    return instance;
}

void AscendMemoryManager::createMemoryPool(size_t pool_size)
{
    uint32_t device_count = 0;
    auto ret = aclrtGetDeviceCount(&device_count);
    MLLM_ACL_CHECK(ret);
    for (size_t i = 0; i < device_count; i++) {

        aclrtSetDevice(i);

        std::shared_ptr<AscendMemoryPool> memory_pool = std::make_shared<AscendMemoryPool>(pool_size);
        memory_pools_.push_back(memory_pool);
        MLLM_INFO("create mempool for device {} success", i);
    }
}

int32_t AscendMemoryManager::getDeviceId()
{
    int32_t device_id = -1;
    auto ret = aclrtGetDevice(&device_id);
    MLLM_ACL_CHECK(ret);
    return device_id;
}

std::shared_ptr<AscendMemoryPool> &AscendMemoryManager::getMemoryPool()
{
    size_t device_id = static_cast<size_t>(getDeviceId());
    if (device_id >= memory_pools_.size()) {
        MLLM_ERROR_EXIT(::mllm::ExitCode::kAscendError, "Invalid device id {}", device_id);
    }
    return memory_pools_[device_id];
}

void AscendMemoryManager::allocateBlock(uint32_t size, int &block_id)
{
    getMemoryPool()->allocateBlock(size, block_id);
}

void AscendMemoryManager::freeBlock(int block_id)
{
    getMemoryPool()->freeBlock(block_id);
}

void AscendMemoryManager::getBlockPtr(int block_id, void *&addr)
{
    getMemoryPool()->getBlockPtr(block_id, addr);
}
}  // namespace mllm::ascend
