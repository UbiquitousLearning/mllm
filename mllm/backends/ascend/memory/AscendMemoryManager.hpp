// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <vector>
#include <cstdint>
#include "AscendMemoryPool.hpp"

namespace mllm::ascend {


class AscendMemoryManager {
public:  
    AscendMemoryManager();

    void createMemoryPool(size_t pool_size);

    int32_t getDeviceId();

    std::shared_ptr<AscendMemoryPool> &getMemoryPool();

    void allocateBlock(uint32_t size, int &block_id);

    void freeBlock(int block_id);

    void getBlockPtr(int block_id, void *&addr);

private:
    std::vector<std::shared_ptr<AscendMemoryPool>> memory_pools_;
};

AscendMemoryManager &getAscendMemoryManager();

}  // namespace mllm::ascend
