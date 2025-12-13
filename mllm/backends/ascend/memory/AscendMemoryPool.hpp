// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <atomic>
#include <mutex>
#include <unordered_map>
#include <cstdint>
#include <cstddef>
#include "MemoryBlock.hpp"

namespace mllm::ascend {

class AscendMemoryPool {
public:
    explicit AscendMemoryPool(size_t pool_size);
    ~AscendMemoryPool();

    void allocateBlock(uint32_t size, int& block_id);

    void freeBlock(int block_id);

    void getBlockPtr(int block_id, void*& addr);

private:
    uint64_t generateBlocksId();

    std::atomic<uint64_t> id_ = 0;
    std::mutex block_mutex_;
    
    void* base_mem_ptr_ = nullptr;
    void* cur_mem_ptr_ = nullptr;
    int64_t remain_size_ = 0;
    
    std::unordered_map<int, MemoryBlock> used_blocks_;  
    std::unordered_map<int, MemoryBlock> free_blocks_;  
};

}  // namespace mllm::ascend
