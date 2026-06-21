// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <atomic>
#include <mutex>
#include <unordered_map>
#include <map>
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

    // Print memory allocation statistics.
    void printStats() const;

private:
    uint64_t generateBlocksId();

    std::atomic<uint64_t> id_ = 0;
    mutable std::mutex block_mutex_;

    void* base_mem_ptr_ = nullptr;
    void* cur_mem_ptr_ = nullptr;
    int64_t remain_size_ = 0;
    size_t pool_size_ = 0;  // Total pool size.

    std::unordered_map<int, MemoryBlock> used_blocks_;
    std::unordered_map<int, MemoryBlock> free_blocks_;

    // === Debug statistics ===
    size_t total_alloc_count_ = 0;
    size_t total_free_count_ = 0;
    size_t total_reuse_count_ = 0;
    size_t total_new_alloc_count_ = 0;
    std::map<size_t, size_t> alloc_size_histogram_;

    // === Memory utilization statistics ===
    size_t total_requested_bytes_ = 0;
    size_t total_allocated_bytes_ = 0;
    size_t large_block_waste_count_ = 0;
    size_t large_block_waste_bytes_ = 0;
};

}  // namespace mllm::ascend
