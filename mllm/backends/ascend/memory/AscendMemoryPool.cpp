// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <atb/types.h>
#include <acl/acl.h>
#include "AscendMemoryPool.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/utils/Log.hpp"

namespace mllm::ascend {

constexpr size_t POOL_SIZE = 104857600;  // 100 MiB，

AscendMemoryPool::AscendMemoryPool(size_t pool_size = POOL_SIZE) {
    auto ret = aclrtMalloc(&base_mem_ptr_, pool_size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        MLLM_ERROR("Failed to allocate Ascend memory pool of size {} bytes: ACL error {}", 
                   pool_size, int(ret));
        base_mem_ptr_ = nullptr;
        cur_mem_ptr_ = nullptr;
        remain_size_ = 0;
        return;
    }
    cur_mem_ptr_ = base_mem_ptr_;
    remain_size_ = pool_size;
}

AscendMemoryPool::~AscendMemoryPool() {
    if (base_mem_ptr_ != nullptr) {
        auto ret = aclrtFree(base_mem_ptr_);
        if (ret != ACL_SUCCESS) {
            MLLM_ERROR("Failed to free Ascend memory pool: ACL error {}", int(ret));
        }
    }
    MLLM_INFO("release MemoryPool success");
}

uint64_t AscendMemoryPool::generateBlocksId() {
    return static_cast<uint64_t>(id_.fetch_add(1, std::memory_order_relaxed));
}

void AscendMemoryPool::allocateBlock(uint32_t size, int &block_id) {
    std::unique_lock<std::mutex> lock(block_mutex_);

    size_t align_size = ((size + 31) & ~31) + 32;  

    for (auto it = free_blocks_.begin(); it != free_blocks_.end(); it++) {
        if (it->second.block_size_ >= align_size) {
            block_id = it->second.block_id_;
            used_blocks_.insert(*it);
            free_blocks_.erase(it);
            MLLM_INFO("find free block id {} to allocate", block_id);
            return;
        }
    }

    if (remain_size_ > align_size) {
        block_id = generateBlocksId();
        uint64_t cur_mem_ptr_align = (reinterpret_cast<uint64_t>(cur_mem_ptr_) + 63) & ~63;  
        remain_size_ -= (cur_mem_ptr_align - reinterpret_cast<uint64_t>(cur_mem_ptr_));
        cur_mem_ptr_ = reinterpret_cast<void *>(cur_mem_ptr_align);

        MemoryBlock block = {block_id, align_size, cur_mem_ptr_};
        used_blocks_.insert({block_id, block});
        remain_size_ -= align_size;
        cur_mem_ptr_ = reinterpret_cast<uint8_t *>(cur_mem_ptr_) + align_size;
        MLLM_INFO("allocate block id {} for size {}", block_id, align_size);
        return;
    }
    MLLM_ERROR("allocate block fail");
}

void AscendMemoryPool::freeBlock(int block_id) {
    std::unique_lock<std::mutex> lock(block_mutex_);

    if (block_id < 0) {
        MLLM_INFO("skip over the invalid block id {}", block_id);
        return;
    }

    auto it = used_blocks_.find(block_id);
    if (it != used_blocks_.end()) {
        free_blocks_.insert(*it);
        used_blocks_.erase(it);
    } else {
        MLLM_ERROR("Double free block id {}", block_id);
    }
}

void AscendMemoryPool::getBlockPtr(int block_id, void *&addr) {
    std::unique_lock<std::mutex> lock(block_mutex_);

    if (block_id < 0) {
        MLLM_INFO("Invalid block id {} to get ptr", block_id);
        return;
    }

    auto it = used_blocks_.find(block_id);
    if (it != used_blocks_.end()) {
        addr = it->second.address_;
    } else {
        MLLM_ERROR("Get block address error, block id {}", block_id);
    }
}

}  // namespace mllm::ascend
