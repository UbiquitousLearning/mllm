// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/ascend/AscendAllocator.hpp"
#include "mllm/backends/ascend/memory/AscendMemoryManager.hpp"

#include "mllm/utils/Common.hpp"

namespace mllm::ascend {

AscendAllocator::AscendAllocator() {
    MLLM_INFO("AscendAllocator created with memory pool support");
}

AscendAllocator::~AscendAllocator() {
    std::lock_guard<std::mutex> lock(block_map_mutex_);
    if (!storage_to_block_id_.empty()) {
        MLLM_WARN("AscendAllocator destroyed with {} storage blocks still allocated",
                  storage_to_block_id_.size());
    }
}

bool AscendAllocator::alloc(Storage* storage) {
    auto& mem_manager = getAscendMemoryManager();
    int block_id = -1;
    mem_manager.allocateBlock(storage->size_, block_id);
    if (block_id < 0) {
        MLLM_ERROR("Failed to allocate block of size {} bytes from memory pool", storage->size_);
        return false;
    }

    mem_manager.getBlockPtr(block_id, storage->ptr_);
    if (storage->ptr_ == nullptr) {
        MLLM_ERROR("Failed to get pointer for block ID {}", block_id);
        mem_manager.freeBlock(block_id);
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(block_map_mutex_);
        storage_to_block_id_[storage->ptr_] = block_id;
    }

    MLLM_INFO("Allocated storage: size={} bytes, block ID={}, ptr={}",
               storage->size_, block_id, storage->ptr_);
    return true;
}

bool AscendAllocator::alloc(const Storage::ptr_t& storage) {
    return alloc(storage.get());
}

void AscendAllocator::free(const Storage::ptr_t& storage) {
    free(storage.get());
}

void AscendAllocator::free(Storage* storage) {
    if (storage->ptr_ == nullptr) {
        return;
    }

    int block_id = -1;
    {
        std::lock_guard<std::mutex> lock(block_map_mutex_);
        auto it = storage_to_block_id_.find(storage->ptr_);
        if (it != storage_to_block_id_.end()) {
            block_id = it->second;
            storage_to_block_id_.erase(it);
        }
    }

    if (block_id >= 0) {
        getAscendMemoryManager().freeBlock(block_id);
        MLLM_INFO("Freed storage: block ID={}, ptr={}", block_id, storage->ptr_);
    } else {
        MLLM_WARN("Attempted to free storage with no block ID mapping: ptr={}", storage->ptr_);
    }

    storage->ptr_ = nullptr;
}

bool AscendAllocator::generalAlloc(void** ptr, size_t cap, size_t align) {
    return true;
}

void AscendAllocator::generalFree(void* ptr) {
    
}

size_t AscendAllocator::allocSize(const Storage::ptr_t& storage) {
  // Ascend allocations don't require manual alignment padding
  // since AscendMemoryPool already provides proper alignment
  return storage->size_;
}

size_t AscendAllocator::allocSize(Storage* storage) {
  // Ascend allocations don't require manual alignment padding
  // since AscendMemoryPool already provides proper alignment
  return storage->size_;
}

size_t AscendAllocator::alignSize() const { return 64; }

std::shared_ptr<AscendAllocator> createAscendAllocator() { return std::make_shared<AscendAllocator>(); }

}  // namespace mllm::ascend
