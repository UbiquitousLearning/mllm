// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <algorithm>
#include <vector>
#include <atb/types.h>
#include <acl/acl.h>
#include "AscendMemoryPool.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/utils/Log.hpp"

namespace mllm::ascend {

constexpr size_t POOL_SIZE = 1073741824;  // 1 GiB

AscendMemoryPool::AscendMemoryPool(size_t pool_size = POOL_SIZE) {
    MLLM_INFO("Attempting to allocate Ascend memory pool of {} MB", pool_size / (1024 * 1024));
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
    pool_size_ = pool_size;  
    // MLLM_INFO("Successfully allocated Ascend memory pool: {} MB at address {}",
    //           pool_size / (1024 * 1024), base_mem_ptr_);
    fmt::print("[Ascend] Memory pool allocated: {} MB\n", pool_size / (1024 * 1024));
}

AscendMemoryPool::~AscendMemoryPool() {
    printStats();

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

    total_alloc_count_++;
    total_requested_bytes_ += size;
    total_allocated_bytes_ += align_size;

    // <1KB, <64KB, <1MB, <16MB, >=16MB
    size_t size_bucket;
    if (align_size < 1024) size_bucket = 1;           // <1KB
    else if (align_size < 64*1024) size_bucket = 64;  // <64KB
    else if (align_size < 1024*1024) size_bucket = 1024;  // <1MB
    else if (align_size < 16*1024*1024) size_bucket = 16*1024;  // <16MB
    else size_bucket = 16*1024 + 1;  // >=16MB
    alloc_size_histogram_[size_bucket]++;

    // Best-Fit
    auto best_fit = free_blocks_.end();
    size_t best_fit_size = SIZE_MAX;

    for (auto it = free_blocks_.begin(); it != free_blocks_.end(); it++) {
        if (it->second.block_size_ >= align_size) {
            if (it->second.block_size_ < best_fit_size) {
                best_fit = it;
                best_fit_size = it->second.block_size_;
            }
        }
    }

    if (best_fit != free_blocks_.end()) {
        block_id = best_fit->second.block_id_;
        size_t actual_block_size = best_fit->second.block_size_;
        size_t block_waste = actual_block_size - align_size;

        best_fit->second.requested_size_ = size;
        best_fit->second.alloc_seq_ = total_alloc_count_;
        best_fit->second.reuse_count_++;

        if (actual_block_size >= align_size * 4 && block_waste >= 1024*1024) {
            large_block_waste_count_++;
            large_block_waste_bytes_ += block_waste;

        }

        used_blocks_.insert(*best_fit);
        free_blocks_.erase(best_fit);
        total_reuse_count_++;  
        MLLM_INFO("find free block id {} to allocate (best-fit)", block_id);
        return;
    }

    if (remain_size_ > align_size) {
        block_id = generateBlocksId();
        uint64_t cur_mem_ptr_align = (reinterpret_cast<uint64_t>(cur_mem_ptr_) + 63) & ~63;
        remain_size_ -= (cur_mem_ptr_align - reinterpret_cast<uint64_t>(cur_mem_ptr_));
        cur_mem_ptr_ = reinterpret_cast<void *>(cur_mem_ptr_align);

        MemoryBlock block = {block_id, align_size, size, cur_mem_ptr_, total_alloc_count_, 0};
        used_blocks_.insert({block_id, block});
        remain_size_ -= align_size;
        cur_mem_ptr_ = reinterpret_cast<uint8_t *>(cur_mem_ptr_) + align_size;
        total_new_alloc_count_++;  
        MLLM_INFO("allocate block id {} for size {}", block_id, align_size);
        return;
    }

    lock.unlock();  
    printStats();
    lock.lock();

    // Log free block sizes for debugging
    size_t total_free = 0;
    for (const auto& [id, block] : free_blocks_) {
        total_free += block.block_size_;
    }
    MLLM_ERROR("allocate block fail: requested={} bytes, remain_size={} bytes, free_blocks={} (total {} bytes), used_blocks={}",
               align_size, remain_size_, free_blocks_.size(), total_free, used_blocks_.size());

    // fmt::print("\n========== Top 50 Used Blocks (at allocation failure) ==========\n");
    // std::vector<std::pair<int, MemoryBlock>> used_vec(used_blocks_.begin(), used_blocks_.end());
    // std::sort(used_vec.begin(), used_vec.end(),
    //           [](const auto& a, const auto& b) { return a.second.block_size_ > b.second.block_size_; });

    // size_t display_count = std::min(size_t(50), used_vec.size());
    // for (size_t i = 0; i < display_count; i++) {
    //     const auto& [id, block] = used_vec[i];
    //     double utilization = block.requested_size_ > 0 ? 100.0 * block.requested_size_ / block.block_size_ : 0.0;
    //     fmt::print("  Block #{}: size={:.2f} MB, requested={:.2f} KB, utilization={:.1f}%, alloc_seq={}, reuse_count={}\n",
    //                id,
    //                block.block_size_ / (1024.0 * 1024.0),
    //                block.requested_size_ / 1024.0,
    //                utilization,
    //                block.alloc_seq_,
    //                block.reuse_count_);
    // }
    // fmt::print("================================================================\n\n");
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
        total_free_count_++;  
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

void AscendMemoryPool::printStats() const {
    std::unique_lock<std::mutex> lock(block_mutex_);

    size_t used_bytes = 0;
    size_t free_bytes = 0;
    for (const auto& [id, block] : used_blocks_) {
        used_bytes += block.block_size_;
    }
    for (const auto& [id, block] : free_blocks_) {
        free_bytes += block.block_size_;
    }

    std::map<size_t, size_t> free_size_distribution;
    for (const auto& [id, block] : free_blocks_) {
        size_t bucket;
        if (block.block_size_ < 1024) bucket = 1;
        else if (block.block_size_ < 64*1024) bucket = 64;
        else if (block.block_size_ < 1024*1024) bucket = 1024;
        else if (block.block_size_ < 16*1024*1024) bucket = 16*1024;
        else bucket = 16*1024 + 1;
        free_size_distribution[bucket]++;
    }

    std::map<size_t, size_t> used_size_distribution;
    size_t low_utilization_count = 0;
    size_t low_utilization_bytes = 0;
    size_t never_reused_count = 0;
    size_t never_reused_bytes = 0;

    for (const auto& [id, block] : used_blocks_) {
        size_t bucket;
        if (block.block_size_ < 1024) bucket = 1;
        else if (block.block_size_ < 64*1024) bucket = 64;
        else if (block.block_size_ < 1024*1024) bucket = 1024;
        else if (block.block_size_ < 16*1024*1024) bucket = 16*1024;
        else bucket = 16*1024 + 1;
        used_size_distribution[bucket]++;

        if (block.requested_size_ > 0 && block.block_size_ > 0) {
            double utilization = 100.0 * block.requested_size_ / block.block_size_;
            if (utilization < 20.0 && block.block_size_ >= 1024*1024) {
                low_utilization_count++;
                low_utilization_bytes += (block.block_size_ - block.requested_size_);
            }
        }

        if (block.reuse_count_ == 0 && block.block_size_ >= 1024*1024) {
            never_reused_count++;
            never_reused_bytes += block.block_size_;
        }
    }

    fmt::print("\n========== Ascend Memory Pool Statistics ==========\n");
    fmt::print("Allocation Strategy: Best-Fit\n");
    fmt::print("Pool size: {} MB\n", pool_size_ / (1024 * 1024));
    fmt::print("Remain size: {} MB ({} bytes)\n", remain_size_ / (1024 * 1024), remain_size_);
    fmt::print("Used: {} MB ({} blocks)\n", used_bytes / (1024 * 1024), used_blocks_.size());
    fmt::print("Free: {} MB ({} blocks)\n", free_bytes / (1024 * 1024), free_blocks_.size());

    fmt::print("\n--- Allocation Stats ---\n");
    fmt::print("Total alloc requests: {}\n", total_alloc_count_);
    fmt::print("  - New allocations (from remain): {}\n", total_new_alloc_count_);
    fmt::print("  - Reused from free blocks: {}\n", total_reuse_count_);
    fmt::print("Total free calls: {}\n", total_free_count_);
    fmt::print("Reuse rate: {:.1f}%\n",
               total_alloc_count_ > 0 ? 100.0 * total_reuse_count_ / total_alloc_count_ : 0.0);

    fmt::print("\n--- Memory Efficiency ---\n");
    fmt::print("Total requested: {:.2f} MB\n", total_requested_bytes_ / (1024.0 * 1024.0));
    fmt::print("Total allocated: {:.2f} MB\n", total_allocated_bytes_ / (1024.0 * 1024.0));
    if (total_allocated_bytes_ > 0) {
        double efficiency = 100.0 * total_requested_bytes_ / total_allocated_bytes_;
        double waste = total_allocated_bytes_ - total_requested_bytes_;
        fmt::print("Memory efficiency: {:.2f}%\n", efficiency);
        fmt::print("Total waste from alignment: {:.2f} MB ({:.2f}%)\n",
                   waste / (1024.0 * 1024.0),
                   100.0 - efficiency);
    }
    fmt::print("Large block waste events: {} times\n", large_block_waste_count_);
    fmt::print("Total large block waste: {:.2f} MB\n", large_block_waste_bytes_ / (1024.0 * 1024.0));

    fmt::print("\n--- Current Used Blocks Waste Analysis ---\n");
    fmt::print("Low utilization blocks (<20%%, >1MB): {} blocks, wasting {:.2f} MB\n",
               low_utilization_count, low_utilization_bytes / (1024.0 * 1024.0));
    fmt::print("Never reused blocks (likely long-lived): {} blocks, {:.2f} MB\n",
               never_reused_count, never_reused_bytes / (1024.0 * 1024.0));

    fmt::print("\n--- Allocation Size Distribution (requests) ---\n");
    for (const auto& [bucket, count] : alloc_size_histogram_) {
        const char* label;
        if (bucket == 1) label = "<1KB";
        else if (bucket == 64) label = "1KB-64KB";
        else if (bucket == 1024) label = "64KB-1MB";
        else if (bucket == 16*1024) label = "1MB-16MB";
        else label = ">=16MB";
        fmt::print("  {}: {} requests\n", label, count);
    }

    fmt::print("\n--- Used Blocks Size Distribution ---\n");
    for (const auto& [bucket, count] : used_size_distribution) {
        const char* label;
        if (bucket == 1) label = "<1KB";
        else if (bucket == 64) label = "1KB-64KB";
        else if (bucket == 1024) label = "64KB-1MB";
        else if (bucket == 16*1024) label = "1MB-16MB";
        else label = ">=16MB";
        fmt::print("  {}: {} blocks\n", label, count);
    }

    fmt::print("\n--- Free Blocks Size Distribution ---\n");
    for (const auto& [bucket, count] : free_size_distribution) {
        const char* label;
        if (bucket == 1) label = "<1KB";
        else if (bucket == 64) label = "1KB-64KB";
        else if (bucket == 1024) label = "64KB-1MB";
        else if (bucket == 16*1024) label = "1MB-16MB";
        else label = ">=16MB";
        fmt::print("  {}: {} blocks\n", label, count);
    }
    fmt::print("====================================================\n\n");
}

}  // namespace mllm::ascend
