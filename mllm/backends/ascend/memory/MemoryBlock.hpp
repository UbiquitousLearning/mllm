// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <cstddef>

namespace mllm::ascend {

struct MemoryBlock {
    int64_t block_id_;
    size_t block_size_;         
    size_t requested_size_ = 0; 
    void* address_ = nullptr;

    size_t alloc_seq_ = 0;  
    size_t reuse_count_ = 0;
};

}  // namespace mllm::ascend
