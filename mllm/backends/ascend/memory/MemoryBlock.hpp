// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <cstddef>

namespace mllm::ascend {

struct MemoryBlock {
    int64_t block_id_;          
    size_t block_size_;         
    void* address_ = nullptr;  
};

}  // namespace mllm::ascend
