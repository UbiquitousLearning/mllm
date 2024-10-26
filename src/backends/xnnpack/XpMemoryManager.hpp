/**
 * @file MemoryManager.hpp
 * @author Chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2024-10-03
 *
 * @copyright Copyright (c) 2024
 *
 */
#pragma once

#include "MemoryManager.hpp"

namespace mllm::xnnpack {

class XpMemoryManager : public MemoryManager {
public:
    XpMemoryManager() = default;
    ~XpMemoryManager() override;

    void alloc(void **ptr, size_t size, size_t alignment) override;

    void free(void *ptr) override;
};

} // namespace mllm::xnnpack