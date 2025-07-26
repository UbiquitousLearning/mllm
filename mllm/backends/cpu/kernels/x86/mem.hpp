/**
 * @file mem.hpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-23
 *
 */
#pragma once

#include <cstddef>

namespace mllm::cpu {

// aligned to 512bit(64B) vector.
void x86_align_alloc(void** ptr, size_t required_bytes, size_t align = 64);

void x86_align_free(void* ptr);

}  // namespace mllm::cpu
