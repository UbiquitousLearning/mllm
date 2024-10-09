#include "backends/xnnpack/XpMemoryManager.hpp"
#include "xnnpack/allocator.h"

namespace mllm::xnnpack {
XpMemoryManager::~XpMemoryManager() = default;

void XpMemoryManager::alloc(void **ptr, size_t size, size_t alignment) {
    xnn_allocate_zero_simd_memory(size + XNN_EXTRA_BYTES);
}

void XpMemoryManager::free(void *ptr) {
    xnn_release_simd_memory(ptr);
}

} // namespace mllm::xnnpack