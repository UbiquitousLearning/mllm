#include "MemoryManager.hpp"
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include<list>

using std::unordered_map;
using std::uint64_t;
using std::list;

#ifndef MLLM_MEMORY_POOL_H
#define MLLM_MEMORY_POOL_H


namespace mllm {
// 这是一个功能和malloc/free相同的分配/释放内存/显存的函数。

struct FreeBlock{
    void* addr;
    size_t size;
};

/**
 * 内存管理类 mem pool ... TODO 管理HostMemory
 */
class MemoryPoolManager : public MemoryManager{
public:
    MemoryPoolManager(size_t pool_size,size_t base_alignment);
    
    ~MemoryPoolManager();

    void alloc(void **ptr, size_t size,size_t alignment) override;

    void free(void **ptr) override;

private:
    // memory buffer
    void * data_; 
    int n_free_blocks_;
    size_t base_alignment_;
    list<struct FreeBlock> free_blocks_;
    unordered_map<uint64_t, size_t> block_size_;

    #ifdef MLLM_ALLOCATOR_DEBUG

    #endif
};

} // namespace mllm
#endif // MLLM_MEMORY_POOL_H
