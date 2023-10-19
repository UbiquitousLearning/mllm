#include "MemoryManager.hpp"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include<list>

using std::unordered_map;
using std::uint64_t;
using std::list;
#ifndef MLLM_MEMORY_POOL_H
#define MLLM_MEMORY_POOL_H
#define MLLM_ALLOCATOR_DEBUG


namespace mllm {
// 这是一个功能和malloc/free相同的分配/释放内存/显存的函数。

struct FreeBlock{
    void* addr;
    size_t size;
    FreeBlock(void*addr,size_t size):addr(addr),size(size){}
};

/**
 * 内存管理类 mem pool .
 */
class MemoryPoolManager : public MemoryManager{
public:
    MemoryPoolManager(size_t pool_size,size_t base_alignment);
    
    ~MemoryPoolManager();

    void alloc(void **ptr, size_t size,size_t alignment) override;

    void free(void *ptr) override;

#ifdef MLLM_ALLOCATOR_DEBUG
    void display();
    unordered_map<uint64_t,size_t> debug_free_blocks;
    unordered_map<uint64_t,size_t> debug_allocate_blocks;
#endif

private:
    // memory buffer
    void * data_; 
    int n_free_blocks_;
    size_t base_alignment_;
    list<struct FreeBlock> free_blocks_;
    unordered_map<uint64_t, size_t> block_size_;
};

inline size_t aligned_offset(size_t offset,size_t alignment){
    assert(alignment && !(alignment & (alignment - 1)));
    auto align = ((alignment - ( offset % alignment))) % alignment;
    return offset + align;
}

} // namespace mllm
#endif // MLLM_MEMORY_POOL_H
