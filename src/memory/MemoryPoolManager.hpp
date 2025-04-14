#ifndef MLLM_MEMORY_POOL_H
#define MLLM_MEMORY_POOL_H
#include "MemoryManager.hpp"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <Log.h>
#include <cstdint>
#include <mutex>

namespace mllm {
class MemoryPoolManager : public MemoryManager {
private:
    void sys_alloc(void **ptr, size_t size, size_t alignment) {
        assert(size > 0);
        // allocate a block of memory, void* is used to store the original pointer
        void *origin = (void *)malloc(size + sizeof(void *) + alignment - 1);
        assert(origin != nullptr);
        if (origin == nullptr) {
            *ptr = nullptr;
            return;
        }
        void **aligned = (void **)(((size_t)(origin) + sizeof(void *) + alignment - 1) & (~(alignment - 1)));
        aligned[-1] = origin;
        *ptr = aligned;
    }

    // 内存块定义（Block结构）
    struct FreeBlock {
        void *addr;      // 内存块起始地址
        size_t size;     // 内存块大小
        FreeBlock *next; // 链表指针（双向链表设计）

        FreeBlock(void *a, size_t s) :
            addr(a), size(s), next(nullptr) {
        }
    };

    // 空闲块链表头指针（链表管理机制）
    FreeBlock *free_blocks_ = nullptr; // 线程安全锁（线程安全要求）
    std::mutex free_list_mutex_;

public:
    MemoryPoolManager() :
        MemoryPoolManager(DEFAULT_POOL_SIZE, DEFAULT_ALIGNMENT) {
    }
    static constexpr size_t DEFAULT_POOL_SIZE = 1024 * 1024 * 1024; // 默认1GB内存池
    static constexpr size_t DEFAULT_ALIGNMENT = 128;                // 默认128字节对齐
    MemoryPoolManager(size_t init_size, size_t alignment = 128) {
        // 初始化时分配整块内存（预分配机制）
#if defined(_WIN32)
        void *pool = _aligned_malloc(init_size, alignment);
#else
        void *pool;
        posix_memalign(&pool, alignment, init_size);
#endif
        // 将初始内存加入空闲链表（初始化逻辑）
        free_blocks_ = new FreeBlock(pool, init_size);
    }

    void alloc(void **ptr, size_t size, size_t alignment) override {
        std::lock_guard<std::mutex> lock(free_list_mutex_);

        // 遍历空闲链表寻找合适块（搜索策略）
        FreeBlock **prev = &free_blocks_;
        while (*prev != nullptr) {
            FreeBlock *curr = *prev;

            // 计算对齐偏移（对齐要求）
            uintptr_t addr = reinterpret_cast<uintptr_t>(curr->addr);
            size_t offset = (alignment - (addr % alignment)) % alignment;

            if (curr->size >= (size + offset)) {
                // 分割内存块（块分割逻辑）
                void *allocated_addr = reinterpret_cast<void *>(addr + offset);
                *ptr = allocated_addr;

                // 更新剩余块信息
                size_t remaining = curr->size - offset - size;
                if (remaining > 0) {
                    FreeBlock *new_block = new FreeBlock(
                        reinterpret_cast<void *>(addr + offset + size),
                        remaining);
                    new_block->next = curr->next;
                    *prev = new_block;
                } else {
                    *prev = curr->next;
                }
                delete curr;
                // std::cout << "mp " << size << "  " << DEFAULT_POOL_SIZE << std::endl;
                return;
            }
            prev = &(curr->next);
        }

        // 退化到系统分配（用户提供的SystemMemoryManager逻辑）
        // std::cout << "sy " << size << "  " << DEFAULT_POOL_SIZE << std::endl;
        sys_alloc(ptr, size, alignment);
    }

    void free(void *ptr) override {
        std::lock_guard<std::mutex> lock(free_list_mutex_);

        // 创建新空闲块（释放逻辑）
        FreeBlock *new_block = new FreeBlock(ptr, 0); // 需要计算实际大小

        // 合并相邻块（碎片优化）
        FreeBlock **prev = &free_blocks_;
        while (*prev != nullptr) {
            FreeBlock *curr = *prev;
            uintptr_t curr_end = reinterpret_cast<uintptr_t>(curr->addr) + curr->size;

            // 前向合并
            if (reinterpret_cast<uintptr_t>(new_block->addr) == curr_end) {
                curr->size += new_block->size;
                delete new_block;
                return;
            }

            // 后向合并
            uintptr_t new_block_end = reinterpret_cast<uintptr_t>(new_block->addr) + new_block->size;
            if (reinterpret_cast<uintptr_t>(curr->addr) == new_block_end) {
                new_block->size += curr->size;
                new_block->next = curr->next;
                *prev = new_block;
                delete curr;
                return;
            }

            prev = &(curr->next);
        }

        // 插入新块到链表
        new_block->next = free_blocks_;
        free_blocks_ = new_block;
    }
};
} // namespace mllm
#endif // MLLM_MEMORY_POOL_H