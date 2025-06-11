#ifndef MLLM_MEMORY_POOL_H
#define MLLM_MEMORY_POOL_H

#include "MemoryManager.hpp"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <mutex>

namespace mllm {

// 高性能临时内存池，仅服务 activation 分配，模型权重/KV Cache 请使用系统分配
class MemoryPoolManager : public MemoryManager {
private:
    struct Header {
        void *raw_ptr; // 新增原始指针
        size_t size;
        bool is_sys;
    };

    struct Block {
        uintptr_t addr;
        size_t size;
        Block *prev;
        Block *next;
    };

    Block *free_head_ = nullptr;
    size_t pool_size_;
    size_t alignment_;
    std::mutex mutex_;

    // 参数场景配置
    static constexpr size_t INITIAL_POOL = 128ULL << 20; // 128MB
    static constexpr size_t EXPAND_UNIT = 256ULL << 20;  // 256MB 每次线性扩容
    static constexpr double LARGE_RATIO = 0.4;           // >40% 大块走系统
    static constexpr double POOL_THRESHOLD = 0.15;       // <15% 触发扩容

    // 系统分配/释放，用于大块或回退
    void sys_alloc(void **ptr, size_t size, size_t alignment) {
        assert(size > 0);
        void *raw = nullptr;
#if defined(_WIN32)
        raw = _aligned_malloc(size + sizeof(Header) + alignment - 1, alignment);
#else
        raw = std::malloc(size + sizeof(Header) + alignment - 1);
#endif
        assert(raw);
        uintptr_t base = reinterpret_cast<uintptr_t>(raw) + sizeof(Header);
        uintptr_t aligned = (base + alignment - 1) & ~(alignment - 1);
        Header *hdr = reinterpret_cast<Header *>(aligned - sizeof(Header));
        hdr->raw_ptr = raw; // 记录原始指针
        hdr->size = size;
        hdr->is_sys = true;
        *ptr = reinterpret_cast<void *>(aligned);
    }

    void sys_free(void *ptr) {
        if (!ptr) return;
        uintptr_t user_ptr = reinterpret_cast<uintptr_t>(ptr);
        Header *hdr = reinterpret_cast<Header *>(user_ptr - sizeof(Header)); // 找到Header
#if defined(_WIN32)
        _aligned_free(hdr->raw_ptr); // 释放原始指针
#else
        std::free(hdr->raw_ptr);
#endif
    }

    // 线性扩容
    void expand(size_t min_bytes) {
        size_t alloc_size = ((min_bytes + EXPAND_UNIT - 1) / EXPAND_UNIT) * EXPAND_UNIT;
        void *raw = nullptr;
#if defined(_WIN32)
        raw = _aligned_malloc(alloc_size, alignment_);
#else
        posix_memalign(&raw, alignment_, alloc_size);
#endif
        assert(raw);
        auto *blk = new Block{reinterpret_cast<uintptr_t>(raw), alloc_size, nullptr, free_head_};
        if (free_head_) free_head_->prev = blk;
        free_head_ = blk;
        pool_size_ += alloc_size;
    }

    size_t total_free() const {
        size_t sum = 0;
        for (auto *b = free_head_; b; b = b->next) sum += b->size;
        return sum;
    }

    Header *hdr_of(void *ptr) {
        return reinterpret_cast<Header *>(reinterpret_cast<uintptr_t>(ptr) - sizeof(Header));
    }

    // 插入并合并空闲块
    void insert_block(uintptr_t addr, size_t size) {
        Block *cur = free_head_;
        Block *prev = nullptr;
        while (cur && cur->addr < addr) {
            prev = cur;
            cur = cur->next;
        }
        auto *blk = new Block{addr, size, prev, cur};
        if (prev)
            prev->next = blk;
        else
            free_head_ = blk;
        if (cur) cur->prev = blk;
        // 向前合并
        if (blk->prev && blk->prev->addr + blk->prev->size == blk->addr) {
            blk->prev->size += blk->size;
            blk->prev->next = blk->next;
            if (blk->next) blk->next->prev = blk->prev;
            delete blk;
            blk = blk->prev;
        }
        // 向后合并
        if (blk->next && blk->addr + blk->size == blk->next->addr) {
            blk->size += blk->next->size;
            Block *tmp = blk->next;
            blk->next = tmp->next;
            if (tmp->next) tmp->next->prev = blk;
            delete tmp;
        }
    }

public:
    MemoryPoolManager(size_t init = INITIAL_POOL, size_t align = 128) :
        pool_size_(0), alignment_(align) {
        expand(init);
    }

    ~MemoryPoolManager() override {
        std::lock_guard<std::mutex> lg(mutex_);
        for (auto *b = free_head_; b;) {
            auto *next = b->next;
#if defined(_WIN32)
            _aligned_free(reinterpret_cast<void *>(b->addr));
#else
            std::free(reinterpret_cast<void *>(b->addr));
#endif
            delete b; // 释放 Block 对象
            b = next;
        }
    }

    void alloc(void **ptr, size_t size, size_t alignment) override {
        assert(size > 0);
        std::lock_guard<std::mutex> lg(mutex_);
        size_t req = size + sizeof(Header);
        // 大块走系统
        if (req > pool_size_ * LARGE_RATIO) {
            sys_alloc(ptr, size, alignment);
            return;
        }
        // 小块服务，需要空间时线性扩容
        if (total_free() < req || total_free() < pool_size_ * POOL_THRESHOLD) {
            expand(req);
        }
        // 首适应分配
        for (auto *b = free_head_; b; b = b->next) {
            uintptr_t start = b->addr;
            uintptr_t base = start + sizeof(Header);
            uintptr_t aligned = (base + alignment - 1) & ~(alignment - 1);
            size_t padding = aligned - start - sizeof(Header);
            if (b->size >= padding + req) {
                uintptr_t user = aligned;
                auto *hdr = reinterpret_cast<Header *>(user - sizeof(Header));
                hdr->size = size;
                hdr->is_sys = false;
                *ptr = reinterpret_cast<void *>(user);
                // 更新块
                uintptr_t next = user + size;
                size_t remain = b->size - (padding + req);
                if (remain > sizeof(Header)) {
                    b->addr = next;
                    b->size = remain;
                } else {
                    if (b->prev)
                        b->prev->next = b->next;
                    else
                        free_head_ = b->next;
                    if (b->next) b->next->prev = b->prev;
                    delete b;
                }
                return;
            }
        }
        // 再回退系统
        sys_alloc(ptr, size, alignment);
    }

    void free(void *ptr) override {
        if (!ptr) return;
        std::lock_guard<std::mutex> lg(mutex_);
        auto *hdr = hdr_of(ptr);
        if (!hdr->is_sys) {
            insert_block(reinterpret_cast<uintptr_t>(hdr), hdr->size + sizeof(Header));
        } else {
            sys_free(ptr);
        }
    }
};

} // namespace mllm

#endif // MLLM_MEMORY_POOL_H
