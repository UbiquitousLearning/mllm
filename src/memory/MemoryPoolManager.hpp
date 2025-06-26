#ifndef MLLM_MEMORY_POOL_H
#define MLLM_MEMORY_POOL_H

#include "MemoryManager.hpp"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <vector>

namespace mllm {

// 高性能临时内存池，仅服务 activation 分配，模型权重/KV Cache 请使用系统分配
class MemoryPoolManager : public MemoryManager {
private:
    std::vector<void *> raw_blocks_allocated_;
    struct Header {
        void *raw_ptr;  // 系统分配时的原始指针
        size_t size;    // 用户请求的大小
        size_t padding; // 为对齐产生的填充大小
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
        hdr->raw_ptr = raw;
        hdr->size = size;
        hdr->is_sys = true;
        hdr->padding = 0;
        *ptr = reinterpret_cast<void *>(aligned);
    }

    void sys_free(void *ptr) {
        if (!ptr) return;
        uintptr_t user_ptr = reinterpret_cast<uintptr_t>(ptr);
        Header *hdr = reinterpret_cast<Header *>(user_ptr - sizeof(Header));
#if defined(_WIN32)
        _aligned_free(hdr->raw_ptr);
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
        raw_blocks_allocated_.push_back(raw);
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
            Block *prev_block = blk->prev; // 在delete之前，安全地缓存 prev 指针
            prev_block->size += blk->size;
            prev_block->next = blk->next;
            if (blk->next) {
                blk->next->prev = prev_block;
            }
            delete blk;
            blk = prev_block; // 使用缓存的、安全的指针进行赋值
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
            delete b;
            b = next;
        }
        free_head_ = nullptr;

        for (void *raw_block : raw_blocks_allocated_) {
#if defined(_WIN32)
            _aligned_free(raw_block);
#else
            std::free(raw_block);
#endif
        }
        raw_blocks_allocated_.clear();
    }

    void alloc(void **ptr, size_t size, size_t alignment) override {
        assert(size > 0);
        std::lock_guard<std::mutex> lg(mutex_);

        size_t req_total = size + sizeof(Header);

        if (req_total > pool_size_ * LARGE_RATIO) {
            sys_alloc(ptr, size, alignment);
            return;
        }
        if (total_free() < req_total || total_free() < pool_size_ * POOL_THRESHOLD) {
            expand(req_total);
        }

        for (auto *b = free_head_; b; b = b->next) {
            uintptr_t start = b->addr;
            uintptr_t base = start + sizeof(Header);
            uintptr_t aligned = (base + alignment - 1) & ~(alignment - 1);

            size_t padding = aligned - start - sizeof(Header);
            size_t total_consumed = req_total + padding;

            if (b->size >= total_consumed) {
                uintptr_t user = aligned;
                auto *hdr = reinterpret_cast<Header *>(user - sizeof(Header));

                hdr->raw_ptr = nullptr;
                hdr->size = size;
                hdr->is_sys = false;
                hdr->padding = padding;

                *ptr = reinterpret_cast<void *>(user);

                size_t remain = b->size - total_consumed;
                if (remain > sizeof(Header)) {
                    b->addr = start + total_consumed;
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
        sys_alloc(ptr, size, alignment);
    }

    void free(void *ptr) override {
        if (!ptr) return;
        std::lock_guard<std::mutex> lg(mutex_);

        auto *hdr = hdr_of(ptr);

        if (!hdr->is_sys) {
            uintptr_t block_start = reinterpret_cast<uintptr_t>(ptr) - sizeof(Header) - hdr->padding;
            size_t block_size = hdr->size + sizeof(Header) + hdr->padding;
            insert_block(block_start, block_size);
        } else {
            sys_free(ptr);
        }
    }
};

} // namespace mllm

#endif // MLLM_MEMORY_POOL_H