#include "MemoryPoolManager.hpp"
#include "MemoryManager.hpp"
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iterator>

namespace mllm {
    MemoryPoolManager::MemoryPoolManager(size_t pool_size,size_t base_alignment):
        data_(nullptr),
        n_free_blocks_(0),
        base_alignment_(base_alignment)
    {
#if defined(_MSC_VER) || defined(__MINGW32__)
        data_ = _aligned_malloc(pool_size,base_alignment);
#else
        data_ = std::aligned_alloc(base_alignment,pool_size);
#endif

        n_free_blocks_ += 1;
        free_blocks_.emplace_back(data_,pool_size);
        #ifdef MLLM_ALLOCATOR_DEBUG
        std::cout<<"MemoryPoolManager init. Range from "<<data_ << " to "<< (void*)((char*)data_+pool_size) <<std::endl;
        #endif
    }

    void MemoryPoolManager::alloc(void **ptr, size_t size,size_t alignment){
        // ensure alignment is the power of 2
        assert((alignment & (alignment -1)) == 0);
        assert( alignment <= base_alignment_);

        // TODO: support different backends
        size = aligned_offset(size,base_alignment_);
        std::list<FreeBlock>::iterator best_fit_block;
        auto last_block = free_blocks_.end()--;
        size_t best_fit_size = SIZE_MAX;
        size_t max_avail = 0;
        for(auto it = free_blocks_.begin();it != last_block;it++){
            auto block_size = it->size;
            max_avail = std::max(max_avail,block_size);
            if(block_size >= size && block_size <= best_fit_size){
                best_fit_block = it;
                best_fit_size = block_size;
            }
        }
        if(best_fit_size == SIZE_MAX){
            auto block_size = last_block->size;
            max_avail = std::max(max_avail,block_size);
            if(block_size >= size){
                best_fit_block = last_block;
                best_fit_size = block_size;
            }else{
                MLLM_LOG_ERROR_STREAM << "Not enough space,max available is " << max_avail;
            }
        }
        #ifdef MLLM_ALLOCATOR_DEBUG
        debug_allocate_blocks[(uint64_t)best_fit_block->addr] = size;
        #endif
        *ptr = best_fit_block->addr;
        block_size_.emplace((uint64_t)best_fit_block->addr,size);
        if (best_fit_size > size){
            #ifdef MLLM_ALLOCATOR_DEBUG
            debug_free_blocks.erase((uint64_t)best_fit_block->addr);
            debug_free_blocks[(uint64_t)best_fit_block->addr + size] = best_fit_block->size - size;
            #endif
            best_fit_block->addr = (void*)((const char*)best_fit_block->addr + size);
            best_fit_block->size -= size;

        }else{
            // best_fit_size == size
            #ifdef MLLM_ALLOCATOR_DEBUG
            debug_free_blocks.erase((uint64_t)best_fit_block->addr);
            #endif
            free_blocks_.erase(best_fit_block);
        }
    }

    void MemoryPoolManager::free(void *ptr){
        size_t size = 0;
        assert(ptr != nullptr);
        auto ptr_addr = (uint64_t)ptr;

        if (auto iter = block_size_.find(ptr_addr);iter != block_size_.end()){
            size = iter->second;
        }else{
            // can not find size
            throw "can not find address of ptr";
        }
        #ifdef MLLM_ALLOCATOR_DEBUG
        debug_allocate_blocks.erase(ptr_addr);
        #endif
        for(auto iter=free_blocks_.begin();iter != free_blocks_.end();iter++){
            uint64_t block_addr = (uint64_t)iter->addr;
            auto block_size = iter->size;
            // check if the ptr is at the end of the block
            if (block_addr + block_size == ptr_addr){
                iter->size += size;
                #ifdef MLLM_ALLOCATOR_DEBUG
                debug_free_blocks[block_addr] = iter->size;
                #endif
                // check if we can merge next free block
                if (auto nxt = std::next(iter);nxt != free_blocks_.end() && (uint64_t)iter->addr + iter->size == (uint64_t)nxt->addr){
                    iter->size += nxt->size;
                    n_free_blocks_--;
                    free_blocks_.erase(nxt);

                    #ifdef MLLM_ALLOCATOR_DEBUG
                    // win有概率出现错误
                    // debug_free_blocks.erase((uint64_t)nxt->addr);
                    #endif
                }
                return;
            }

            // check if the ptr is at the front of the block
            if (ptr_addr + size == block_addr){
                iter->addr = (void*)ptr_addr;
                iter->size += size;

                #ifdef MLLM_ALLOCATOR_DEBUG
                debug_free_blocks[block_addr] = iter->size;
                #endif

                // check if we can merge previous free block
                if (iter != free_blocks_.begin()){
                    auto prev = std::prev(iter);
                    if ((uint64_t)prev->addr + prev->size == ptr_addr){
                        prev->size += iter->size;
                        n_free_blocks_--;
                        free_blocks_.erase(prev);
                        #ifdef MLLM_ALLOCATOR_DEBUG
                        debug_free_blocks.erase((uint64_t)prev->addr);
                        #endif
                    }
                }
                return;
            }
        }

        // otherwise, create a new block
        bool inserted = false;
        for(auto iter = free_blocks_.begin();iter!=free_blocks_.end();iter++){
            if((uint64_t)iter->addr >= ptr_addr){
                free_blocks_.insert(iter,FreeBlock{ptr,size});
                inserted = true;
                break;
            }
        }
        if (!inserted){
            free_blocks_.emplace_back(ptr,size);
        }
        #ifdef MLLM_ALLOCATOR_DEBUG
        debug_free_blocks[(uint64_t)ptr] = size;
        #endif
        n_free_blocks_++;
    }
    MemoryPoolManager::~MemoryPoolManager(){
        free(data_);
    }
    #ifdef MLLM_ALLOCATOR_DEBUG
    void MemoryPoolManager::display(){
        // show all blocks in the pool
        std::cout<<"n_free_blocks: "<<n_free_blocks_<<std::endl;
        for (auto iter = free_blocks_.begin();iter != free_blocks_.end();iter++){
            std::cout<<"addr: "<<iter->addr<<" size: "<<iter->size<<std::endl;
        }
    }
    #endif

}