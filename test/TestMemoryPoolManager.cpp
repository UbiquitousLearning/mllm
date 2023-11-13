#include "memory/MemoryPoolManager.hpp"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <iostream>

using namespace std;
#define N_BLOCKS 4095

void check_alignment(void* ptr,size_t alignment){
    assert(((uint64_t)ptr & (alignment - 1)) == 0);
}

void batch_allocate_and_batch_free(size_t pool_size,size_t block_size,size_t alignment){
    printf("batch_allocate_and_batch_free(%ld,%ld,%ld)\n",pool_size,block_size,alignment);
    mllm::MemoryPoolManager manager(pool_size,16);
    block_size = mllm::aligned_offset(block_size,alignment);
    auto n_blocks = pool_size / block_size;
    auto largest_block = pool_size;
    assert(n_blocks > 0);
    vector<void*> ptrs(n_blocks);

    for (int i=0;i<n_blocks;i++){
        manager.alloc(&ptrs[i],block_size,alignment);
        check_alignment(ptrs[i],alignment);
        // write the data to the memory
        memset(ptrs[i],i,block_size);
        assert(manager.debug_allocate_blocks.size() == i+1);
        largest_block -= block_size;
        for(auto it = manager.debug_free_blocks.begin();it != manager.debug_free_blocks.end();it++){
            auto x = it->second;
            assert(x == largest_block);
        }
      
    }
    assert(manager.debug_free_blocks.empty());
    // free all pointers
    for (int i=0;i<n_blocks;i++){
        manager.free(ptrs[i]);
    }

    // auto free here
    printf("batch_allocate_and_batch_free(%ld,%ld,%ld)\n",pool_size,block_size,alignment);
}

void batch_allocate_but_odd_free(size_t pool_size,size_t block_size,size_t alignment){
    printf("batch_allocate_but_odd_free(%ld,%ld,%ld)\n",pool_size,block_size,alignment);
    mllm::MemoryPoolManager manager(pool_size,16);
    block_size = mllm::aligned_offset(block_size,alignment);
    auto n_blocks = pool_size / block_size;
    auto largest_block = pool_size;
    assert(n_blocks > 0);
    vector<void*> ptrs(n_blocks);
    
    for (int i=0;i<n_blocks;i++){
        manager.alloc(&ptrs[i],block_size,alignment);
        check_alignment(ptrs[i],alignment);
        // write the data to the memory
        memset(ptrs[i],i,block_size);
    }
    assert(manager.debug_free_blocks.empty());

    largest_block = 0;
    auto counter = 0;
    for (int i=0;i<n_blocks;i++){
        if (i%2 ==0){
            manager.free(ptrs[i]);
            counter++;
            assert(counter == manager.debug_free_blocks.size());
        }
    }
    for (int i=0;i<n_blocks;i++){
        if (i%2 ==1){
            manager.free(ptrs[i]);
            counter--;
            if (i!=n_blocks - 1){
                assert(counter == manager.debug_free_blocks.size());
            }else{
                assert(manager.debug_free_blocks.size() == 1);
                assert(manager.debug_free_blocks.begin()->second == pool_size);
            }

        }
    }
    printf("pass batch_allocate_but_odd_free(%ld,%ld,%ld)\n",pool_size,block_size,alignment);
}

int main(){
    batch_allocate_and_batch_free(1024*1024,256,16);
    // batch_allocate_and_batch_free(1024*1024,257,16);
    batch_allocate_but_odd_free(1024*1024,256,16);
    
    return 0;
}