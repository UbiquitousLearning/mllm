#include "memory/SystemMemoryManager.hpp"
#include <cassert>
#include <cstddef>
#include <cstdio>

using mllm::SystemMemoryManager;

void test_batch_allocate(size_t alloc_times,size_t size,size_t alignment){
    SystemMemoryManager manager;
    void* pointers[alloc_times];
    for (int i=0;i<alloc_times;i++){
        manager.alloc(&pointers[i], size, alignment);
        assert(pointers[i] != nullptr);
        assert(((long long)pointers[i] & (alignment - 1)) == 0);
    }

    for (int i=0;i<alloc_times;i++){
        manager.free(pointers[i]);
    }
    printf("pass test_batch_allocate,size=%ld,times=%ld,alignment=%ld\n",size,alloc_times,alignment);

}

int main(){
    void* ptr ;

    /**
    * test allocate
    */
    
    test_batch_allocate(100,256,16);
    test_batch_allocate(100,256,4);
    test_batch_allocate(100,256,8);
    test_batch_allocate(100,256,32);

    test_batch_allocate(100,257,8);
    test_batch_allocate(100,257,16);
    test_batch_allocate(100,257,4);
    test_batch_allocate(100,257,8);
    test_batch_allocate(100,257,32);

    return 0;
}