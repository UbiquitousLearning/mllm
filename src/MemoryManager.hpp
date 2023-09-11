

#ifndef MLLM_MEMORY_H
#define MLLM_MEMORY_H

// #include "common.h"

#include <algorithm>
#include<string.h>
#include <string>
#include <vector>
#include <iostream>
#include <string>  
#include <iostream> 
#include <memory>
#include <sstream>
using std::vector;
using std::string;
using std::shared_ptr;
using std::ostringstream;

#include "Check.hpp"
//TODO:  aliganed_malloc

namespace mllm {
    // 这是一个功能和malloc/free相同的分配/释放内存/显存的函数。
    /** 参考MNN,NCNN
    static inline void **alignPointer(void **ptr, size_t alignment) {
        return (void **)((intptr_t)((unsigned char *)ptr + alignment - 1) & -alignment);
    }

    extern "C" void *MNNMemoryAllocAlign(size_t size, size_t alignment) {
        MNN_ASSERT(size > 0);

    #ifdef MNN_DEBUG_MEMORY
        return malloc(size);
    #else
        void **origin = (void **)malloc(size + sizeof(void *) + alignment);
        MNN_ASSERT(origin != NULL);
        if (!origin) {
            return NULL;
        }

        void **aligned = alignPointer(origin + 1, alignment);
        aligned[-1]    = origin;
        return aligned;
    #endif
    }

    extern "C" void MNNMemoryFreeAlign(void *aligned) {
    #ifdef MNN_DEBUG_MEMORY
        free(aligned);
    #else
        if (aligned) {
            void *origin = ((void **)aligned)[-1];
            free(origin);
        }
    #endif
    }
    */
    inline void mllmMallocHost(void** ptr, size_t size){ //TODO: 改名为mllmMallocAlign?希望保留Host? 实现内存对齐，参考上面的注释
        *ptr = malloc(size);
        // CHECK(*ptr) << "host allocation of size " << size << " failed";
    }
    inline void mllmMemset(void *X, const int alpha, const size_t N) {
        memset(X, alpha, N);
    }
    inline void mllmFreeHost(void* ptr){
        free(ptr);
    }

    /**
     * CPU内存  ABANDEN OR接入MemoryManager? ... 
    */
    class HostMemory {
    public:
        HostMemory(): cpu_ptr_(nullptr), size_(0){};
        explicit HostMemory(size_t size): cpu_ptr_(nullptr), size_(size){};
        ~HostMemory();
        // enum MemoryState{UNINIT, CPUSTATE, GPUSTATE, SYNCED};
        // MemoryState state() { return state_; }
        size_t size() const { return size_; }
        void set_cpu_data(void* data);
        const void* cpu_data(); //当你想读取数据的时候请使用cpu_data （X） malloc here.

    private:
        size_t size_; //数据大小
        // MemoryState state_; //数据状态，有四种：UNINIT, CPUSTATE, GPUSTATE, SYNCED
        void *cpu_ptr_;//another name: host_ptr
        void to_cpu(); // malloc here. 
        bool own_cpu_data_;

    };

    
    /**
     * 内存管理类 mem pool ... TODO 管理HostMemory
    */
    class MemoryManager {
    public:
        MemoryManager();
        ~MemoryManager();
    };

    
}
#endif //MLLM_MEMORY_H
