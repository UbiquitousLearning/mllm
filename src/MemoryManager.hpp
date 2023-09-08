

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
    // 这是一个功能和malloc/free相同的分配/释放内存/显存的函数。如果使用了GPU，则在在GPU上分配和释放，否则在内存上分配和释放。
    inline void mllmMallocHost(void** ptr, size_t size){ //TODO: 改名为mllmMallocAlign 实现内存对其
        *ptr = malloc(size);
        // CHECK(*ptr) << "host allocation of size " << size << " failed";
    }
    inline void mllmMemset(void *X, const int alpha, const size_t N) {
        memset(X, alpha, N);
    }
    inline void mllmFreeHost(void* ptr){
        free(ptr);
    }

    class AlignedMemory {
    public:
        AlignedMemory();
        explicit AlignedMemory(size_t size);
        ~AlignedMemory();
        enum MemoryState{UNINIT, CPUSTATE, GPUSTATE, SYNCED};
        // MemoryState state() { return state_; }
        size_t size() const { return size_; }
        void set_cpu_data(void* data);
        const void* cpu_data(); //当你想读取数据的时候请使用cpu_data

    private:
        size_t size_; //数据大小
        // MemoryState state_; //数据状态，有四种：UNINIT, CPUSTATE, GPUSTATE, SYNCED
        void *cpu_ptr_;
        void to_cpu();
        bool own_cpu_data_;

    };
}
#endif //MLLM_MEMORY_H
