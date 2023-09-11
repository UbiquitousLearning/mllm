
#include "MemoryManager.hpp"

namespace mllm{
    

    // HostMemory::HostMemory()
    // : cpu_ptr_(nullptr), size_(0){}

    // HostMemory::HostMemory(size_t size)
    // : cpu_ptr_(nullptr), size_(size){}

    HostMemory::~HostMemory() {
        if (cpu_ptr_ ) {
            mllmFreeHost(cpu_ptr_);
        }
    }

    void HostMemory::to_cpu() {
        mllmMallocHost(&cpu_ptr_, size_);
        mllmMemset(cpu_ptr_, 0, size_);
    }


    void HostMemory::set_cpu_data(void *data) {
        CHECK(data);
        if (own_cpu_data_) {
            mllmFreeHost(cpu_ptr_);
        }
        cpu_ptr_ = data;
        own_cpu_data_ = false;//外部的数据，不是自己的指针，所以是false
    }

    const void *HostMemory::cpu_data() {
        to_cpu();
        return (const void*)cpu_ptr_;
    }


}