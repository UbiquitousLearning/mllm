
#include "MemoryManager.hpp"

namespace mllm {

// HostMemory::HostMemory()
// : host_ptr_(nullptr), size_(0){}

// HostMemory::HostMemory(size_t size)
// : host_ptr_(nullptr), size_(size){}

// HostMemory::~HostMemory() {
//     if (host_ptr_ ) {
//         mllmFreeHost(host_ptr_);
//     }
// }

// void HostMemory::to_cpu() {
//     mllmMallocHost(&host_ptr_, size_);
//     mllmMemset(host_ptr_, 0, size_);
// }

// void HostMemory::set_cpu_data(void *data) {
//     CHECK(data);
//     if (own_cpu_data_) {
//         mllmFreeHost(host_ptr_);
//     }
//     host_ptr_ = data;
//     own_cpu_data_ = false;//外部的数据，不是自己的指针，所以是false
// }

// const void *HostMemory::cpu_data() {
//     to_cpu();
//     return (const void*)host_ptr_;
// }

MemoryManager::MemoryManager() {
}

} // namespace mllm