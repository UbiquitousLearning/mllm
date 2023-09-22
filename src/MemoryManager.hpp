

#ifndef MLLM_MEMORY_H
#define MLLM_MEMORY_H

// #include "common.h"

// #include <algorithm>
#include <string.h>
#include <vector>
#include <iostream>
#include <memory>
#include <sstream>
#include <map>
#include <unordered_map>
#include "Types.hpp"

using std::vector;
using std::string;
using std::shared_ptr;
using std::ostringstream;
using std::unordered_map;

#include "Check.hpp"
// TODO:  aliganed_malloc

namespace mllm {
// 这是一个功能和malloc/free相同的分配/释放内存/显存的函数。

/**
 * 内存管理类 mem pool ... TODO 管理HostMemory
 */
class MemoryManager {
public:
    MemoryManager(){}
    virtual ~MemoryManager(){}

    virtual void Alloc(void **ptr, size_t size,size_t alignment) = 0;

    virtual void Free(void **ptr) = 0;

};

class SystemMemoryManager : public MemoryManager {
public:
    SystemMemoryManager(){}
    ~SystemMemoryManager(){}

    void Alloc(void **ptr, size_t size,size_t alignment) override ;

    void Free(void **ptr) override;

};

} // namespace mllm
#endif // MLLM_MEMORY_H
