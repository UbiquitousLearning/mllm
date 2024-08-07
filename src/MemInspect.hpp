#ifndef MLLM_MEMINSPECT_H
#define MLLM_MEMINSPECT_H

#include <cctype>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace mllm {

#ifdef DEBUGPRINT
#define PRINT_MEMORY_USAGE(message) \
    printf("\nMemory Usage: %ld MB(%ld) at: %s\n", physical_memory_used_by_process() / 1000, virtual_memory_used_by_process()/1000, message);
#else
#define PRINT_MEMORY_USAGE(message)
#endif

// get memory in kb in unix env
size_t physical_memory_used_by_process();
size_t virtual_memory_used_by_process();
} // namespace mllm

#endif
