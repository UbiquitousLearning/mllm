#ifndef MLLM_MEMINSPECT_H
#define MLLM_MEMINSPECT_H

#include <cctype>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace mllm {

#define PRINT_MEMORY_USAGE(message) \
    printf("\nMemory Usage: %ld MB(%ld) at: %s\n", physical_memory_used_by_process() / 1000, virtual_memory_used_by_process()/1000, message);

#define TIME_COUNT_START() \
    struct timespec start, end; \
    clock_gettime(CLOCK_MONOTONIC, &start);

#define TIME_COUNT_END() \
    clock_gettime(CLOCK_MONOTONIC, &end); \
    printf("Time: %ld ms", (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1000000);

// 实时获取程序占用的内存，单位：kb。
size_t physical_memory_used_by_process();
size_t virtual_memory_used_by_process();
} // namespace mllm

#endif
