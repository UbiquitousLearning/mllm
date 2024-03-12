//
// Created by Rongjie Yi on 23-11-8.
//

#ifndef MLLM_TIMING_HPP
#define MLLM_TIMING_HPP


#if defined(_MSC_VER) || defined(__MINGW32__)
#include <bemapiset.h>
#include  <windows.h>
#else
#include <cstdint>
#include <ctime>
#endif


namespace mllm {


#if defined(_MSC_VER) || defined(__MINGW32__)
static __int64 timer_freq, timer_start;
inline void mllm_time_init(void) {
    LARGE_INTEGER t;
    QueryPerformanceFrequency(&t);
    timer_freq = t.QuadPart;

    // The multiplication by 1000 or 1000000 below can cause an overflow if timer_freq
    // and the uptime is high enough.
    // We subtract the program start time to reduce the likelihood of that happening.
    QueryPerformanceCounter(&t);
    timer_start = t.QuadPart;
}
inline __int64 mllm_time_ms(void) {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return ((t.QuadPart-timer_start) * 1000) / timer_freq;
}
inline __int64 mllm_time_us(void) {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return ((t.QuadPart-timer_start) * 1000000) / timer_freq;
}
#else
inline void mllm_time_init(void) {}
inline int64_t mllm_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec*1000 + (int64_t)ts.tv_nsec/1000000;
}

inline int64_t mllm_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec*1000000 + (int64_t)ts.tv_nsec/1000;
}
#endif

}

#endif // MLLM_TIMING_HPP
