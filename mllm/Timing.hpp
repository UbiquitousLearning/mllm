//
// Created by Rongjie Yi on 23-11-8.
//

#ifndef MLLM_TIMING_HPP
#define MLLM_TIMING_HPP

#include <chrono>

namespace mllm {

inline void mllm_time_init(void) {}
inline int64_t mllm_time_ms(void) {
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch());
    return ms.count();
}

inline int64_t mllm_time_us(void) {
    auto now = std::chrono::system_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch());
    return us.count();
}

}

#endif // MLLM_TIMING_HPP
