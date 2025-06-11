/**
 * @file Metrics.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-11
 *
 * @copyright Copyright (c) 2025
 *
 */

#include <cstdint>
#include <fstream>

inline float powerNow() {
#if defined(__ANDROID__)
    std::ifstream voltage_file("/sys/class/power_supply/battery/voltage_now");
    std::ifstream current_file("/sys/class/power_supply/battery/current_now");

    int64_t voltage = 0, current = 0;

    if (voltage_file.good() && voltage_file.is_open()) { voltage_file >> voltage; }
    if (current_file.good() && current_file.is_open()) { current_file >> current; }

    float power_w = (static_cast<float>(voltage) / 1e6f) * (static_cast<float>(current) / 1e6f);
    return power_w;
#elif defined(__linux__)
// TODO
#else
// TODO
#endif
    return 0.f;
}