/**
 * @file Logger.hpp
 * @author Chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief logger. wrap on fmt
 * @version 0.1
 * @date 2024-10-03
 *
 * @copyright Copyright (c) 2024
 *
 */
#pragma once

#include "fmt/base.h"
#include "fmt/core.h"
#include "fmt/color.h"

namespace mllm::xnnpack {

class Log {
public:
    enum LogLevel {
        INFO = 0,
        WARN = 1,
        ERROR = 2,
    };

    static inline LogLevel log_level = INFO;

    template <typename... Args>
    static inline void info(Args &&...args) {
        switch (Log::log_level) {
        case INFO:
            fmt::print(fmt::emphasis::bold, "<I> ");
            fmt::println(std::forward<Args>(args)...);
            break;
        case WARN:
        case ERROR: break;
        }
    }

    template <typename... Args>
    static inline void warn(Args &&...args) {
        switch (Log::log_level) {
        case INFO:
        case WARN:
            fmt::print(fg(fmt::color::yellow_green) | fmt::emphasis::bold, "<W> ");
            fmt::println(std::forward<Args>(args)...);
            break;
        case ERROR: break;
        }
    }

    template <typename... Args>
    static inline void error(Args &&...args) {
        switch (Log::log_level) {
        case INFO:
        case WARN:
        case ERROR:
            fmt::print(fg(fmt::color::red) | fmt::emphasis::bold, "<E> ");
            fmt::println(std::forward<Args>(args)...);
            break;
        }
    }
};

} // namespace mllm::xnnpack