// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <fmt/color.h>
#include <fmt/core.h>

namespace mllm {

enum class LogLevel {  // NOLINT
  kInfo = 0,
  kWarn = 1,
  kError = 2,
  kFatal = 3,
  kAssert = 4,
};

class Logger {
 public:
  static LogLevel& level() { return level_; }

 private:
  static LogLevel level_;
};

#define MLLM_INFO(...)                                                       \
  if (mllm::Logger::level() <= mllm::LogLevel::kInfo) {                      \
    fmt::print(fg(fmt::color::green) | fmt::emphasis::bold, "[INFO]");       \
    fmt::print(" {}:{} {}\n", __FILE__, __LINE__, fmt::format(__VA_ARGS__)); \
  }

#define MLLM_WARN(...)                                                        \
  if (mllm::Logger::level() <= mllm::LogLevel::kWarn) {                       \
    fmt::print(fg(fmt::color::green_yellow) | fmt::emphasis::bold, "[WARN]"); \
    fmt::print(" {}:{} {}\n", __FILE__, __LINE__, fmt::format(__VA_ARGS__));  \
  }

#define MLLM_WARN_EXIT(code, ...)                                             \
  if (mllm::Logger::level() <= mllm::LogLevel::kWarn) {                       \
    fmt::print(fg(fmt::color::green_yellow) | fmt::emphasis::bold, "[WARN]"); \
    fmt::print(" {}:{} {}\n", __FILE__, __LINE__, fmt::format(__VA_ARGS__));  \
  }                                                                           \
  exit((int32_t)(code))

#define MLLM_ERROR(...)                                                      \
  if (mllm::Logger::level() <= mllm::LogLevel::kError) {                     \
    fmt::print(fg(fmt::color::red) | fmt::emphasis::bold, "[ERROR]");        \
    fmt::print(" {}:{} {}\n", __FILE__, __LINE__, fmt::format(__VA_ARGS__)); \
  }

#define MLLM_ERROR_EXIT(code, ...)                                           \
  if (mllm::Logger::level() <= mllm::LogLevel::kError) {                     \
    fmt::print(fg(fmt::color::red) | fmt::emphasis::bold, "[ERROR]");        \
    fmt::print(" {}:{} {}\n", __FILE__, __LINE__, fmt::format(__VA_ARGS__)); \
  }                                                                          \
  abort();                                                                   \
  exit((int32_t)(code))

#define MLLM_FATAL_EXIT(code, ...)                                           \
  if (mllm::Logger::level() <= mllm::LogLevel::kFatal) {                     \
    fmt::print(fg(fmt::color::red) | fmt::emphasis::bold, "[FATAL]");        \
    fmt::print(" {}:{} {}\n", __FILE__, __LINE__, fmt::format(__VA_ARGS__)); \
  }                                                                          \
  exit((int32_t)(code))

#define MLLM_ASSERT_EXIT(code, ...)                                          \
  if (mllm::Logger::level() <= mllm::LogLevel::kAssert) {                    \
    fmt::print(fg(fmt::color::red) | fmt::emphasis::bold, "[ASSERT]");       \
    fmt::print(" {}:{} {}\n", __FILE__, __LINE__, fmt::format(__VA_ARGS__)); \
  }                                                                          \
  abort();                                                                   \
  exit((int32_t)(code))

}  // namespace mllm
