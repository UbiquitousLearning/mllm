// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <cstdint>

namespace mllm {

// Platform flags
enum class Platform : int32_t {
  UNKNOWN_PLATFORM = 0,
  WINDOWS_PLATFORM = 1,
  LINUX_PLATFORM = 2,
  ANDROID_PLATFORM = 3,
  MACOS_PLATFORM = 4,
  IOS_PLATFORM = 5
};

// Platform detection using preprocessor macros
#if defined(__ANDROID__)
constexpr Platform CURRENT_PLATFORM = Platform::ANDROID_PLATFORM;
constexpr const char* MLLM_CURRENT_PLATFORM_STRING = "android";
#define MLLM_HOST_PLATFORM_ANDROID 1
#elif defined(_WIN32) || defined(_WIN64)
constexpr Platform CURRENT_PLATFORM = Platform::WINDOWS_PLATFORM;
constexpr const char* MLLM_CURRENT_PLATFORM_STRING = "windows";
#define MLLM_HOST_PLATFORM_WINDOWS 1
#elif defined(__linux__)
constexpr Platform CURRENT_PLATFORM = Platform::LINUX_PLATFORM;
constexpr const char* MLLM_CURRENT_PLATFORM_STRING = "linux";
#define MLLM_HOST_PLATFORM_LINUX 1
#elif defined(__APPLE__)
#include <TargetConditionals.h>
#if TARGET_OS_IPHONE
constexpr Platform CURRENT_PLATFORM = Platform::IOS_PLATFORM;
constexpr const char* MLLM_CURRENT_PLATFORM_STRING = "ios";
#define MLLM_HOST_PLATFORM_IOS 1
#elif TARGET_OS_MAC
constexpr Platform CURRENT_PLATFORM = Platform::MACOS_PLATFORM;
constexpr const char* MLLM_CURRENT_PLATFORM_STRING = "macos";
#define MLLM_HOST_PLATFORM_MACOS 1
#else
constexpr Platform CURRENT_PLATFORM = Platform::UNKNOWN_PLATFORM;
constexpr const char* MLLM_CURRENT_PLATFORM_STRING = "unknown";
#define MLLM_HOST_PLATFORM_UNKNOWN 1
#endif
#else
constexpr Platform CURRENT_PLATFORM = Platform::UNKNOWN_PLATFORM;
constexpr const char* MLLM_CURRENT_PLATFORM_STRING = "unknown";
#define MLLM_HOST_PLATFORM_UNKNOWN 1
#endif

// Helper functions to check specific platforms
constexpr bool isWindows() { return CURRENT_PLATFORM == Platform::WINDOWS_PLATFORM; }

constexpr bool isLinux() { return CURRENT_PLATFORM == Platform::LINUX_PLATFORM; }

constexpr bool isAndroid() { return CURRENT_PLATFORM == Platform::ANDROID_PLATFORM; }

constexpr bool isMacOS() { return CURRENT_PLATFORM == Platform::MACOS_PLATFORM; }

constexpr bool isIOS() { return CURRENT_PLATFORM == Platform::IOS_PLATFORM; }

constexpr bool isUnknownPlatform() { return CURRENT_PLATFORM == Platform::UNKNOWN_PLATFORM; }
}  // namespace mllm