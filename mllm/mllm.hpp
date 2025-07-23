/**
 * @file mllm.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-22
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <vector>
#include <cstdint>

#include <fmt/core.h>
#include <fmt/format.h>

#include "mllm/core/DataTypes.hpp"
#include "mllm/core/DeviceTypes.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/engine/Context.hpp"

//===----------------------------------------------------------------------===//
// Print Stuff
//===----------------------------------------------------------------------===//
namespace fmt {
template<>
struct formatter<mllm::DataTypes> {
  constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }
  template<typename FormatContext>
  auto format(const mllm::DataTypes& dtype, FormatContext& ctx) const {
    auto out = ctx.out();
    // TODO
    return out;
  }
};

template<>
struct formatter<mllm::DeviceTypes> {
  constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }
  template<typename FormatContext>
  auto format(const mllm::DeviceTypes& device, FormatContext& ctx) const {
    auto out = ctx.out();
    out = fmt::format_to(out, "{}", mllm::deviceTypes2Str(device));
    return out;
  }
};

template<>
struct formatter<mllm::Tensor> {
  constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }
  template<typename FormatContext>
  auto format(const mllm::Tensor& tensor, FormatContext& ctx) const {
    auto out = ctx.out();
    // TODO
    return out;
  }
};

template<>
struct formatter<std::vector<int32_t>> {
  constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }
  template<typename FormatContext>
  auto format(const std::vector<int32_t>& vec, FormatContext& ctx) const {
    auto out = ctx.out();
    *out++ = '[';
    for (size_t i = 0; i < vec.size(); ++i) {
      if (i > 0) {
        *out++ = ',';
        *out++ = ' ';
      }
      out = fmt::format_to(out, "{}", vec[i]);
    }
    *out++ = ']';
    return out;
  }
};
}  // namespace fmt

#if defined(__aarch64__)
#include "mllm/backends/arm/ArmBackend.hpp"
#define __MLLM_HOST_BACKEND_CREATE arm::createArmBackend()
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include "mllm/backends/x86/X86Backend.hpp"
#define __MLLM_HOST_BACKEND_CREATE x86::createX86Backend()
#endif

namespace mllm {

inline void initializeContext() {
  auto& ctx = Context::instance();

  // 1. Register host backend
  auto host_backend = __MLLM_HOST_BACKEND_CREATE;
  ctx.registerBackend(host_backend);

  // 2. Initialize memory manager
  ctx.memoryManager()->registerAllocator(kCPU, host_backend->allocator(), MemoryManagerOptions());
}

void shutdownContext();

void setRandomSeed(uint32_t seed);

void setMaximumNumThreads(uint32_t num_threads);

void memoryReport();

bool isOpenCLAvailable();

bool isQnnAvailable();

//===----------------------------------------------------------------------===//
// Print Stuff
//===----------------------------------------------------------------------===//
// The iron armor of C++, a weary soul's refrain,
// Through endless loops and templates, a world of silent pain.
// Life, a fleeting moment, whispers, "Python's ease I crave,"
// A single line of "print" to pull me from the grave.
// No grand designs I seek, no fame in compiled art,
// Just one clean build to soothe a coder's aching heart.
// Let this sweet sugar shine, a beacon in the night,
// And banish debugging's darkness with a single ray of light.
template<typename... Args>
inline void print(const Args&... args) {
  (fmt::print("{} ", args), ...);
  fmt::print("\n");
}

}  // namespace mllm