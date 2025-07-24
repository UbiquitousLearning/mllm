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
#include <algorithm>
#include <unordered_map>

#include <fmt/core.h>
#include <fmt/format.h>

#include "mllm/core/DataTypes.hpp"
#include "mllm/core/DeviceTypes.hpp"
#include "mllm/core/ParameterFile.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/engine/Context.hpp"
#include "mllm/engine/SessionTCB.hpp"

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

template<>
struct formatter<mllm::ParameterFile::ptr_t> {
  constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }
  template<typename FormatContext>
  auto format(const mllm::ParameterFile::ptr_t& params, FormatContext& ctx) const {
    if (!params) { return fmt::format_to(ctx.out(), "ParameterFile[nullptr]"); }

    static std::unordered_map<mllm::ModelFileVersion, std::string> version_names = {
        {mllm::ModelFileVersion::kUserTemporary, "UserTemporary"},
        {mllm::ModelFileVersion::kV1, "V1"},
        {mllm::ModelFileVersion::kV2, "V2"},
    };
    std::string version_str = "Unknown";
    if (auto it = version_names.find(params->version()); it != version_names.end()) {
      version_str = it->second;
    } else {
      version_str = fmt::format("{}", static_cast<int>(params->version()));
    }

    struct ParamInfo {
      std::string name;
      std::string shape;
      std::string dtype;
    };

    std::vector<ParamInfo> param_infos;
    for (auto it = params->begin(); it != params->end(); ++it) {
      auto tensor = params->pull(it->first);
      param_infos.push_back({it->first, fmt::format("{}", tensor.shape()), fmt::format("{}", tensor.dtype())});
    }

    std::sort(param_infos.begin(), param_infos.end(), [](const ParamInfo& a, const ParamInfo& b) { return a.name < b.name; });

    const size_t MAX_SHOW = 2048;
    const size_t total_params = param_infos.size();
    auto out = fmt::format_to(ctx.out(), "ParameterFile[{} params, version={}]:\n", total_params, version_str);

    for (size_t i = 0; i < std::min(MAX_SHOW, total_params); ++i) {
      const auto& info = param_infos[i];
      out = fmt::format_to(out, "  {:20} {:20} {:>10}\n", info.name, info.shape, info.dtype);
    }

    if (total_params > MAX_SHOW) {
      out = fmt::format_to(out, "  ... and {} more parameters\n", total_params - MAX_SHOW);
    } else if (total_params == 0) {
      out = fmt::format_to(out, "  [No parameters]\n");
    }
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

SessionTCB::ptr_t thisThread();

ParameterFile::ptr_t load(const std::string& file_name, DeviceTypes map_2_device = kCPU);

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