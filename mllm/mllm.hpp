// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <vector>         // IWYU pragma: export
#include <cstdint>        // IWYU pragma: export
#include <algorithm>      // IWYU pragma: export
#include <unordered_map>  // IWYU pragma: export
#include <csignal>

#include <fmt/core.h>
#include <fmt/format.h>

// The headfile will be used in mllm.inl Do not be confused with clang's fixes
#include "mllm/backends/cpu/CPUDispatcher.hpp"  // IWYU pragma: export
#include "mllm/compile/ir/IRPrinter.hpp"        // IWYU pragma: export
#include "mllm/compile/ir/Node.hpp"             // IWYU pragma: export
#include "mllm/core/DataTypes.hpp"              // IWYU pragma: export
#include "mllm/core/DeviceTypes.hpp"            // IWYU pragma: export
#include "mllm/core/ParameterFile.hpp"          // IWYU pragma: export
#include "mllm/core/Tensor.hpp"                 // IWYU pragma: export
#include "mllm/engine/Context.hpp"              // IWYU pragma: export
#include "mllm/engine/MemoryManager.hpp"        // IWYU pragma: export
#include "mllm/engine/SessionTCB.hpp"           // IWYU pragma: export
#include "mllm/engine/Task.hpp"                 // IWYU pragma: export
#include "mllm/utils/Argparse.hpp"              // IWYU pragma: export
#include "mllm/nn/Nn.hpp"                       // IWYU pragma: export
#include "mllm/engine/ConfigFile.hpp"           // IWYU pragma: export

namespace mllm::test {

struct AllCloseResult {
  bool is_close = false;
  size_t total_elements = 0;
  size_t mismatched_elements = 0;
  float max_absolute_diff = 0.0f;
  float max_relative_diff = 0.0f;

  explicit inline operator bool() const noexcept { return is_close; }
};

template<typename T>
void __allCloseProcessIntegerType(const T* a_ptr, const T* b_ptr, size_t numel, AllCloseResult& result, float rtol,
                                  float atol) {
  for (size_t i = 0; i < numel; ++i) {
    T a_val = a_ptr[i];
    T b_val = b_ptr[i];

    double abs_diff = std::abs(static_cast<double>(a_val) - static_cast<double>(b_val));
    double rel_diff = abs_diff / (std::abs(static_cast<double>(b_val)) + 1e-12);

    result.max_absolute_diff = std::max(result.max_absolute_diff, static_cast<float>(abs_diff));
    result.max_relative_diff = std::max(result.max_relative_diff, static_cast<float>(rel_diff));

    if (abs_diff > (atol + rtol * std::abs(static_cast<double>(b_val)))) { result.mismatched_elements++; }
  }
}

/**
 * @brief |a - b| <= atol + rtol * |b|
 *
 * @param a
 * @param b
 * @param rtol
 * @param atol
 * @param equal_nan
 * @return AllCloseResult
 */
AllCloseResult allClose(const Tensor& a, const Tensor& b, float rtol = 1e-5, float atol = 1e-5, bool equal_nan = false);

}  // namespace mllm::test

namespace mllm::async {

template<typename __Module, typename... __Args>
std::pair<TaskResult::sender_t, Task::ptr_t> fork(__Module& module, __Args&&... args) {
  std::vector<Tensor> tensors;
  std::vector<AnyValue> others;

  (..., [&] {
    // The type must can be inference in compile time
    using CleanType = std::decay_t<decltype(args)>;
    if constexpr (std::is_convertible_v<CleanType, Tensor>) {
      tensors.push_back(std::forward<__Args>(args));
    } else if constexpr (std::is_convertible_v<CleanType, AnyValue>) {
      others.push_back(std::forward<__Args>(args));
    } else {
      static_assert(false, "Unsupported argument type!");
    }
  }());
  auto task = std::make_shared<Task>();
  task->type = TaskTypes::kExecuteModule;
  task->inputs = tensors;
  task->args = others;
  task->custom_context_ptr = &module;
  auto& ctx = Context::instance();
  return {ctx.dispatcherManager()->asyncSubmit(module.impl()->getDevice(), task), task};
}

std::vector<Tensor> wait(std::pair<TaskResult::sender_t, Task::ptr_t>& sender);

template<typename... __Args>
std::array<std::vector<Tensor>, sizeof...(__Args)> wait(__Args&&... args) {
  // Phase 1: Wait for all tasks to complete concurrently.
  // We use `when_all` to create a single sender that completes only when all
  // individual task senders have completed.
  auto when_all_sender = stdexec::when_all(std::move(std::forward<__Args>(args).first)...);

  // `sync_wait` blocks the current thread until `when_all_sender` is done.
  // We are only interested in its side-effect of synchronization. The value it
  // returns (likely void or an empty tuple) is not used.
  // We still check the optional to detect execution errors.
  auto result_opt = stdexec::sync_wait(std::move(when_all_sender));
  if (!result_opt) { throw std::runtime_error("Waiting on tasks failed during execution."); }

  // Phase 2: Aggregate results from the task objects.
  // Now that we know all tasks have finished, the `outputs` member of each task
  // object is guaranteed to be populated.
  // We use a pack expansion on the second element of each pair (`.second`, the Task::ptr_t)
  // to access its `outputs` member. This creates a comma-separated list of `std::vector<Tensor>`.
  // This list is then used to directly initialize the returned std::array.
  return {std::forward<__Args>(args).second->outputs...};
}

}  // namespace mllm::async

// The inline file should be included at the last of all head
#include "mllm/mllm.inl"

// The host device backend must be included.
#include "mllm/backends/cpu/CPUBackend.hpp"  // IWYU pragma: export

namespace mllm {

inline void initializeContext() {
  auto& ctx = Context::instance();

  // 1. Register host backend
  auto host_backend = cpu::createCPUBackend();
  ctx.registerBackend(host_backend);

  // 2. Initialize memory manager
  ctx.memoryManager()->registerAllocator(kCPU, host_backend->allocator(), MemoryManagerOptions());

  // 3. Initialize dispatcher manager
  ctx.dispatcherManager()->registerDispatcher(
      cpu::createCPUDispatcher(ctx.dispatcherManager()->getExecutor(), cpu::CPUDispatcherOptions()));
}

void shutdownContext();

void setRandomSeed(uint64_t seed);

void setMaximumNumThreads(uint32_t num_threads);

void memoryReport();

bool isOpenCLAvailable();

bool isQnnAvailable();

void cleanThisThread();

SessionTCB::ptr_t thisThread();

ParameterFile::ptr_t load(const std::string& file_name, ModelFileVersion version = ModelFileVersion::kV1,
                          DeviceTypes map_2_device = kCPU);

void save(const std::string& file_name, const ParameterFile::ptr_t& parameter_file,
          ModelFileVersion version = ModelFileVersion::kV1, DeviceTypes map_2_device = kCPU);

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

inline void __signal_handler(int signal) {
  ::mllm::print("Received signal ", signal);
  ::mllm::shutdownContext();
  exit(signal);
}

inline void __setup_signal_handler() {
  std::signal(SIGINT, __signal_handler);
  std::signal(SIGTERM, __signal_handler);
  std::signal(SIGABRT, __signal_handler);
  std::signal(SIGSEGV, __signal_handler);
}

template<typename Func>
inline int __mllm_exception_main(Func&& func) {
  try {
    return func();
  } catch (const std::exception& e) {
    ::mllm::print("Exception caught: ", e.what());
    ::mllm::shutdownContext();
    return 1;
  } catch (...) {
    ::mllm::print("Caught unknown exception");
    ::mllm::shutdownContext();
    return 1;
  }
}

}  // namespace mllm

#define MLLM_MAIN(x)                                       \
  int main(int argc, char** argv) {                        \
    ::mllm::__setup_signal_handler();                      \
    ::mllm::initializeContext();                           \
    auto user_main = [&]() -> int {                        \
      x;                                                   \
      return 0;                                            \
    };                                                     \
    int result = ::mllm::__mllm_exception_main(user_main); \
    ::mllm::shutdownContext();                             \
    return result;                                         \
  }

#ifdef MLLM_PERFETTO_ENABLE
#include "mllm/engine/Perf.hpp"  // IWYU pragma: export
#endif
