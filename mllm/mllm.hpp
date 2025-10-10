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
#include "mllm/utils/ScopedRedirect.hpp"        // IWYU pragma: export
#include "mllm/utils/Ignore.hpp"                // IWYU pragma: export

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
      static_assert(always_false<CleanType>::value, "Unsupported argument type!");
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

void setLogLevel(const LogLevel& level);

void setRandomSeed(uint64_t seed);

int64_t getRandomState();

void setMaximumNumThreads(uint32_t num_threads);

void setPrintPrecision(int precision);

void setPrintMaxElementsPerDim(int max_elements);

void memoryReport();

bool isOpenCLAvailable();

bool isQnnAvailable();

extern void initQnnBackend();

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

}  // namespace mllm

#include <cstdlib>
#include <cstring>

#if defined(_WIN32) || defined(_WIN64)
#define __MLLM_SIGNAL_WINDOWS 1
#include <windows.h>
#elif defined(__APPLE__)
#define __MLLM_SIGNAL_MACOS 1
#include <execinfo.h>
#include <unistd.h>
#elif defined(__linux__) && !defined(__ANDROID__)
#define __MLLM_SIGNAL_LINUX 1
#include <execinfo.h>
#include <unistd.h>
#elif defined(__ANDROID__)
#define __MLLM_SIGNAL_ANDROID 1
#include <unwind.h>
#include <dlfcn.h>
#include <cxxabi.h>
#endif

namespace mllm {

//===----------------------------------------------------------------------===//
// Signal Handler
//===----------------------------------------------------------------------===//

// Android-specific stack trace implementation
#if defined(__MLLM_SIGNAL_ANDROID)
struct AndroidBacktraceState {
  void** current;
  void** end;
};

static _Unwind_Reason_Code android_unwind_callback(struct _Unwind_Context* context, void* arg) {
  AndroidBacktraceState* state = static_cast<AndroidBacktraceState*>(arg);
  uintptr_t pc = _Unwind_GetIP(context);
  if (pc) {
    if (state->current == state->end) {
      return _URC_END_OF_STACK;
    } else {
      *state->current++ = reinterpret_cast<void*>(pc);  // NOLINT
    }
  }
  return _URC_NO_REASON;
}

static size_t capture_backtrace(void** buffer, size_t max) {
  AndroidBacktraceState state = {.current = buffer, .end = buffer + max};
  _Unwind_Backtrace(android_unwind_callback, &state);
  return state.current - buffer;
}
#endif

inline void safe_write(const char* msg, size_t len) {
#if defined(__MLLM_SIGNAL_WINDOWS)
  HANDLE hStderr = GetStdHandle(STD_ERROR_HANDLE);
  DWORD bytesWritten;
  WriteFile(hStderr, msg, static_cast<DWORD>(len), &bytesWritten, NULL);
#else
  IGNORE(write(STDERR_FILENO, msg, len));
#endif
}

inline const char* signal_description(int signal) {
  switch (signal) {
    case SIGINT: return "SIGINT (Interrupt from keyboard)";
    case SIGTERM: return "SIGTERM (Termination signal)";
    case SIGABRT: return "SIGABRT (Abort signal from abort())";
#if !defined(__MLLM_SIGNAL_WINDOWS)
    case SIGSEGV: return "SIGSEGV (Segmentation violation)";
    case SIGILL: return "SIGILL (Illegal instruction)";
    case SIGFPE: return "SIGFPE (Floating-point exception)";
#endif
    default: return "Unknown signal";
  }
}

inline void print_stack_trace() {
#if defined(__MLLM_SIGNAL_MACOS) || defined(__MLLM_SIGNAL_LINUX)
  void* buffer[100];
  int size = backtrace(buffer, 100);
  safe_write("Stack trace:\n", 13);
  backtrace_symbols_fd(buffer, size, STDERR_FILENO);
#elif defined(__MLLM_SIGNAL_WINDOWS)
  safe_write("Stack trace not available on Windows in signal handler\n", 52);
#elif defined(__MLLM_SIGNAL_ANDROID)
  void* buffer[100];
  const int size = capture_backtrace(buffer, 100);
  safe_write("Stack trace:\n", 13);

  for (int i = 0; i < size; ++i) {
    Dl_info info;
    if (dladdr(buffer[i], &info) && info.dli_sname) {
      char* demangled = abi::__cxa_demangle(info.dli_sname, nullptr, nullptr, nullptr);
      const char* name = demangled ? demangled : info.dli_sname;
      char line[256];
      int len = snprintf(line, sizeof(line), "#%d %p %s\n", i, buffer[i], name);
      safe_write(line, len);
      free(demangled);
    } else {
      char line[256];
      int len = snprintf(line, sizeof(line), "#%d %p\n", i, buffer[i]);
      safe_write(line, len);
    }
  }
#endif
}

inline void __signal_handler(int signal) {
  const char* desc = signal_description(signal);
  safe_write("Error: Received signal ", 22);
  char sig_str[12];
#if defined(__MLLM_SIGNAL_WINDOWS)
  _itoa_s(signal, sig_str, 10);
#else
  snprintf(sig_str, sizeof(sig_str), "%d", signal);
#endif
  safe_write(sig_str, strlen(sig_str));
  safe_write(" - ", 3);
  safe_write(desc, strlen(desc));
  safe_write("\n", 1);
  switch (signal) {
    case SIGSEGV:
      print_stack_trace();
      safe_write("Possible causes: invalid memory access, dangling pointer, stack overflow.\n", 74);
      break;
    case SIGABRT:
      print_stack_trace();
      safe_write("Possible causes: failed assertion, memory corruption, double-free.\n", 68);
      break;
    default: break;
  }
  safe_write("Shutting down...\n", 17);
  mllm::shutdownContext();
#if defined(__MLLM_SIGNAL_WINDOWS)
  _exit(signal);
#else
  ::_exit(signal);
#endif
}

inline void __setup_signal_handler() {
#if defined(__MLLM_SIGNAL_WINDOWS)
  signal(SIGINT, __signal_handler);
  signal(SIGTERM, __signal_handler);
  signal(SIGABRT, __signal_handler);
#else
  struct sigaction sa;
  sa.sa_handler = __signal_handler;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = SA_RESTART;

  sigaction(SIGINT, &sa, nullptr);
  sigaction(SIGTERM, &sa, nullptr);
  sigaction(SIGABRT, &sa, nullptr);
  sigaction(SIGSEGV, &sa, nullptr);
  sigaction(SIGILL, &sa, nullptr);
  sigaction(SIGFPE, &sa, nullptr);
#endif
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

#define MLLM_MAIN(...)                                     \
  int main(int argc, char** argv) {                        \
    ::mllm::__setup_signal_handler();                      \
    ::mllm::initializeContext();                           \
    auto user_main = [&]() -> int {                        \
      __VA_ARGS__;                                         \
      return 0;                                            \
    };                                                     \
    int result = ::mllm::__mllm_exception_main(user_main); \
    ::mllm::shutdownContext();                             \
    return result;                                         \
  }

#ifdef MLLM_PERFETTO_ENABLE
#include "mllm/engine/Perf.hpp"  // IWYU pragma: export
#endif

namespace mllm::perf {

void warmup(const ParameterFile::ptr_t& params);

}
