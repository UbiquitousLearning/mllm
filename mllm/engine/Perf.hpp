// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <string>

#ifdef MLLM_PERFETTO_ENABLE

#include <perfetto.h>

// Declare the tracing categories. These group related events in the Perfetto UI.
// The actual definition of these categories happens in perf.cpp.
PERFETTO_DEFINE_CATEGORIES(
    perfetto::Category("mllm.func_lifecycle")
        .SetDescription("Tracks the lifecycle of functions within the MLLM, such as the start and end of their execution."),
    perfetto::Category("mllm.tensor_lifecycle")
        .SetDescription("Tracks the lifecycle of tensors, including their creation, allocation, usage, and destruction."),
    perfetto::Category("mllm.kernel")
        .SetDescription("Tracks the execution of computational kernels on accelerators like GPUs."),
    perfetto::Category("mllm.ar_step").SetDescription("Auto regressive step"));

// Define wrapper macros for our application. This makes it easy to disable all
// tracing calls from one central place.
#define MLLM_PERF_TRACE_EVENT(...) TRACE_EVENT(__VA_ARGS__)
#define MLLM_PERF_TRACE_COUNTER(...) TRACE_COUNTER(__VA_ARGS__)
#define MLLM_PERF_TRACE_BEGIN(...) TRACE_EVENT_BEGIN(__VA_ARGS__)
#define MLLM_PERF_TRACE_END(...) TRACE_EVENT_END(__VA_ARGS__)

#else
// When tracing is disabled, these macros become empty do-while(0) statements.
// The compiler completely optimizes them away, resulting in zero performance cost.
#define MLLM_PERF_TRACE_EVENT(...) \
  do {                             \
  } while (0)
#define MLLM_PERF_TRACE_COUNTER(...) \
  do {                               \
  } while (0)
#define MLLM_PERF_TRACE_BEGIN(...) \
  do {                             \
  } while (0)
#define MLLM_PERF_TRACE_END(...) \
  do {                           \
  } while (0)

#endif

namespace mllm::perf {

// Initializes the Perfetto tracing system and starts a recording session.
// Call this once at the very beginning of your main() function.
void start();

// Stops the recording session and writes the collected trace data to a file.
// Call this once at the end of your main() function for a clean shutdown.
void stop();

void saveReport(const std::string& file_path);

}  // namespace mllm::perf
