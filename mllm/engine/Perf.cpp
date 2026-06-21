// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/engine/Perf.hpp"

// The implementation is also guarded by the preprocessor flag. When disabled,
// this file compiles to a set of empty functions.
#ifdef MLLM_PERFETTO_ENABLE

#include <fstream>
#include <memory>
#include <vector>

// 2. Instantiate the static storage required by the track event system.
PERFETTO_TRACK_EVENT_STATIC_STORAGE();

namespace mllm::perf {

namespace {
// A global pointer to hold the active tracing session.
std::unique_ptr<perfetto::TracingSession> g_tracing_session;
}  // namespace

void start() {
  perfetto::TracingInitArgs args;
  // Use the in-process backend so the application can control tracing
  // without needing an external Perfetto daemon.
  args.backends = perfetto::kInProcessBackend;
  perfetto::Tracing::Initialize(args);

  // Register our application as a producer of "track_event" data.
  perfetto::TrackEvent::Register();

  // --- Configure and Start a Tracing Session ---
  perfetto::TraceConfig cfg;
  // Use a reasonably large buffer (e.g., 32MB) for detailed traces.
  cfg.add_buffers()->set_size_kb(32768);

  auto* ds_cfg = cfg.add_data_sources()->mutable_config();
  ds_cfg->set_name("track_event");  // Enable the track_event data source.

  g_tracing_session = perfetto::Tracing::NewTrace();
  g_tracing_session->Setup(cfg);
  g_tracing_session->StartBlocking();  // Block until the session is ready.
}

void stop() {
  if (!g_tracing_session) {
    return;  // Do nothing if tracing was never initialized.
  }

  // Stop the session, flushing all buffered data.
  g_tracing_session->StopBlocking();
}

void saveReport(const std::string& file_path) {
  // Read the binary trace data from the session.
  std::vector<char> trace_data = g_tracing_session->ReadTraceBlocking();

  // Write the trace data to a file. This file can be opened in ui.perfetto.dev.
  std::ofstream output_file(file_path, std::ios::out | std::ios::binary);
  if (output_file) {
    output_file.write(trace_data.data(), trace_data.size());
    output_file.close();
  }

  // Clean up the session object.
  g_tracing_session.reset();
}

}  // namespace mllm::perf

#else  // MLLM_PERFETTO_ENABLE is not defined or 0

// Provide empty function implementations when tracing is disabled.
// This ensures that the application links successfully without any changes.
namespace mllm {
namespace perf {

void start() { /* No-op */ }

void stop() { /* No-op */ }

void saveReport(const std::string& file_path) { /* No-op */ }

}  // namespace perf
}  // namespace mllm

#endif  // MLLM_PERFETTO_ENABLE
