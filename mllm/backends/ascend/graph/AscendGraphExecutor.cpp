// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "AscendGraphExecutor.hpp"
#include "mllm/backends/ascend/AscendCommon.hpp"
#include "mllm/utils/Common.hpp"
#include <acl/acl.h>

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>

namespace mllm::ascend {

namespace MLLM_ANONYMOUS_NAMESPACE {

using Clock = std::chrono::high_resolution_clock;

double elapsedMs(const Clock::time_point& start, const Clock::time_point& end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

bool isEnvEnabled(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && value[0] != '0';
}

int getEnvInt(const char* name, int default_value) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') return default_value;
  char* end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (end == value || parsed <= 0) return default_value;
  return static_cast<int>(parsed);
}

bool shouldProfileAscendGraph() {
  return isEnvEnabled("MLLM_PROFILE_ASCEND_GRAPH");
}

int profileAscendGraphEvery() {
  return getEnvInt("MLLM_PROFILE_ASCEND_GRAPH_EVERY", 20);
}

struct GraphProfileStats {
  uint64_t calls{0};
  double setup_ms{0.0};
  double alloc_ms{0.0};
  double execute_ms{0.0};
  double sync_ms{0.0};
  double total_ms{0.0};
};

std::mutex& graphProfileMutex() {
  static std::mutex mutex;
  return mutex;
}

std::unordered_map<std::string, GraphProfileStats>& graphProfileMap() {
  static std::unordered_map<std::string, GraphProfileStats> stats_map;
  return stats_map;
}

void printGraphProfileSummary(const std::string& graph_name, const GraphProfileStats& stats) {
  auto avg = [](double total, uint64_t count) { return count == 0 ? 0.0 : total / static_cast<double>(count); };
  std::cout << std::fixed << std::setprecision(3)
            << "[AscendGraphProfile] graph=" << graph_name
            << " avg_ms(total=" << avg(stats.total_ms, stats.calls)
            << ", setup=" << avg(stats.setup_ms, stats.calls)
            << ", alloc=" << avg(stats.alloc_ms, stats.calls)
            << ", execute=" << avg(stats.execute_ms, stats.calls)
            << ", sync=" << avg(stats.sync_ms, stats.calls)
            << ") calls=" << stats.calls << "\n";
}

void recordGraphProfile(const std::string& graph_name,
                               double setup_ms,
                               double alloc_ms,
                               double execute_ms,
                               double sync_ms,
                               double total_ms) {
  std::lock_guard<std::mutex> lock(graphProfileMutex());
  auto& stats = graphProfileMap()[graph_name];
  stats.calls += 1;
  stats.setup_ms += setup_ms;
  stats.alloc_ms += alloc_ms;
  stats.execute_ms += execute_ms;
  stats.sync_ms += sync_ms;
  stats.total_ms += total_ms;

  const int every = profileAscendGraphEvery();
  if (stats.calls == 1 || (every > 0 && stats.calls % static_cast<uint64_t>(every) == 0)) {
    printGraphProfileSummary(graph_name, stats);
  }
}

}  // namespace MLLM_ANONYMOUS_NAMESPACE

AscendGraphExecutor::AscendGraphExecutor(atb::Operation* graph_op, atb::Context* context)
    : graph_op_(graph_op),
      context_(context),
      workspace_(nullptr),
      workspace_size_(0) {

  if (graph_op_ == nullptr) {
    MLLM_ERROR_EXIT(ExitCode::kAscendError, "Graph operation is null");
  }

  if (context_ == nullptr) {
    MLLM_ERROR_EXIT(ExitCode::kAscendError, "ATB context is null");
  }
}

AscendGraphExecutor::~AscendGraphExecutor() {
  // Free workspace
  if (workspace_ != nullptr) {
    aclrtFree(workspace_);
    workspace_ = nullptr;
  }

  // Destroy graph operation
  if (graph_op_ != nullptr) {
    atb::DestroyOperation(graph_op_);
    graph_op_ = nullptr;
  }
}

void AscendGraphExecutor::execute(const std::vector<Tensor>& inputs,
                                   std::vector<Tensor>& outputs) {
  const bool profile_enabled = shouldProfileAscendGraph();
  const auto total_start = profile_enabled ? Clock::now() : Clock::time_point{};
  double setup_ms = 0.0;
  double alloc_ms = 0.0;
  double execute_ms = 0.0;
  double sync_ms = 0.0;
  const std::string graph_name = graph_op_ != nullptr ? graph_op_->GetName() : "null_graph";

  // 1. Build VariantPack
  atb::VariantPack variantPack;
  variantPack.inTensors.resize(inputs.size());
  variantPack.outTensors.resize(outputs.size());

  for (size_t i = 0; i < inputs.size(); ++i) {
    fillAtbTensor(inputs[i], variantPack.inTensors[i]);
  }

  for (size_t i = 0; i < outputs.size(); ++i) {
    fillAtbTensor(outputs[i], variantPack.outTensors[i]);
  }

  // 2. Setup: compute required workspace size and refresh ATB execution state.
  uint64_t required_workspace = 0;
  atb::Status ret = atb::NO_ERROR;
  const auto setup_start = profile_enabled ? Clock::now() : Clock::time_point{};
  ret = graph_op_->Setup(variantPack, required_workspace, context_);
  if (ret != atb::NO_ERROR) {
    MLLM_ERROR_EXIT(ExitCode::kAscendError,
                    "Graph Setup failed, status={}", static_cast<int>(ret));
  }
  if (profile_enabled) setup_ms = elapsedMs(setup_start, Clock::now());

  // 3. Allocate or resize workspace if needed
  const auto alloc_start = profile_enabled ? Clock::now() : Clock::time_point{};
  if (required_workspace > workspace_size_) {
    // Free old workspace
    if (workspace_ != nullptr) {
      aclrtFree(workspace_);
      workspace_ = nullptr;
    }

    // Allocate new workspace
    if (required_workspace > 0) {
      auto acl_ret = aclrtMalloc(&workspace_, required_workspace,
                                 ACL_MEM_MALLOC_HUGE_FIRST);
      if (acl_ret != ACL_SUCCESS) {
        MLLM_ERROR_EXIT(ExitCode::kAscendError,
                        "Failed to allocate workspace of {} bytes, acl_ret={}",
                        required_workspace, static_cast<int>(acl_ret));
      }
      workspace_size_ = required_workspace;
    }
  }
  if (profile_enabled) alloc_ms = elapsedMs(alloc_start, Clock::now());

  // 4. Execute graph
  const auto execute_start = profile_enabled ? Clock::now() : Clock::time_point{};
  ret = graph_op_->Execute(variantPack,
                           reinterpret_cast<uint8_t*>(workspace_),
                           workspace_size_,
                           context_);
  if (ret != atb::NO_ERROR) {
    MLLM_ERROR_EXIT(ExitCode::kAscendError,
                    "Graph Execute failed, status={}", static_cast<int>(ret));
  }
  if (profile_enabled) execute_ms = elapsedMs(execute_start, Clock::now());

  // 5. Synchronize stream to ensure execution completes
  const auto sync_start = profile_enabled ? Clock::now() : Clock::time_point{};
  syncGlobalAtbStream();
  if (profile_enabled) {
    sync_ms = elapsedMs(sync_start, Clock::now());
    recordGraphProfile(graph_name,
                       setup_ms,
                       alloc_ms,
                       execute_ms,
                       sync_ms,
                       elapsedMs(total_start, Clock::now()));
  }
}

}  // namespace mllm::ascend
