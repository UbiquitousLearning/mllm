// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <limits>
#include <cstdint>

#include "mllm/experiments/auto_tune/TuningSpace.hpp"
#include "mllm/engine/Context.hpp"
#include "mllm/engine/Task.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/utils/ProgressBar.hpp"

namespace mllm {

OpTunningSpace::OpTunningSpace(OpTypes op_type, DeviceTypes device_type) : op_type_(op_type), device_type_(device_type) {}

std::vector<std::unordered_map<std::string, std::any>> OpTunningSpace::product() const {
  std::vector<std::unordered_map<std::string, std::any>> result;

  if (space_.empty()) { return result; }

  std::vector<std::string> keys;
  std::vector<size_t> sizes;
  for (const auto& pair : space_) {
    keys.push_back(pair.first);
    sizes.push_back(pair.second.size());
  }

  size_t total_combinations = 1;
  for (size_t size : sizes) { total_combinations *= size; }

  for (size_t i = 0; i < total_combinations; ++i) {
    std::unordered_map<std::string, std::any> combination;
    size_t index = i;

    for (int j = static_cast<int>(keys.size()) - 1; j >= 0; --j) {
      const std::string& key = keys[j];
      size_t key_index = index % sizes[j];
      combination[key] = space_.at(key)[key_index];
      index /= sizes[j];
    }

    result.push_back(std::move(combination));
  }

  return result;
}

void OpTunningSpace::addTuningParameter(const std::string& name, const std::vector<std::any>& values) {
  if (space_.count(name)) { MLLM_ERROR_EXIT(ExitCode::kCoreError, "Tuning parameter {} already exists", name); }
  space_.insert({name, values});
}

void OpTunningSpace::tune() {
  auto spaces = product();

  uint64_t cur_best_times = std::numeric_limits<uint64_t>::max();
  std::unordered_map<std::string, std::any> cur_best_space;

  std::string label = "Tuning Op: " + optype2Str(op_type_) + " On: " + deviceTypes2Str(device_type_);
  ProgressBar bar(label, spaces.size());
  int cnt = 0;
  for (auto& space : spaces) {
    auto op = buildOp(space);
    auto inputs = buildInputs(space);

    // Setup task
    auto task = std::make_shared<Task>();
    task->type = TaskTypes::kExecuteOp;
    task->op = op;
    task->inputs = inputs;

    // Run task
    auto starts = Context::instance().curTime();
    Context::instance().dispatcherManager()->submit(op->getDevice(), task);
    // Some backends may need async.
    auto ends = Context::instance().curTime();

    if (ends - starts < cur_best_times) {
      cur_best_times = ends - starts;
      cur_best_space = space;
    }
    bar.update(cnt++);
  }
}

}  // namespace mllm
