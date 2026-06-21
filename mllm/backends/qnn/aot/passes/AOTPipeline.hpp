// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <vector>

#include "mllm/backends/qnn/aot/QnnWrappersAPI.hpp"
#include "mllm/compile/passes/Pass.hpp"
#include "mllm/core/ParameterFile.hpp"

namespace mllm::qnn::aot {

std::vector<std::shared_ptr<ir::Pass>> createQnnAOTLoweringPipeline(QnnAOTEnv* env, const std::string& config_path,
                                                                    const ParameterFile::ptr_t& pf);

}  // namespace mllm::qnn::aot
