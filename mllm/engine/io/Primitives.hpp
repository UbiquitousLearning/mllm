// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/Tensor.hpp"
#include "mllm/engine/Task.hpp"
#include "mllm/core/ParameterFile.hpp"

namespace mllm::async::io {

TaskResult::sender_t copy(const Tensor& dst, const Tensor& src);

TaskResult::sender_t promoteMMAPTensor2AnonymousMemoryTensor(const Tensor& dst, const Tensor& src);

TaskResult::sender_t loadAnonymousMemoryTensorFromDisk(Tensor& dst, const std::string& tensor_name,
                                                       const std::string& file_name,
                                                       ModelFileVersion version = ModelFileVersion::kV2);

}  // namespace mllm::async::io
