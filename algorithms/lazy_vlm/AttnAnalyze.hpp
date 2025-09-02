// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <mllm/core/Tensor.hpp>

std::vector<int32_t> analyzePrefillAttn(mllm::Tensor attn);

std::vector<int32_t> analyzeDecodeAttn(mllm::Tensor attn);
