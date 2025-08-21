// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <vector>

namespace mllm::audio {
std::vector<std::vector<std::vector<std::vector<float>>>> processWAV(const std::vector<std::string>& waves,
                                                                     int resample_rate = 16000);
}