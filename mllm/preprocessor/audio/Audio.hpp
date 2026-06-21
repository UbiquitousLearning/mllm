// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <vector>

namespace mllm::audio {
// specialized for imagebind preprocessing
// TODO: refactor
std::vector<std::vector<std::vector<std::vector<float>>>> processWAV(const std::vector<std::string>& waves,
                                                                     int resample_rate = 16000);

// Read a WAV file and returns MONO data
std::vector<float> readWAV(const std::string& file_path, int resample_rate = 16000);
}  // namespace mllm::audio