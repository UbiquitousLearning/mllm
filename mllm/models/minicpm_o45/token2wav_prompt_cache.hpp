// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <array>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Log.hpp"

namespace mllm::models::minicpm_o45 {

struct MiniCPMO45Token2WavPromptCache {
  std::vector<int32_t> prompt_tokens;
  Tensor prompt_mels = Tensor::nil();  // [1, Tm, 80], float32
  Tensor spk_emb = Tensor::nil();      // [1, 192], float32
};

inline MiniCPMO45Token2WavPromptCache loadMiniCPMO45Token2WavPromptCache(const std::string& file_path) {
  MiniCPMO45Token2WavPromptCache out;

  std::ifstream in(file_path, std::ios::binary);
  if (!in.is_open()) {
    MLLM_ERROR_EXIT(ExitCode::kIOError, "Failed to open MiniCPM-o-4_5 prompt cache: {}", file_path);
  }

  std::array<char, 8> magic{};
  in.read(magic.data(), static_cast<std::streamsize>(magic.size()));
  if (!in.good()) { MLLM_ERROR_EXIT(ExitCode::kIOError, "Prompt cache header read failed: {}", file_path); }
  const std::array<char, 8> expected = {'M', '4', '5', 'P', 'C', '1', '\0', '\0'};
  if (magic != expected) { MLLM_ERROR_EXIT(ExitCode::kIOError, "Invalid prompt cache magic: {}", file_path); }

  uint32_t version = 0;
  in.read(reinterpret_cast<char*>(&version), sizeof(version));
  if (version != 1) { MLLM_ERROR_EXIT(ExitCode::kIOError, "Unsupported prompt cache version {}: {}", version, file_path); }

  int32_t token_len = 0;
  int32_t mel_frames = 0;
  int32_t mel_dim = 0;
  int32_t spk_dim = 0;
  in.read(reinterpret_cast<char*>(&token_len), sizeof(token_len));
  in.read(reinterpret_cast<char*>(&mel_frames), sizeof(mel_frames));
  in.read(reinterpret_cast<char*>(&mel_dim), sizeof(mel_dim));
  in.read(reinterpret_cast<char*>(&spk_dim), sizeof(spk_dim));
  if (!in.good()) { MLLM_ERROR_EXIT(ExitCode::kIOError, "Prompt cache meta read failed: {}", file_path); }
  if (token_len <= 0 || mel_frames <= 0 || mel_dim <= 0 || spk_dim <= 0) {
    MLLM_ERROR_EXIT(ExitCode::kIOError, "Prompt cache has invalid shape metadata: {}", file_path);
  }

  out.prompt_tokens.resize(static_cast<size_t>(token_len));
  in.read(reinterpret_cast<char*>(out.prompt_tokens.data()), sizeof(int32_t) * static_cast<size_t>(token_len));
  if (!in.good()) { MLLM_ERROR_EXIT(ExitCode::kIOError, "Prompt token section read failed: {}", file_path); }

  out.prompt_mels = Tensor::empty({1, mel_frames, mel_dim}, kFloat32, kCPU).alloc();
  in.read(reinterpret_cast<char*>(out.prompt_mels.ptr<float>()),
          sizeof(float) * static_cast<size_t>(mel_frames) * static_cast<size_t>(mel_dim));
  if (!in.good()) { MLLM_ERROR_EXIT(ExitCode::kIOError, "Prompt mel section read failed: {}", file_path); }

  out.spk_emb = Tensor::empty({1, spk_dim}, kFloat32, kCPU).alloc();
  in.read(reinterpret_cast<char*>(out.spk_emb.ptr<float>()), sizeof(float) * static_cast<size_t>(spk_dim));
  if (!in.good()) { MLLM_ERROR_EXIT(ExitCode::kIOError, "Prompt speaker embedding section read failed: {}", file_path); }

  return out;
}

}  // namespace mllm::models::minicpm_o45

