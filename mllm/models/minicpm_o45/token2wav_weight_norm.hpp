// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "mllm/core/ParameterFile.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Log.hpp"

namespace mllm::models::minicpm_o45 {

inline bool _endsWith(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

inline Tensor _materializeWeightNorm(Tensor g_in, Tensor v_in) {
  auto g = g_in.dtype() == kFloat32 ? g_in.contiguous() : g_in.to(kFloat32).contiguous();
  auto v = v_in.dtype() == kFloat32 ? v_in.contiguous() : v_in.to(kFloat32).contiguous();

  const int64_t out_dim = static_cast<int64_t>(g.numel());
  const int64_t total = static_cast<int64_t>(v.numel());
  if (out_dim <= 0 || total <= 0 || (total % out_dim) != 0) {
    MLLM_ERROR_EXIT(ExitCode::kCoreError,
                    "Invalid weight-norm tensor shape: g.numel()={}, v.numel()={}",
                    out_dim, total);
  }

  const int64_t row = total / out_dim;
  auto w = Tensor::empty({static_cast<int32_t>(total)}, kFloat32, kCPU).alloc();
  auto* g_ptr = g.ptr<float>();
  auto* v_ptr = v.ptr<float>();
  auto* w_ptr = w.ptr<float>();

  constexpr float kEps = 1e-12f;
  for (int64_t i = 0; i < out_dim; ++i) {
    const int64_t base = i * row;
    float norm = 0.0f;
    for (int64_t j = 0; j < row; ++j) {
      const float val = v_ptr[base + j];
      norm += val * val;
    }
    norm = std::sqrt(std::max(norm, kEps));
    const float scale = g_ptr[i] / norm;
    for (int64_t j = 0; j < row; ++j) { w_ptr[base + j] = v_ptr[base + j] * scale; }
  }
  return w;
}

inline int32_t materializeWeightNormParameters(const ParameterFile::ptr_t& param_file, const std::string& scope_prefix) {
  std::vector<std::string> keys;
  keys.reserve(param_file->dict().size());
  for (const auto& kv : param_file->dict()) { keys.push_back(kv.first); }

  const std::string marker = ".parametrizations.weight.original0";
  int32_t count = 0;
  for (const auto& key : keys) {
    if (!_endsWith(key, marker)) { continue; }
    if (!scope_prefix.empty() && key.rfind(scope_prefix, 0) != 0) { continue; }

    const auto prefix = key.substr(0, key.size() - marker.size());
    const auto key_g = prefix + ".parametrizations.weight.original0";
    const auto key_v = prefix + ".parametrizations.weight.original1";
    const auto key_w = prefix + ".weight";

    if (param_file->has(key_w)) { continue; }
    if (!param_file->has(key_g) || !param_file->has(key_v)) { continue; }

    auto weight = _materializeWeightNorm(param_file->pull(key_g), param_file->pull(key_v));
    param_file->push(key_w, weight);
    ++count;
  }
  return count;
}

}  // namespace mllm::models::minicpm_o45
