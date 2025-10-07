// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include <cstdint>

#include "mllm/core/Tensor.hpp"
#include "mllm/core/aops/MatMulOp.hpp"
#include "mllm/core/aops/SplitOp.hpp"
#include "mllm/engine/Context.hpp"

namespace mllm::nn::functional {

Tensor matmul(const Tensor& A, const Tensor& B, bool transpose_A = false, bool transpose_B = false,
              aops::MatMulOpType type = aops::MatMulOpType::kDefault);

Tensor view(const Tensor& x, const std::vector<int32_t>& shape);

std::vector<Tensor> split(const Tensor& x, int32_t split_size_or_sections, int32_t dim);

std::vector<Tensor> split(const Tensor& x, const std::vector<int32_t>& split_size_or_sections, int32_t dim);

// For structure binding usage. But will increase compile time.
// e.g.:
// Tensor x = Tensor::ones({10, 2, 1024}, kFp32, kCPU);
// auto [x1, x2, x3, x4] = split<4>(x, 256, -1);
// assert(x1.shape()[2] == 1024 / 4)
// assert(x2.shape()[2] == 1024 / 4)
// assert(x3.shape()[2] == 1024 / 4)
// assert(x4.shape()[2] == 1024 / 4)
template<int32_t RET_NUM>
std::array<Tensor, RET_NUM> split(const Tensor& x, int32_t split_size_or_sections, int32_t dim) {
  auto outputs = Context::instance().buildOpAndSubmitTask(
      OpTypes::kSplit, aops::SplitOpOptions{.dim = dim, .split_size_or_sections = {split_size_or_sections}}, {x});
  std::array<Tensor, RET_NUM> ret;

#pragma unroll
  for (int i = 0; i < RET_NUM; ++i) { ret[i] = outputs[i]; }

  return ret;
}

template<int32_t RET_NUM>
std::array<Tensor, RET_NUM> split(const Tensor& x, const std::vector<int32_t>& split_size_or_sections, int32_t dim) {
  auto outputs = Context::instance().buildOpAndSubmitTask(
      OpTypes::kSplit, aops::SplitOpOptions{.dim = dim, .split_size_or_sections = split_size_or_sections}, {x});
  std::array<Tensor, RET_NUM> ret;

#pragma unroll
  for (int i = 0; i < RET_NUM; ++i) { ret[i] = outputs[i]; }

  return ret;
}

template<int32_t RET_NUM>
std::array<Tensor, RET_NUM> split(const Tensor& x, int32_t dim) {
  auto outputs = Context::instance().buildOpAndSubmitTask(
      OpTypes::kSplit, aops::SplitOpOptions{.dim = dim, .split_size_or_sections = {x.shape()[dim] / RET_NUM}}, {x});
  std::array<Tensor, RET_NUM> ret;

#pragma unroll
  for (int i = 0; i < RET_NUM; ++i) { ret[i] = outputs[i]; }

  return ret;
}

inline std::vector<Tensor> split(int32_t num, const Tensor& x, int32_t dim) {
  auto outputs = Context::instance().buildOpAndSubmitTask(
      OpTypes::kSplit, aops::SplitOpOptions{.dim = dim, .split_size_or_sections = {x.shape()[dim] / num}}, {x});
  std::vector<Tensor> ret;

#pragma unroll
  for (int i = 0; i < num; ++i) { ret.push_back(outputs[i]); }

  return ret;
}

template<int32_t RET_NUM>
std::array<Tensor, RET_NUM> chunk(const Tensor& x, int32_t dim) {
  auto outputs = Context::instance().buildOpAndSubmitTask(
      OpTypes::kSplit, aops::SplitOpOptions{.dim = dim, .split_size_or_sections = {x.shape()[dim] / RET_NUM}}, {x});
  std::array<Tensor, RET_NUM> ret;

#pragma unroll
  for (int i = 0; i < RET_NUM; ++i) { ret[i] = outputs[i].contiguous(); }

  return ret;
}

inline std::vector<Tensor> chunk(int32_t num, const Tensor& x, int32_t dim) {
  auto outputs = Context::instance().buildOpAndSubmitTask(
      OpTypes::kSplit, aops::SplitOpOptions{.dim = dim, .split_size_or_sections = {x.shape()[dim] / num}}, {x});
  std::vector<Tensor> ret;

#pragma unroll
  for (int i = 0; i < num; ++i) { ret.push_back(outputs[i].contiguous()); }

  return ret;
}

Tensor concat(const std::vector<Tensor>& ins, int32_t dim);

Tensor flashAttention2(const Tensor& Q, const Tensor& K, const Tensor& V);

Tensor softmax(const Tensor& x, int32_t dim);

Tensor log(const Tensor& x);

Tensor exp(const Tensor& x);

Tensor sin(const Tensor& x);

Tensor cos(const Tensor& x);

std::array<Tensor, 2> topk(const Tensor& x, int32_t k, int32_t dim = -1, bool largest = true, bool sorted = true);

Tensor clip(const Tensor& x, float min_val, float max_val);

// Reduction operations
Tensor min(const Tensor& x, int32_t dim = std::numeric_limits<int32_t>::max(), bool keep_dim = false);
Tensor max(const Tensor& x, int32_t dim = std::numeric_limits<int32_t>::max(), bool keep_dim = false);
Tensor sum(const Tensor& x, int32_t dim = std::numeric_limits<int32_t>::max(), bool keep_dim = false);
Tensor mean(const Tensor& x, int32_t dim = std::numeric_limits<int32_t>::max(), bool keep_dim = false);

Tensor silu(const Tensor& x);
Tensor silu_(const Tensor& x);

void scatter2Shards(const Tensor& src, const Tensor& shards_pointer, int32_t dim);

}  // namespace mllm::nn::functional
