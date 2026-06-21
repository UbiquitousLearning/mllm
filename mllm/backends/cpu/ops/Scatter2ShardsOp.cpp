// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cstring>
#include "mllm/utils/Common.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/backends/cpu/ops/Scatter2ShardsOp.hpp"

namespace mllm::cpu {

CPUScatter2ShardsOp::CPUScatter2ShardsOp(const aops::Scatter2ShardsOpOptions& options) : aops::Scatter2ShardsOp(options) {}

void CPUScatter2ShardsOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  const auto& src = inputs[0];
  const auto& dst_ptrs = inputs[1];

  // Validation
  MLLM_RT_ASSERT(dst_ptrs.dtype() == kInt64 && dst_ptrs.rank() == 1 && dst_ptrs.shape()[0] == src.shape()[options_.dim]);

  // [B, H, S, D]
  // floating_shards_point_left = B * H
  // floating_shards_point_right = D
  int32_t floating_shards_point_left = 1;
  int32_t floating_shards_point_right = 1;
  for (int i = 0; i < options_.dim; ++i) { floating_shards_point_left *= src.shape()[i]; }
  for (int i = options_.dim + 1; i < src.rank(); ++i) { floating_shards_point_right *= src.shape()[i]; }
  int32_t floating_shards_point_stride = 1;
  for (int i = options_.dim; i < src.rank(); ++i) { floating_shards_point_stride *= src.shape()[i]; }
  if (options_.dim == src.rank() - 1) { floating_shards_point_stride = 1; }
  if (options_.dim == 0) { floating_shards_point_left = src.stride()[0]; }

  int32_t loop_times = src.shape()[options_.dim];
  for (int lo = 0; lo < loop_times; ++lo) {
    for (int ho = 0; ho < floating_shards_point_left; ++ho) {
      auto src_ptr = src.ptr<mllm_byte_t>()
                     + (ho * floating_shards_point_stride + lo * floating_shards_point_right)
                           * (bytesOfType(src.dtype()) / lanesOfType(src.dtype()));
      auto dst_ptr = dst_ptrs.ptr<mllm_byte_t*>()[lo]
                     + ho * floating_shards_point_right * (bytesOfType(src.dtype()) / lanesOfType(src.dtype()));
      memcpy(dst_ptr, src_ptr, floating_shards_point_right * (bytesOfType(src.dtype()) / lanesOfType(src.dtype())));
    }
  }
}

}  // namespace mllm::cpu
