// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/cpu/ops/GatherOp.hpp"
#include "mllm/core/Tensor.hpp"

namespace mllm::cpu {

CPUGatherOp::CPUGatherOp(const aops::GatherOpOptions& options) : aops::GatherOp(options) {}

void CPUGatherOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& table = inputs[0];
  auto& indices = inputs[1];
  auto& output = outputs[0];

  int dim = options_.dim;
  if (dim < 0) dim += table.shape().size();

  int64_t outer_size = 1;
  for (int i = 0; i < dim; ++i) outer_size *= table.shape()[i];

  int64_t inner_size = 1;
  for (int i = dim + 1; i < table.shape().size(); ++i) inner_size *= table.shape()[i];

  int64_t dim_size = table.shape()[dim];
  int64_t indices_count = indices.numel();

  size_t data_type_size = 4;
  switch (table.dtype()) {
    case MLLM_TYPE_F32: data_type_size = sizeof(float); break;
    case MLLM_TYPE_F16: data_type_size = sizeof(mllm_fp16_t); break;
    case MLLM_TYPE_I32: data_type_size = sizeof(int32_t); break;
    default: MLLM_ERROR("GatherOp table type not supported: {}", (int)table.dtype());
  }

  const uint8_t* table_ptr = table.ptr<uint8_t>();
  uint8_t* output_ptr = output.ptr<uint8_t>();

  const int32_t* indices_i32 = indices.dtype() == MLLM_TYPE_I32 ? indices.ptr<int32_t>() : nullptr;
  const float* indices_f32 = !indices_i32 && indices.dtype() == MLLM_TYPE_F32 ? indices.ptr<float>() : nullptr;

  if (!indices_i32 && !indices_f32) {
    MLLM_ERROR("GatherOp indices type not supported: {}", (int)indices.dtype());
    return;
  }

  // FIXME: parallel
  for (int64_t o = 0; o < outer_size; ++o) {
    for (int64_t i = 0; i < indices_count; ++i) {
      int64_t idx = 0;
      if (indices_i32) {
        idx = indices_i32[i];
      } else if (indices_f32) {
        idx = (int64_t)indices_f32[i];
      }

      if (idx < 0) idx += dim_size;

      if (idx < 0 || idx >= dim_size) { continue; }

      int64_t src_offset = (o * dim_size + idx) * inner_size * data_type_size;
      int64_t dst_offset = (o * indices_count + i) * inner_size * data_type_size;

      std::memcpy(output_ptr + dst_offset, table_ptr + src_offset, inner_size * data_type_size);
    }
  }
}

}  // namespace mllm::cpu
