// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/cpu/ops/CmpOp.hpp"
#include "mllm/backends/cpu/ops/ElewiseOps.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/mllm.hpp"

namespace mllm::cpu {

CPUEqualOp::CPUEqualOp(const aops::EqualOpOptions& options) : aops::EqualOp(options) {}

void CPUEqualOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& input0 = inputs[0];
  auto& input1 = inputs[1];
  auto& output = outputs[0];

  auto dtype = input0.dtype();

  auto broadcast_info = calculateBroadcastInfo(input0.shape(), input1.shape());
  bool can_be_broadcast_naive = broadcast_info.can_be_broadcast_naive;
  int32_t batch_dims = broadcast_info.batch_dims;
  int32_t broadcast_naive_loops = broadcast_info.broadcast_naive_loops;
  int32_t vector_size = broadcast_info.size;

  switch (dtype) {
    case kFloat32: {
      const float* a = input0.ptr<mllm_fp32_t>();
      const float* b = input1.ptr<mllm_fp32_t>();
      uint8_t* out = output.ptr<mllm_uint8_t>();

      if (input0.numel() == input1.numel()) {
        for (size_t i = 0; i < output.numel(); ++i) out[i] = (a[i] == b[i]) ? 1 : 0;
      } else if (input1.numel() == 1) {
        float bv = *b;
        for (size_t i = 0; i < output.numel(); ++i) out[i] = (a[i] == bv) ? 1 : 0;
      } else if (can_be_broadcast_naive) {
        for (int batch = 0; batch < batch_dims; ++batch) {
          for (int l = 0; l < broadcast_naive_loops; ++l) {
            size_t a_offset = batch * broadcast_naive_loops * vector_size + l * vector_size;
            size_t b_offset = batch * vector_size;
            size_t out_offset = batch * broadcast_naive_loops * vector_size + l * vector_size;
            for (int v = 0; v < vector_size; ++v) out[out_offset + v] = (a[a_offset + v] == b[b_offset + v]) ? 1 : 0;
          }
        }
      } else {
        NYI("EqualOp broadcast not supported for this dtype.");
      }
      break;
    }

    case kInt32: {
      const int32_t* a = input0.ptr<mllm_int32_t>();
      const int32_t* b = input1.ptr<mllm_int32_t>();
      uint8_t* out = output.ptr<mllm_uint8_t>();
      if (input0.numel() == input1.numel()) {
        for (size_t i = 0; i < output.numel(); ++i) out[i] = (a[i] == b[i]) ? 1 : 0;
      } else if (input1.numel() == 1) {
        int32_t bv = *b;
        for (size_t i = 0; i < output.numel(); ++i) out[i] = (a[i] == bv) ? 1 : 0;
      } else if (can_be_broadcast_naive) {
        for (int batch = 0; batch < batch_dims; ++batch) {
          for (int l = 0; l < broadcast_naive_loops; ++l) {
            size_t a_offset = batch * broadcast_naive_loops * vector_size + l * vector_size;
            size_t b_offset = batch * vector_size;
            size_t out_offset = batch * broadcast_naive_loops * vector_size + l * vector_size;
            for (int v = 0; v < vector_size; ++v) out[out_offset + v] = (a[a_offset + v] == b[b_offset + v]) ? 1 : 0;
          }
        }
      } else {
        NYI("EqualOp broadcast not supported for this dtype.");
      }
      break;
    }

    case kInt16: {
      const int16_t* a = input0.ptr<mllm_int16_t>();
      const int16_t* b = input1.ptr<mllm_int16_t>();
      uint8_t* out = output.ptr<mllm_uint8_t>();
      if (input0.numel() == input1.numel()) {
        for (size_t i = 0; i < output.numel(); ++i) out[i] = (a[i] == b[i]) ? 1 : 0;
      } else if (input1.numel() == 1) {
        int16_t bv = *b;
        for (size_t i = 0; i < output.numel(); ++i) out[i] = (a[i] == bv) ? 1 : 0;
      } else {
        NYI("EqualOp broadcast not supported for this dtype.");
      }
      break;
    }

    case kInt8: {
      const int8_t* a = input0.ptr<mllm_int8_t>();
      const int8_t* b = input1.ptr<mllm_int8_t>();
      uint8_t* out = output.ptr<mllm_uint8_t>();
      if (input0.numel() == input1.numel()) {
        for (size_t i = 0; i < output.numel(); ++i) out[i] = (a[i] == b[i]) ? 1 : 0;
      } else if (input1.numel() == 1) {
        int8_t bv = *b;
        for (size_t i = 0; i < output.numel(); ++i) out[i] = (a[i] == bv) ? 1 : 0;
      } else {
        NYI("EqualOp broadcast not supported for this dtype.");
      }
      break;
    }

    default: NYI("EqualOp not supported for this dtype.");
  }
}

}  // namespace mllm::cpu
