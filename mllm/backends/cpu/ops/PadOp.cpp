// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cstring>
#include <algorithm>
#include <cmath>
#include "mllm/backends/cpu/ops/PadOp.hpp"

namespace mllm::cpu {

CPUPadOp::CPUPadOp(const aops::PadOpOptions& options) : aops::PadOp(options) {}

// Helper: compute output shape from input shape and pad vector (starting from last dimension)
static std::vector<int32_t> compute_padded_shape(const std::vector<int32_t>& in_shape, const std::vector<int32_t>& pad) {
  const int D = static_cast<int>(in_shape.size());
  std::vector<int32_t> out_shape = in_shape;
  const int pairs = static_cast<int>(pad.size()) / 2;
  for (int i = 0; i < D; ++i) {
    int base = 2 * (D - 1 - i);
    int32_t before = (base < pad.size()) ? pad[base] : 0;
    int32_t after = (base + 1 < pad.size()) ? pad[base + 1] : 0;
    out_shape[i] = in_shape[i] + before + after;
  }
  return out_shape;
}

// Helper: reflect index to [0, size-1] without repeating edge (PyTorch-like reflect)
static inline int32_t reflect_index(int32_t x, int32_t size) {
  if (size <= 1) return 0;
  int32_t m = size - 1;
  // Map x to the range [-(size-1), size-1]
  int32_t p = std::abs(x);
  int32_t period = 2 * m;
  int32_t r = p % period;
  if (r >= size) { r = period - r; }
  return r;
}

// Helper: replicate index (clamp)
static inline int32_t replicate_index(int32_t x, int32_t size) {
  if (size <= 1) return 0;
  return std::max<int32_t>(0, std::min<int32_t>(x, size - 1));
}

// Helper: circular index (wrap)
static inline int32_t circular_index(int32_t x, int32_t size) {
  if (size <= 1) return 0;
  int32_t r = x % size;
  if (r < 0) r += size;
  return r;
}

void CPUPadOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  const auto& X = inputs[0];
  auto& Y = outputs[0];

  const auto& in_shape = X.shape();
  const auto& opts = options();
  const auto& pad = opts.pad;  // [last_dim_left, last_dim_right, ..., first_dim_left, first_dim_right]

  // Compute output shape and allocate Y if needed
  std::vector<int32_t> out_shape = Y.isNil() ? compute_padded_shape(in_shape, pad) : Y.shape();
  if (Y.isNil() || Y.numel() == 0) { Y = Tensor::empty(out_shape, X.dtype(), X.device()).alloc(); }

  // Precompute pad_before/after per dimension in order
  const int D = static_cast<int>(in_shape.size());
  std::vector<int32_t> pad_before(D, 0), pad_after(D, 0);
  for (int i = 0; i < D; ++i) {
    int base = 2 * (D - 1 - i);
    if (base < pad.size()) pad_before[i] = pad[base];
    if (base + 1 < pad.size()) pad_after[i] = pad[base + 1];
  }

  // Only implement float32 for now
  switch (X.dtype()) {
    case kFloat32: {
      const float* in = X.ptr<float>();
      float* out = Y.ptr<float>();

      // Compute input/output strides for index mapping
      std::vector<int64_t> in_stride(D, 1), out_stride(D, 1);
      for (int i = D - 2; i >= 0; --i) { in_stride[i] = in_stride[i + 1] * in_shape[i + 1]; }
      const auto& actual_out_shape = Y.shape();
      for (int i = D - 2; i >= 0; --i) { out_stride[i] = out_stride[i + 1] * actual_out_shape[i + 1]; }

      const int64_t out_numel = Y.numel();
      const auto mode = opts.mode;
      const float constant_val = opts.value;

      // Iterate over all output elements
      for (int64_t idx = 0; idx < out_numel; ++idx) {
        // Decode idx into coordinates
        int64_t t = idx;
        std::vector<int32_t> oc(D, 0);
        for (int i = 0; i < D; ++i) {
          int64_t s = out_stride[i];
          oc[i] = (i == D - 1) ? static_cast<int32_t>(t) : static_cast<int32_t>(t / s);
          if (i != D - 1) t %= s;
        }

        // Map to input coordinates
        bool oob = false;
        std::vector<int32_t> ic(D, 0);
        for (int i = 0; i < D; ++i) {
          int32_t xi = oc[i] - pad_before[i];
          int32_t s = in_shape[i];
          switch (mode) {
            case aops::PadMode::kConstant:
              if (xi < 0 || xi >= s) {
                oob = true;
                ic[i] = 0;
              } else {
                ic[i] = xi;
              }
              break;
            case aops::PadMode::kReflect: ic[i] = reflect_index(xi, s); break;
            case aops::PadMode::kReplicate: ic[i] = replicate_index(xi, s); break;
            case aops::PadMode::kCircular: ic[i] = circular_index(xi, s); break;
            default: ic[i] = replicate_index(xi, s); break;
          }
        }

        if (mode == aops::PadMode::kConstant && oob) {
          out[idx] = constant_val;
        } else {
          // Compute input linear index and copy value
          int64_t in_idx = 0;
          for (int i = 0; i < D; ++i) { in_idx += static_cast<int64_t>(ic[i]) * in_stride[i]; }
          out[idx] = in[in_idx];
        }
      }

      break;
    }
    default: NYI("CPUPadOp::forward not support dtype {}", nameOfType(X.dtype())); break;
  }
}

}  // namespace mllm::cpu
