// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/cpu/ops/LinearOp.hpp"
#include "mllm/backends/cpu/kernels/Kernels.hpp"

namespace mllm::cpu {

CPULinearOp::CPULinearOp(const aops::LinearOpOptions& options) : LinearOp(options) {}

void CPULinearOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& input = inputs[0];
  auto& o = outputs[0];

  auto impl_type = options_.impl_type;
  if (impl_type == aops::LinearImplTypes::kDefault) {
#if defined(MLLM_USE_BLAS)
    impl_type = aops::LinearImplTypes::kBLAS;
#endif
  }

  auto input_shape = input.shape();
  MLLM_RT_ASSERT(input_shape.size() >= 2);

  // In Linear
  // inputs is always: [..., S, in_channels]
  // outputs is always: [out_channels, in_channels]
  int M = input_shape[input_shape.size() - 2];
  int K = input_shape[input_shape.size() - 1];
  int N = options_.out_channels;
  MLLM_RT_ASSERT_EQ(K, options_.in_channels);

  int batch_count = 1;
  for (size_t i = 0; i < input_shape.size() - 2; ++i) { batch_count *= input_shape[i]; }

  switch (impl_type) {
    case aops::LinearImplTypes::kBLAS: {
#if defined(MLLM_USE_BLAS)
      MLLM_RT_ASSERT_EQ(input.dtype(), kFloat32);
      MLLM_RT_ASSERT_EQ(weight_.dtype(), kFloat32);
      MLLM_RT_ASSERT_EQ(o.dtype(), kFloat32);
      if (bias_) { MLLM_RT_ASSERT_EQ(bias_.dtype(), kFloat32); }
      if (batch_count == 1) {
        blas::matmul_fp32(input.ptr<mllm_fp32_t>(), weight_.ptr<mllm_fp32_t>(), o.ptr<mllm_fp32_t>(),
                          bias_ ? bias_.ptr<mllm_fp32_t>() : nullptr, M, N, K, false, true);
      } else {
        blas::batch_matmul_fp32(input.ptr<mllm_fp32_t>(), weight_.ptr<mllm_fp32_t>(), o.ptr<mllm_fp32_t>(),
                                bias_ ? bias_.ptr<mllm_fp32_t>() : nullptr, batch_count, M, N, K,
                                input.stride()[input_shape.size() - 3], 0, o.stride()[o.shape().size() - 3], false, true);
      }
#else
      NYI("BLAS not supported. Pls set MLLM_USE_BLAS=ON to enable BLAS supports in cmake.");
#endif
      break;
    }
    default: {
      NYI("LinearImplTypes not supported");
      break;
    }
  }
}

void CPULinearOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  // FIXME: kleidiai may need self-hosted reshape
  LinearOp::reshape(inputs, outputs);
}

}  // namespace mllm::cpu
