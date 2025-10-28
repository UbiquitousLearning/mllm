// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cstring>
#include "mllm/core/aops/MatMulOp.hpp"
#include "mllm/backends/cpu/ops/Conv2DOp.hpp"
#include "mllm/backends/cpu/kernels/Kernels.hpp"

namespace mllm::cpu {

CPUConv2DOp::CPUConv2DOp(const aops::Conv2DOpOptions& options) : aops::Conv2DOp(options) {}

void CPUConv2DOp::load(const ParameterFile::ptr_t& ploader) {
  switch (ploader->version()) {
    case ModelFileVersion::kV1: {
      weight_ = ploader->pull(getName() + ".weight");
      if (options_.bias) { bias_ = ploader->pull(getName() + ".bias"); }
      weight_ = weight_.view({
          options_.out_channels,
          options_.in_channels,
          options_.kernel_size[0],
          options_.kernel_size[1],
      });
      if (options_.bias) { bias_ = bias_.view({options_.out_channels}); }
      break;
    }
    case ModelFileVersion::kUserTemporary:
    case ModelFileVersion::kV2: {
      weight_ = ploader->pull(getName() + ".weight");
      if (options_.bias) { bias_ = ploader->pull(getName() + ".bias"); }
      break;
    }
    default: NYI("Unsupported model file version")
  }

  auto& kernel_size = options_.kernel_size;
  auto& stride = options_.stride;
  auto& padding = options_.padding;
  auto& dilation = options_.dilation;

  // Pack data
  switch (options_.impl_type) {
    case aops::Conv2DOpImplType::kDefault: {
      // We will do im2col algorithm when using default impl. We will packing weight here.
      MLLM_INFO("Packing Conv2D weight to im2col format. kh={}, kw={}, pw={}, ph={}, dw={}, dh={}, sw={}, sh={}",
                kernel_size[0], kernel_size[1], padding[0], padding[1], dilation[0], dilation[1], stride[0], stride[1]);
      auto packed_weight = Tensor::empty(
                               {
                                   options_.out_channels,
                                   options_.in_channels * kernel_size[0] * kernel_size[1],

                               },
                               weight_.dtype(), weight_.device())
                               .alloc();
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
      arm::conv2d_fp32_im2col_weight(weight_.ptr<mllm_fp32_t>(), packed_weight.ptr<mllm_fp32_t>(), options_.out_channels,
                                     options_.in_channels, kernel_size[0], kernel_size[1]);
#else
      MLLM_ERROR_EXIT(ExitCode::kCoreError, "Unsupported architecture for packing conv2d weight into im2col format.");
#endif
      weight_ = packed_weight;
      break;
    }
    default: {
      NYI("Unsupported impl type")
    }
  }
}

void CPUConv2DOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& input = inputs[0];
  auto& output = outputs[0];
  auto& kernel_size = options_.kernel_size;
  auto& stride = options_.stride;
  auto& padding = options_.padding;
  auto& dilation = options_.dilation;

  MLLM_RT_ASSERT_EQ(input.rank(), 4);
  MLLM_RT_ASSERT_EQ(output.rank(), 4);
  auto batch_size = input.size(0);
  auto _1 = input.size(1);
  auto _2 = input.size(2);
  auto _3 = input.size(3);
  auto _out_1 = output.size(1);
  auto _out_2 = output.size(2);
  auto _out_3 = output.size(3);

  switch (input.dtype()) {
    case kFloat32: {
      switch (options_.impl_type) {
        case aops::Conv2DOpImplType::kDefault: {
          // Weight is M x K  (out_channels x (in_channels * kernel_h * kernel_w))
          // Input is K x N  ((in_channels * kernel_h * kernel_w) x (out_h * out_w))
          // Output is M x N  (out_channels x (out_h * out_w))

          auto mt = aops::MatMulOpType::kDefault;
          if (mt == aops::MatMulOpType::kDefault) {
#if defined(MLLM_USE_BLAS)
            mt = aops::MatMulOpType::kBLAS;
#else
            mt = aops::MatMulOpType::kMllmBlas;
#endif
          }
          int MATMUL_M = options_.out_channels;
          int MATMUL_K = options_.in_channels * kernel_size[0] * kernel_size[1];
          int MATMUL_N = output.shape()[2] * output.shape()[3];

          // step 1. im2col inputs to tmp
          auto packed_inputs = Tensor::empty({MATMUL_K, MATMUL_N}, input.dtype(), input.device()).alloc();

          for (int _b_idx = 0; _b_idx < batch_size; ++_b_idx) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)

            arm::conv2d_fp32_im2col_input(input.ptr<mllm_fp32_t>() + _b_idx * (_1 * _2 * _3), options_.in_channels,
                                          input.shape()[2], input.shape()[3], kernel_size[0], kernel_size[1], padding[0],
                                          padding[1], stride[0], stride[1], dilation[0], dilation[1],
                                          packed_inputs.ptr<mllm_fp32_t>());
            // step 2. Do matmul
            switch (mt) {  // NOLINT
              case aops::MatMulOpType::kBLAS: {
#if defined(MLLM_USE_BLAS)
                blas::matmul_fp32(weight_.ptr<mllm_fp32_t>(), packed_inputs.ptr<mllm_fp32_t>(),
                                  output.ptr<mllm_fp32_t>() + _b_idx * (_out_1 * _out_2 * _out_3), nullptr, MATMUL_M, MATMUL_N,
                                  MATMUL_K, false, false);

                // Add Bias
                if (options_.bias) {
                  auto out_ptr = output.ptr<mllm_fp32_t>() + _b_idx * (_out_1 * _out_2 * _out_3);
                  const auto bias_ptr = bias_.ptr<mllm_fp32_t>();
                  for (int m = 0; m < MATMUL_M; ++m) {
                    const float b = bias_ptr[m];
                    for (int n = 0; n < MATMUL_N; ++n) { out_ptr[m * MATMUL_N + n] += b; }
                  }
                }
#else
                NYI("BLAS not supported. Pls set MLLM_USE_BLAS=ON to enable BLAS supports in cmake.");
#endif
                break;
              }
              case aops::MatMulOpType::kMllmBlas: {
                auto thread_count = options_.getThreads();
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
                arm::mllm_blas_matmul_fp32(
                    MATMUL_M, MATMUL_K, MATMUL_N, output.ptr<mllm_fp32_t>() + _b_idx * (_out_1 * _out_2 * _out_3),
                    weight_.ptr<mllm_fp32_t>(), packed_inputs.ptr<mllm_fp32_t>(), nullptr, false, false, thread_count);
                // Add Bias
                if (options_.bias) {
                  auto out_ptr = output.ptr<mllm_fp32_t>() + _b_idx * (_out_1 * _out_2 * _out_3);
                  const auto bias_ptr = bias_.ptr<mllm_fp32_t>();
                  for (int m = 0; m < MATMUL_M; ++m) {
                    const float b = bias_ptr[m];
                    for (int n = 0; n < MATMUL_N; ++n) { out_ptr[m * MATMUL_N + n] += b; }
                  }
                }
#else
                NYI("MllmBlas only support MLLM_HOST_ARCH_ARM64 or MLLM_HOST_ARCH_ARM right now.")
#endif
                break;
              }
              default: {
                NYI("Unsupported matmul type");
              }
            }
#else
            MLLM_ERROR_EXIT(ExitCode::kCoreError, "Unsupported architecture for perform im2col conv2d.");
#endif
          }
        }
      }
      break;
    }
    default: {
      NYI("Unsupported data type");
    }
  }
}

}  // namespace mllm::cpu
