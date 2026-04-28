// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/ascend/ops/AscendLinearQuant.hpp"

#include <acl/acl.h>
#include <aclnnop/aclnn_trans_quant_param.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "mllm/core/DataTypes.hpp"
#include "mllm/utils/Common.hpp"

namespace mllm::ascend {
namespace MLLM_ANONYMOUS_NAMESPACE {

Tensor toCpuTensor(const Tensor& tensor) {
  Tensor tmp = tensor;
  return tmp.device() == kAscend ? tmp.to(kCPU) : tmp;
}

Tensor makeTransQuantParamTensor(const std::string& context, const std::vector<float>& scales) {
  uint64_t* deq_u64_host = nullptr;
  uint64_t deq_u64_count = 0;
  auto acl_st = aclnnTransQuantParam(
      scales.data(), static_cast<uint64_t>(scales.size()),
      nullptr, 0,
      &deq_u64_host, &deq_u64_count);
  if (acl_st != ACL_SUCCESS || deq_u64_host == nullptr) {
    MLLM_ERROR_EXIT(ExitCode::kAscendError,
                    "{}: aclnnTransQuantParam failed, status={}",
                    context,
                    static_cast<int>(acl_st));
  }
  if (deq_u64_count != static_cast<uint64_t>(scales.size())) {
    free(deq_u64_host);
    MLLM_ERROR_EXIT(ExitCode::kAscendError,
                    "{}: aclnnTransQuantParam returned {} values for {} scales",
                    context,
                    deq_u64_count,
                    scales.size());
  }

  Tensor deq_cpu = Tensor::empty({static_cast<int32_t>(scales.size())}, kUInt64, kCPU).alloc();
  std::memcpy(deq_cpu.ptr<uint64_t>(), deq_u64_host, deq_u64_count * sizeof(uint64_t));
  free(deq_u64_host);
  return deq_cpu.to(kAscend);
}

}  // namespace MLLM_ANONYMOUS_NAMESPACE

AscendLinearW8A8Artifacts prepareLinearW8A8Artifacts(const std::string& layer_name,
                                                     int out_channels,
                                                     const Tensor& scale_w_raw,
                                                     const Tensor& scale_x_raw) {
  AscendLinearW8A8Artifacts artifacts;
  artifacts.scale_w_cpu = toCpuTensor(scale_w_raw);
  artifacts.scale_x_cpu = toCpuTensor(scale_x_raw);

  if (artifacts.scale_w_cpu.dtype() != kFloat32
      || artifacts.scale_w_cpu.numel() != static_cast<size_t>(out_channels)) {
    MLLM_ERROR_EXIT(ExitCode::kAscendError,
                    "AscendLinearOp W8A8 load: layer {} expected FP32 scale_w with {} elements",
                    layer_name,
                    out_channels);
  }
  if (artifacts.scale_x_cpu.dtype() != kFloat32 || artifacts.scale_x_cpu.numel() != 1) {
    MLLM_ERROR_EXIT(ExitCode::kAscendError,
                    "AscendLinearOp W8A8 load: layer {} expected FP32 scale_x with 1 element",
                    layer_name);
  }

  artifacts.scale_x = artifacts.scale_x_cpu.ptr<float>()[0];
  if (artifacts.scale_x <= 0.0f) {
    MLLM_ERROR_EXIT(ExitCode::kAscendError,
                    "AscendLinearOp W8A8 load: layer {} has non-positive scale_x={}",
                    layer_name,
                    artifacts.scale_x);
  }

  const float* sw = artifacts.scale_w_cpu.ptr<float>();
  std::vector<float> static_deq(static_cast<size_t>(out_channels));
  std::vector<float> dynamic_deq(static_cast<size_t>(out_channels));
  for (int n = 0; n < out_channels; ++n) {
    static_deq[static_cast<size_t>(n)] = artifacts.scale_x * sw[n];
    dynamic_deq[static_cast<size_t>(n)] = sw[n];
  }

  artifacts.deq_scale_npu =
      makeTransQuantParamTensor("AscendLinearOp W8A8 static deq scale for " + layer_name, static_deq);
  artifacts.deq_scale_w_npu =
      makeTransQuantParamTensor("AscendLinearOp W8A8 dynamic deq scale for " + layer_name, dynamic_deq);

  Tensor bias_cpu = Tensor::empty({1, out_channels}, kInt32, kCPU).alloc();
  std::memset(bias_cpu.ptr<int32_t>(), 0, static_cast<size_t>(out_channels) * sizeof(int32_t));
  artifacts.bias_int32_npu = bias_cpu.to(kAscend);

  return artifacts;
}

}  // namespace mllm::ascend
