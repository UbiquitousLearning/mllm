// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <string>
#include <unordered_map>

#include "mllm/mllm.hpp"
#include "mllm/backends/cpu/kernels/arm/hpc/hpc_sgemm.hpp"

#include "KernelTestHelper.hpp"

class HpcArmSgemvKernelTest : public KernelTest {
 public:
  HpcArmSgemvKernelTest() = default;
  ~HpcArmSgemvKernelTest() override = default;

  bool cmp_hpc_matmul_fp32_gemv_nt_t_decode_small_d_qk(std::unordered_map<std::string, int32_t>& vars) {
    int D = vars["D"];
    int S = vars["S"];

    auto A = mllm::Tensor::random({1, D}, mllm::kFloat32, mllm::kCPU);
    auto B = mllm::Tensor::random({S, D}, mllm::kFloat32, mllm::kCPU);
    auto C = mllm::Tensor::random({1, D}, mllm::kFloat32, mllm::kCPU);
    auto DST = mllm::Tensor::random({1, D}, mllm::kFloat32, mllm::kCPU);

    auto a_ptr = A.ptr<float>();
    auto b_ptr = B.ptr<float>();
    auto c_ptr = C.ptr<float>();
    auto dst_ptr = DST.ptr<float>();

    mllm::cpu::arm::__hpc_matmul_fp32_gemv_nt_t_decode_small_d_qk_baseline(1, D, S, dst_ptr, a_ptr, b_ptr, c_ptr, false, true,
                                                                           1);

    auto DSTP = mllm::Tensor::random({1, D}, mllm::kFloat32, mllm::kCPU);
    auto dstp_ptr = DSTP.ptr<float>();
    mllm::cpu::arm::__hpc_matmul_fp32_gemv_nt_t_decode_small_d_qk(1, D, S, dstp_ptr, a_ptr, b_ptr, c_ptr, false, true, 1);
    auto result = mllm::test::allClose(DSTP, DST);
    if (!result.is_close) {
      mllm::print(result);
      return false;
    }
    return true;
  }

  bool test_hpc_matmul_fp32_gemv_nt_t_decode_small_d_qk(const std::vector<std::unordered_map<std::string, int32_t>>& vars) {
    for (auto v : vars) {
      if (!cmp_hpc_matmul_fp32_gemv_nt_t_decode_small_d_qk(v)) { return false; }
    }
    return true;
  }
};