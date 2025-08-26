// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "half/half.hpp"
#include "mllm/mllm.hpp"
#include "mllm/nn/Functional.hpp"

#include "KernelTestHelper.hpp"

#define MLLM_CPU_KERNEL_TEST_GEN_REDUCE_TESTS(OpName, EnumDType, CDType) \
  bool OpName(const std::vector<mllm::Tensor::shape_t>& shapes) {        \
    using mllm::kCPU;                                                    \
    using mllm::EnumDType;                                               \
    using mllm::Tensor;                                                  \
    for (auto& s : shapes) {                                             \
      Tensor a = Tensor::random(s, -1, 1, EnumDType, kCPU);              \
      Tensor ref_c = Tensor::zeros({1}, EnumDType, kCPU);                \
      {                                                                  \
        auto a_ptr = a.ptr<mllm::CDType>();                              \
        auto c_ptr = ref_c.ptr<mllm::CDType>();                          \
        mllm::CDType sum = 0;                                            \
        auto num_elements = a.numel();                                   \
        for (size_t i = 0; i < num_elements; i++) { sum += a_ptr[i]; }   \
        c_ptr[0] = sum / static_cast<mllm::CDType>(num_elements);        \
      }                                                                  \
      Tensor c = mllm::nn::functional::mean(a);                          \
      float rtol = 1e-5;                                                 \
      float atol = 1e-5;                                                 \
      if (EnumDType == mllm::kFloat16) {                                 \
        rtol = 1e-3;                                                     \
        atol = 1e-3;                                                     \
      }                                                                  \
      auto result = mllm::test::allClose(c, ref_c, rtol, atol);          \
      if (!result) {                                                     \
        mllm::print(s);                                                  \
        mllm::print(c);                                                  \
        mllm::print(ref_c);                                              \
        mllm::print(result);                                             \
        return false;                                                    \
      }                                                                  \
    }                                                                    \
    return true;                                                         \
  }

class ReduceKernelTest : public KernelTest {
 public:
  ReduceKernelTest() = default;
  ~ReduceKernelTest() override = default;

  MLLM_CPU_KERNEL_TEST_GEN_REDUCE_TESTS(ReduceMeanFloat32Test, kFloat32, mllm_fp32_t)
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
  MLLM_CPU_KERNEL_TEST_GEN_REDUCE_TESTS(ReduceMeanFloat16Test, kFloat16, mllm_fp16_t)
#endif

  bool ReduceSumFloat32Test(const std ::vector<mllm::Tensor::shape_t>& shapes) {
    using mllm ::kCPU;
    using mllm ::kFloat32;
    using mllm ::Tensor;
    for (auto& s : shapes) {
      Tensor a = Tensor ::random(s, -1, 1, kFloat32, kCPU);
      Tensor ref_c = Tensor ::zeros({1}, kFloat32, kCPU);
      {
        auto a_ptr = a.ptr<mllm ::mllm_fp32_t>();
        auto c_ptr = ref_c.ptr<mllm ::mllm_fp32_t>();
        mllm ::mllm_fp32_t sum = 0;
        auto num_elements = a.numel();
        for (size_t i = 0; i < num_elements; i++) { sum += a_ptr[i]; }
        c_ptr[0] = sum;
      }
      Tensor c = mllm ::nn ::functional ::sum(a);
      float rtol = 1e-4;
      float atol = 1e-4;
      auto result = mllm ::test ::allClose(c, ref_c, rtol, atol);
      if (!result) {
        mllm ::print(s);
        mllm ::print(c);
        mllm ::print(ref_c);
        mllm ::print(result);
        return false;
      }
    }
    return true;
  }
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
  bool ReduceSumFloat16Test(const std ::vector<mllm::Tensor::shape_t>& shapes) {
    using mllm::kCPU;
    using mllm::kFloat16;
    using mllm::Tensor;
    for (auto& s : shapes) {
      Tensor a = Tensor ::random(s, -1, 1, kFloat16, kCPU);
      Tensor ref_c = Tensor ::zeros({1}, kFloat16, kCPU);
      {
        auto a_ptr = a.ptr<mllm::mllm_fp16_t>();
        auto c_ptr = ref_c.ptr<mllm::mllm_fp16_t>();
        half_float::half sum(0);
        auto num_elements = a.numel();
        for (size_t i = 0; i < num_elements; i++) { sum += half_float::half(a_ptr[i]); }
        c_ptr[0] = sum;
      }
      Tensor c = mllm::nn::functional::sum(a);
      float rtol = 1e-4;
      float atol = 1e-4;
      auto result = mllm ::test ::allClose(c, ref_c, rtol, atol);
      if (!result) {
        mllm::print(a);
        mllm::print(s);
        mllm::print(c);
        mllm::print(ref_c);
        mllm::print(result);
        return false;
      }
    }
    return true;
  }
#endif
};
