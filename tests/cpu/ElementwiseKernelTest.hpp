// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/mllm.hpp"

#include "KernelTestHelper.hpp"

#define MLLM_CPU_KERNEL_TEST_GEN_EW_TESTS(OpName, EnumDType, CDType, __Op)               \
  bool OpName(const std::vector<mllm::Tensor::shape_t>& shapes) {                        \
    using mllm::kCPU;                                                                    \
    using mllm::EnumDType;                                                               \
    using mllm::Tensor;                                                                  \
    for (auto& s : shapes) {                                                             \
      Tensor a = Tensor::random(s, -3, 3, EnumDType, kCPU);                              \
      Tensor b = Tensor::random(s, -3, 3, EnumDType, kCPU);                              \
      Tensor ref_c = Tensor::zeros(s, EnumDType, kCPU);                                  \
      {                                                                                  \
        auto a_ptr = a.ptr<mllm::CDType>();                                              \
        auto b_ptr = b.ptr<mllm::CDType>();                                              \
        auto c_ptr = ref_c.ptr<mllm::CDType>();                                          \
        auto num_elements = a.numel();                                                   \
        for (size_t i = 0; i < num_elements; i++) { c_ptr[i] = a_ptr[i] __Op b_ptr[i]; } \
      }                                                                                  \
      auto c = a __Op b;                                                                 \
      auto result = mllm::test::allClose(c, ref_c);                                      \
      if (!result) {                                                                     \
        mllm::print(c);                                                                  \
        mllm::print(ref_c);                                                              \
        mllm::print(result);                                                             \
        return false;                                                                    \
      }                                                                                  \
    }                                                                                    \
    return true;                                                                         \
  }

#define MLLM_CPU_KERNEL_TEST_GEN_EW_SCALAR_TESTS(OpName, EnumDType, CDType, __Op)                      \
  bool OpName(const std::vector<mllm::Tensor::shape_t>& shapes) {                                      \
    using mllm::kCPU;                                                                                  \
    using mllm::EnumDType;                                                                             \
    using mllm::Tensor;                                                                                \
    for (auto& s : shapes) {                                                                           \
      Tensor a = Tensor::random(s, -3, 3, EnumDType, kCPU);                                            \
      Tensor b = Tensor::random({1}, 1, 3, EnumDType, kCPU);                                           \
      Tensor ref_c = Tensor::zeros(s, EnumDType, kCPU);                                                \
      {                                                                                                \
        auto a_ptr = a.ptr<mllm::CDType>();                                                            \
        auto b_ptr = b.ptr<mllm::CDType>();                                                            \
        auto c_ptr = ref_c.ptr<mllm::CDType>();                                                        \
        auto num_elements = a.numel();                                                                 \
        for (size_t i = 0; i < num_elements; i++) { c_ptr[i] = (float)a_ptr[i] __Op(float) b_ptr[0]; } \
      }                                                                                                \
      auto c = a __Op b;                                                                               \
      float rtol = 1e-5;                                                                               \
      float atol = 1e-5;                                                                               \
      if (EnumDType == mllm::kFloat16 && std::string(#__Op) == "/") {                                  \
        rtol = 1e-3;                                                                                   \
        atol = 1e-3;                                                                                   \
      }                                                                                                \
      auto result = mllm::test::allClose(c, ref_c, rtol, atol);                                        \
      if (!result) {                                                                                   \
        mllm::print(c);                                                                                \
        mllm::print(ref_c);                                                                            \
        mllm::print(result);                                                                           \
        return false;                                                                                  \
      }                                                                                                \
    }                                                                                                  \
    return true;                                                                                       \
  }

class ElementwiseKernelTest : public KernelTest {
 public:
  ElementwiseKernelTest() = default;
  ~ElementwiseKernelTest() override = default;

  MLLM_CPU_KERNEL_TEST_GEN_EW_TESTS(AddFloat32Test, kFloat32, mllm_fp32_t, +)
  MLLM_CPU_KERNEL_TEST_GEN_EW_TESTS(AddFloat16Test, kFloat16, mllm_fp16_t, +)
  MLLM_CPU_KERNEL_TEST_GEN_EW_TESTS(AddInt8Test, kInt8, mllm_int8_t, +)
  MLLM_CPU_KERNEL_TEST_GEN_EW_TESTS(AddInt16Test, kInt16, mllm_int16_t, +)
  MLLM_CPU_KERNEL_TEST_GEN_EW_TESTS(AddInt32Test, kInt32, mllm_int32_t, +)
  MLLM_CPU_KERNEL_TEST_GEN_EW_TESTS(SubFloat32Test, kFloat32, mllm_fp32_t, -)
  MLLM_CPU_KERNEL_TEST_GEN_EW_TESTS(SubFloat16Test, kFloat16, mllm_fp16_t, -)
  MLLM_CPU_KERNEL_TEST_GEN_EW_TESTS(SubInt8Test, kInt8, mllm_int8_t, -)
  MLLM_CPU_KERNEL_TEST_GEN_EW_TESTS(SubInt16Test, kInt16, mllm_int16_t, -)
  MLLM_CPU_KERNEL_TEST_GEN_EW_TESTS(SubInt32Test, kInt32, mllm_int32_t, -)
  MLLM_CPU_KERNEL_TEST_GEN_EW_TESTS(MulFloat32Test, kFloat32, mllm_fp32_t, *)
  MLLM_CPU_KERNEL_TEST_GEN_EW_TESTS(MulFloat16Test, kFloat16, mllm_fp16_t, *)
  MLLM_CPU_KERNEL_TEST_GEN_EW_TESTS(MulInt8Test, kInt8, mllm_int8_t, *)
  MLLM_CPU_KERNEL_TEST_GEN_EW_TESTS(MulInt16Test, kInt16, mllm_int16_t, *)
  MLLM_CPU_KERNEL_TEST_GEN_EW_TESTS(MulInt32Test, kInt32, mllm_int32_t, *)
  MLLM_CPU_KERNEL_TEST_GEN_EW_TESTS(DivFloat32Test, kFloat32, mllm_fp32_t, /)
  MLLM_CPU_KERNEL_TEST_GEN_EW_TESTS(DivFloat16Test, kFloat16, mllm_fp16_t, /)
  MLLM_CPU_KERNEL_TEST_GEN_EW_TESTS(DivInt8Test, kInt8, mllm_int8_t, /)
  MLLM_CPU_KERNEL_TEST_GEN_EW_TESTS(DivInt16Test, kInt16, mllm_int16_t, /)
  MLLM_CPU_KERNEL_TEST_GEN_EW_TESTS(DivInt32Test, kInt32, mllm_int32_t, /)

  MLLM_CPU_KERNEL_TEST_GEN_EW_SCALAR_TESTS(AddScalarFloat32Test, kFloat32, mllm_fp32_t, +)
  MLLM_CPU_KERNEL_TEST_GEN_EW_SCALAR_TESTS(AddScalarFloat16Test, kFloat16, mllm_fp16_t, +)
  MLLM_CPU_KERNEL_TEST_GEN_EW_SCALAR_TESTS(AddScalarInt8Test, kInt8, mllm_int8_t, +)
  MLLM_CPU_KERNEL_TEST_GEN_EW_SCALAR_TESTS(AddScalarInt16Test, kInt16, mllm_int16_t, +)
  MLLM_CPU_KERNEL_TEST_GEN_EW_SCALAR_TESTS(AddScalarInt32Test, kInt32, mllm_int32_t, +)
  MLLM_CPU_KERNEL_TEST_GEN_EW_SCALAR_TESTS(SubScalarFloat32Test, kFloat32, mllm_fp32_t, -)
  MLLM_CPU_KERNEL_TEST_GEN_EW_SCALAR_TESTS(SubScalarFloat16Test, kFloat16, mllm_fp16_t, -)
  MLLM_CPU_KERNEL_TEST_GEN_EW_SCALAR_TESTS(SubScalarInt8Test, kInt8, mllm_int8_t, -)
  MLLM_CPU_KERNEL_TEST_GEN_EW_SCALAR_TESTS(SubScalarInt16Test, kInt16, mllm_int16_t, -)
  MLLM_CPU_KERNEL_TEST_GEN_EW_SCALAR_TESTS(SubScalarInt32Test, kInt32, mllm_int32_t, -)
  MLLM_CPU_KERNEL_TEST_GEN_EW_SCALAR_TESTS(MulScalarFloat32Test, kFloat32, mllm_fp32_t, *)
  MLLM_CPU_KERNEL_TEST_GEN_EW_SCALAR_TESTS(MulScalarFloat16Test, kFloat16, mllm_fp16_t, *)
  MLLM_CPU_KERNEL_TEST_GEN_EW_SCALAR_TESTS(MulScalarInt8Test, kInt8, mllm_int8_t, *)
  MLLM_CPU_KERNEL_TEST_GEN_EW_SCALAR_TESTS(MulScalarInt16Test, kInt16, mllm_int16_t, *)
  MLLM_CPU_KERNEL_TEST_GEN_EW_SCALAR_TESTS(MulScalarInt32Test, kInt32, mllm_int32_t, *)
  MLLM_CPU_KERNEL_TEST_GEN_EW_SCALAR_TESTS(DivScalarFloat32Test, kFloat32, mllm_fp32_t, /)
  MLLM_CPU_KERNEL_TEST_GEN_EW_SCALAR_TESTS(DivScalarFloat16Test, kFloat16, mllm_fp16_t, /)
  MLLM_CPU_KERNEL_TEST_GEN_EW_SCALAR_TESTS(DivScalarInt8Test, kInt8, mllm_int8_t, /)
  MLLM_CPU_KERNEL_TEST_GEN_EW_SCALAR_TESTS(DivScalarInt16Test, kInt16, mllm_int16_t, /)
  MLLM_CPU_KERNEL_TEST_GEN_EW_SCALAR_TESTS(DivScalarInt32Test, kInt32, mllm_int32_t, /)
};
