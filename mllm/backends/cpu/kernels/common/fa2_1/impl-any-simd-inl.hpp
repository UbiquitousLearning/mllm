// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <hwy/highway.h>

// Optional, can instead add HWY_ATTR to all functions.
HWY_BEFORE_NAMESPACE();
namespace mllm::cpu::flash_attn2::details {  // NOLINT
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;
void vector_dot_product_fp32_fp32_fp32(const float* lhs, const float* rhs, float* out, size_t len) {
  using D = hn::ScalableTag<float>;
  const D d;
  auto sum = hn::Zero(d);
  size_t i = 0;
  for (; i + hn::Lanes(d) <= len; i += hn::Lanes(d)) {
    auto l = hn::LoadU(d, lhs + i);
    auto r = hn::LoadU(d, rhs + i);
    sum = hn::MulAdd(l, r, sum);
  }
  float result = hn::ReduceSum(d, sum);
  for (; i < len; ++i) result += lhs[i] * rhs[i];
  *out = result;
}

void mul_from_const_fp32(float* from, float c, size_t len) {
  const hn::ScalableTag<float> d;
  const auto vc = hn::Set(d, c);
  size_t i = 0;
  for (; i + hn::Lanes(d) <= len; i += hn::Lanes(d)) {
    auto v = hn::LoadU(d, from + i);
    hn::StoreU(hn::Mul(v, vc), d, from + i);
  }
  for (; i < len; ++i) from[i] *= c;
}

void fma_const_array_fp32(float* acc_o, float acc_s, const float* v_token, size_t len) {
  const hn::ScalableTag<float> d;
  const auto vscale = hn::Set(d, acc_s);
  size_t i = 0;
  for (; i + hn::Lanes(d) <= len; i += hn::Lanes(d)) {
    auto acc = hn::LoadU(d, acc_o + i);
    auto tok = hn::LoadU(d, v_token + i);
    hn::StoreU(hn::MulAdd(tok, vscale, acc), d, acc_o + i);
  }
  for (; i < len; ++i) acc_o[i] += acc_s * v_token[i];
}

void filled_with_const_fp32(float* a, float v, size_t len) {
  const hn::ScalableTag<float> d;
  const auto vv = hn::Set(d, v);
  size_t i = 0;
  for (; i + hn::Lanes(d) <= len; i += hn::Lanes(d)) hn::StoreU(vv, d, a + i);
  for (; i < len; ++i) a[i] = v;
}

}  // namespace HWY_NAMESPACE
}  // namespace mllm::cpu::flash_attn2::details
HWY_AFTER_NAMESPACE();
