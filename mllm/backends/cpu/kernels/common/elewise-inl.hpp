// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <hwy/highway.h>
#include "mllm/core/DataTypes.hpp"

HWY_BEFORE_NAMESPACE();
namespace mllm::cpu::common {  // NOLINT
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

//===----------------------------------------------------------------------===//
// Elementwise + - * / By Matrix
//===----------------------------------------------------------------------===//
template<typename T, typename Op>
HWY_INLINE void __elementwise(const T* HWY_RESTRICT x, const T* HWY_RESTRICT y, T* HWY_RESTRICT out, size_t count, Op&& op) {
  const hn::ScalableTag<T> d;
  const size_t N = hn::Lanes(d);
  size_t idx = 0;

  for (; idx + N <= count; idx += N) {
    const hn::Vec<decltype(d)> vx = hn::LoadU(d, x + idx);
    const hn::Vec<decltype(d)> vy = hn::LoadU(d, y + idx);
    const hn::Vec<decltype(d)> result = op(d, vx, vy);
    hn::StoreU(result, d, out + idx);
  }

  if (idx < count) {
    const hn::Vec<decltype(d)> vx = hn::LoadN(d, x + idx, count - idx);
    const hn::Vec<decltype(d)> vy = hn::LoadN(d, y + idx, count - idx);
    const hn::Vec<decltype(d)> result = op(d, vx, vy);
    hn::StoreN(result, d, out + idx, count - idx);
  }
}

struct AddOp {
  template<class D, class V>
  HWY_INLINE V operator()(D d, V a, V b) const {
    return hn::Add(a, b);
  }
};

struct SubOp {
  template<class D, class V>
  HWY_INLINE V operator()(D d, V a, V b) const {
    return hn::Sub(a, b);
  }
};

struct MulOp {
  template<class D, class V>
  HWY_INLINE V operator()(D d, V a, V b) const {
    return hn::Mul(a, b);
  }
};

struct DivOp {
  template<class D, class V>
  HWY_INLINE V operator()(D d, V a, V b) const {
    return hn::Div(a, b);
  }
};

HWY_NOINLINE HWY_MAYBE_UNUSED void elewise_add_fp32(mllm_fp32_t* out, const mllm_fp32_t* x, const mllm_fp32_t* y, size_t n) {
  __elementwise(x, y, out, n, AddOp{});
}

HWY_NOINLINE HWY_MAYBE_UNUSED void elewise_sub_fp32(mllm_fp32_t* out, const mllm_fp32_t* x, const mllm_fp32_t* y, size_t n) {
  __elementwise(x, y, out, n, SubOp{});
}

HWY_NOINLINE HWY_MAYBE_UNUSED void elewise_mul_fp32(mllm_fp32_t* out, const mllm_fp32_t* x, const mllm_fp32_t* y, size_t n) {
  __elementwise(x, y, out, n, MulOp{});
}

HWY_NOINLINE HWY_MAYBE_UNUSED void elewise_div_fp32(mllm_fp32_t* out, const mllm_fp32_t* x, const mllm_fp32_t* y, size_t n) {
  __elementwise(x, y, out, n, DivOp{});
}

//===----------------------------------------------------------------------===//
// Elementwise + - * / By Const
//===----------------------------------------------------------------------===//

template<typename T, typename Op>
HWY_INLINE void __elementwise_scalar(T* HWY_RESTRICT out, const T* HWY_RESTRICT x, const T y, size_t count, Op&& op) {
  const hn::ScalableTag<T> d;
  const size_t N = hn::Lanes(d);
  size_t idx = 0;

  const T scalar = y;
  const hn::Vec<decltype(d)> sVec = hn::Set(d, scalar);

  for (; idx + N <= count; idx += N) {
    const hn::Vec<decltype(d)> vx = hn::LoadU(d, x + idx);
    const hn::Vec<decltype(d)> result = op(d, vx, sVec);
    hn::StoreU(result, d, out + idx);
  }

  if (idx < count) {
    const hn::Vec<decltype(d)> vx = hn::LoadN(d, x + idx, count - idx);
    const hn::Vec<decltype(d)> result = op(d, vx, sVec);
    hn::StoreN(result, d, out + idx, count - idx);
  }
}

struct AddScalarOp {
  template<class D, class V>
  HWY_INLINE V operator()(D d, V a, V b) const {
    return hn::Add(a, b);
  }
};

struct SubScalarOp {
  template<class D, class V>
  HWY_INLINE V operator()(D d, V a, V b) const {
    return hn::Sub(a, b);
  }
};

struct MulScalarOp {
  template<class D, class V>
  HWY_INLINE V operator()(D d, V a, V b) const {
    return hn::Mul(a, b);
  }
};

struct DivScalarOp {
  template<class D, class V>
  HWY_INLINE V operator()(D d, V a, V b) const {
    return hn::Div(a, b);
  }
};

HWY_NOINLINE HWY_MAYBE_UNUSED void elewise_add_scalar_fp32(mllm_fp32_t* out, const mllm_fp32_t* x, const mllm_fp32_t y, size_t n) {
  __elementwise_scalar(out, x, y, n, AddScalarOp{});
}

HWY_NOINLINE HWY_MAYBE_UNUSED void elewise_sub_scalar_fp32(mllm_fp32_t* out, const mllm_fp32_t* x, const mllm_fp32_t y, size_t n) {
  __elementwise_scalar(out, x, y, n, SubScalarOp{});
}

HWY_NOINLINE HWY_MAYBE_UNUSED void elewise_mul_scalar_fp32(mllm_fp32_t* out, const mllm_fp32_t* x, const mllm_fp32_t y, size_t n) {
  __elementwise_scalar(out, x, y, n, MulScalarOp{});
}

HWY_NOINLINE HWY_MAYBE_UNUSED void elewise_div_scalar_fp32(mllm_fp32_t* out, const mllm_fp32_t* x, const mllm_fp32_t y, size_t n) {
  __elementwise_scalar(out, x, y, n, DivScalarOp{});
}

//===----------------------------------------------------------------------===//
// Inplace Elementwise + - * /
//
// 1. AddFrom
// 2. MulFrom
// 3. MulByConst
//===----------------------------------------------------------------------===//
// TODO

}  // namespace HWY_NAMESPACE
}  // namespace mllm::cpu::common
HWY_AFTER_NAMESPACE();
