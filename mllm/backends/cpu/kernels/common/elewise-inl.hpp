// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <hwy/highway.h>
#include <hwy/contrib/math/math-inl.h>
#include <hwy/contrib/algo/transform-inl.h>

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

template<typename T>
HWY_NOINLINE HWY_MAYBE_UNUSED void element_wise_add(T* x, const T* y, size_t n) {
  __elementwise(x, y, n, AddOp{});
}

template<typename T>
HWY_NOINLINE HWY_MAYBE_UNUSED void element_wise_sub(T* x, const T* y, size_t n) {
  __elementwise(x, y, n, SubOp{});
}

template<typename T>
HWY_NOINLINE HWY_MAYBE_UNUSED void element_wise_mul(T* x, const T* y, size_t n) {
  __elementwise(x, y, n, MulOp{});
}

template<typename T>
HWY_NOINLINE HWY_MAYBE_UNUSED void element_wise_div(T* x, const T* y, size_t n) {
  __elementwise(x, y, n, DivOp{});
}

//===----------------------------------------------------------------------===//
// Elementwise + - * / By Const
//===----------------------------------------------------------------------===//

// TODO

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
