// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <hwy/highway.h>
#include "mllm/core/DataTypes.hpp"

HWY_BEFORE_NAMESPACE();
namespace mllm::cpu::common {  // NOLINT
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;
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

//===----------------------------------------------------------------------===//
// Elementwise + - * / By Matrix
//===----------------------------------------------------------------------===//
template<typename T, typename Op>
HWY_INLINE void elementwise_impl(const T* HWY_RESTRICT x, const T* HWY_RESTRICT y, T* HWY_RESTRICT out, size_t count, Op&& op) {
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

HWY_NOINLINE HWY_MAYBE_UNUSED void elewise_add_fp32(mllm_fp32_t* out, const mllm_fp32_t* x, const mllm_fp32_t* y, size_t n) {
  elementwise_impl(x, y, out, n, AddOp{});
}

HWY_NOINLINE HWY_MAYBE_UNUSED void elewise_sub_fp32(mllm_fp32_t* out, const mllm_fp32_t* x, const mllm_fp32_t* y, size_t n) {
  elementwise_impl(x, y, out, n, SubOp{});
}

HWY_NOINLINE HWY_MAYBE_UNUSED void elewise_mul_fp32(mllm_fp32_t* out, const mllm_fp32_t* x, const mllm_fp32_t* y, size_t n) {
  elementwise_impl(x, y, out, n, MulOp{});
}

HWY_NOINLINE HWY_MAYBE_UNUSED void elewise_div_fp32(mllm_fp32_t* out, const mllm_fp32_t* x, const mllm_fp32_t* y, size_t n) {
  elementwise_impl(x, y, out, n, DivOp{});
}


// HWY_NOINLINE HWY_MAYBE_UNUSED void elewise_add_fp16(mllm_fp16_t* out, const mllm_fp16_t* x, const mllm_fp16_t* y, size_t n) {
//   elementwise_impl(x, y, out, n, AddOp{});
// }

// HWY_NOINLINE HWY_MAYBE_UNUSED void elewise_sub_fp16(mllm_fp16_t* out, const mllm_fp16_t* x, const mllm_fp16_t* y, size_t n) {
//   elementwise_impl(x, y, out, n, SubOp{});
// }

// HWY_NOINLINE HWY_MAYBE_UNUSED void elewise_mul_fp16(mllm_fp16_t* out, const mllm_fp16_t* x, const mllm_fp16_t* y, size_t n) {
//   elementwise_impl(x, y, out, n, MulOp{});
// }

// HWY_NOINLINE HWY_MAYBE_UNUSED void elewise_div_fp16(mllm_fp16_t* out, const mllm_fp16_t* x, const mllm_fp16_t* y, size_t n) {
//   elementwise_impl(x, y, out, n, DivOp{});
// }


HWY_NOINLINE HWY_MAYBE_UNUSED void elewise_add_int32(mllm_int32_t* out, const mllm_int32_t* x, const mllm_int32_t* y, size_t n) {
  elementwise_impl(x, y, out, n, AddOp{});
}

HWY_NOINLINE HWY_MAYBE_UNUSED void elewise_sub_int32(mllm_int32_t* out, const mllm_int32_t* x, const mllm_int32_t* y, size_t n) {
  elementwise_impl(x, y, out, n, SubOp{});
}

HWY_NOINLINE HWY_MAYBE_UNUSED void elewise_mul_int32(mllm_int32_t* out, const mllm_int32_t* x, const mllm_int32_t* y, size_t n) {
  elementwise_impl(x, y, out, n, MulOp{});
}

HWY_NOINLINE HWY_MAYBE_UNUSED void elewise_div_int32(mllm_int32_t* out, const mllm_int32_t* x, const mllm_int32_t* y, size_t n) {
  elementwise_impl(x, y, out, n, DivOp{});
}


HWY_NOINLINE HWY_MAYBE_UNUSED void elewise_add_int16(mllm_int16_t* out, const mllm_int16_t* x, const mllm_int16_t* y, size_t n) {
  elementwise_impl(x, y, out, n, AddOp{});
}

HWY_NOINLINE HWY_MAYBE_UNUSED void elewise_sub_int16(mllm_int16_t* out, const mllm_int16_t* x, const mllm_int16_t* y, size_t n) {
  elementwise_impl(x, y, out, n, SubOp{});
}

HWY_NOINLINE HWY_MAYBE_UNUSED void elewise_mul_int16(mllm_int16_t* out, const mllm_int16_t* x, const mllm_int16_t* y, size_t n) {
  elementwise_impl(x, y, out, n, MulOp{});
}

// HWY_NOINLINE HWY_MAYBE_UNUSED void elewise_div_int16(mllm_int16_t* out, const mllm_int16_t* x, const mllm_int16_t* y, size_t n) {
//   elementwise_impl(x, y, out, n, DivOp{});
// }


HWY_NOINLINE HWY_MAYBE_UNUSED void elewise_add_int8(mllm_int8_t* out, const mllm_int8_t* x, const mllm_int8_t* y, size_t n) {
  elementwise_impl(x, y, out, n, AddOp{});
}

HWY_NOINLINE HWY_MAYBE_UNUSED void elewise_sub_int8(mllm_int8_t* out, const mllm_int8_t* x, const mllm_int8_t* y, size_t n) {
  elementwise_impl(x, y, out, n, SubOp{});
}

HWY_NOINLINE HWY_MAYBE_UNUSED void elewise_mul_int8(mllm_int8_t* out, const mllm_int8_t* x, const mllm_int8_t* y, size_t n) {
  elementwise_impl(x, y, out, n, MulOp{});
}

// HWY_NOINLINE HWY_MAYBE_UNUSED void elewise_div_int8(mllm_int8_t* out, const mllm_int8_t* x, const mllm_int8_t* y, size_t n) {
//   elementwise_impl(x, y, out, n, DivOp{});
// }


//===----------------------------------------------------------------------===//
// Elementwise + - * / By Const
//===----------------------------------------------------------------------===//

template<typename T, typename Op>
HWY_INLINE void elementwise_scl_impl(T* HWY_RESTRICT out, const T* HWY_RESTRICT x, const T y, size_t count, Op&& op) {
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

HWY_NOINLINE HWY_MAYBE_UNUSED void elewise_add_scl_fp32(mllm_fp32_t* out, const mllm_fp32_t* x, const mllm_fp32_t y, size_t n) {
  elementwise_scl_impl(out, x, y, n, AddOp{});
}

HWY_NOINLINE HWY_MAYBE_UNUSED void elewise_sub_scl_fp32(mllm_fp32_t* out, const mllm_fp32_t* x, const mllm_fp32_t y, size_t n) {
  elementwise_scl_impl(out, x, y, n, SubOp{});
}

HWY_NOINLINE HWY_MAYBE_UNUSED void elewise_mul_scl_fp32(mllm_fp32_t* out, const mllm_fp32_t* x, const mllm_fp32_t y, size_t n) {
  elementwise_scl_impl(out, x, y, n, MulOp{});
}

HWY_NOINLINE HWY_MAYBE_UNUSED void elewise_div_scl_fp32(mllm_fp32_t* out, const mllm_fp32_t* x, const mllm_fp32_t y, size_t n) {
  elementwise_scl_impl(out, x, y, n, DivOp{});
}


// HWY_NOINLINE HWY_MAYBE_UNUSED void elewise_add_scl_fp16(mllm_fp16_t* out, const mllm_fp16_t* x, const mllm_fp16_t y, size_t n) {
//   elementwise_scl_impl(out, x, y, n, AddOp{});
// }

// HWY_NOINLINE HWY_MAYBE_UNUSED void elewise_sub_scl_fp16(mllm_fp16_t* out, const mllm_fp16_t* x, const mllm_fp16_t y, size_t n) {
//   elementwise_scl_impl(out, x, y, n, SubOp{});
// }

// HWY_NOINLINE HWY_MAYBE_UNUSED void elewise_mul_scl_fp16(mllm_fp16_t* out, const mllm_fp16_t* x, const mllm_fp16_t y, size_t n) {
//   elementwise_scl_impl(out, x, y, n, MulOp{});
// }

// HWY_NOINLINE HWY_MAYBE_UNUSED void elewise_div_scl_fp16(mllm_fp16_t* out, const mllm_fp16_t* x, const mllm_fp16_t y, size_t n) {
//   elementwise_scl_impl(out, x, y, n, DivOp{});
// }


HWY_NOINLINE HWY_MAYBE_UNUSED void elewise_add_scl_int32(mllm_int32_t* out, const mllm_int32_t* x, const mllm_int32_t y, size_t n) {
  elementwise_scl_impl(out, x, y, n, AddOp{});
}

HWY_NOINLINE HWY_MAYBE_UNUSED void elewise_sub_scl_int32(mllm_int32_t* out, const mllm_int32_t* x, const mllm_int32_t y, size_t n) {
  elementwise_scl_impl(out, x, y, n, SubOp{});
}

HWY_NOINLINE HWY_MAYBE_UNUSED void elewise_mul_scl_int32(mllm_int32_t* out, const mllm_int32_t* x, const mllm_int32_t y, size_t n) {
  elementwise_scl_impl(out, x, y, n, MulOp{});
}

HWY_NOINLINE HWY_MAYBE_UNUSED void elewise_div_scl_int32(mllm_int32_t* out, const mllm_int32_t* x, const mllm_int32_t y, size_t n) {
  elementwise_scl_impl(out, x, y, n, DivOp{});
}


HWY_NOINLINE HWY_MAYBE_UNUSED void elewise_add_scl_int16(mllm_int16_t* out, const mllm_int16_t* x, const mllm_int16_t y, size_t n) {
  elementwise_scl_impl(out, x, y, n, AddOp{});
}

HWY_NOINLINE HWY_MAYBE_UNUSED void elewise_sub_scl_int16(mllm_int16_t* out, const mllm_int16_t* x, const mllm_int16_t y, size_t n) {
  elementwise_scl_impl(out, x, y, n, SubOp{});
}

HWY_NOINLINE HWY_MAYBE_UNUSED void elewise_mul_scl_int16(mllm_int16_t* out, const mllm_int16_t* x, const mllm_int16_t y, size_t n) {
  elementwise_scl_impl(out, x, y, n, MulOp{});
}

HWY_NOINLINE HWY_MAYBE_UNUSED void elewise_div_scl_int16(mllm_int16_t* out, const mllm_int16_t* x, const mllm_int16_t y, size_t n) {
  elementwise_scl_impl(out, x, y, n, DivOp{});
}


HWY_NOINLINE HWY_MAYBE_UNUSED void elewise_add_scl_int8(mllm_int8_t* out, const mllm_int8_t* x, const mllm_int8_t y, size_t n) {
  elementwise_scl_impl(out, x, y, n, AddOp{});
}

HWY_NOINLINE HWY_MAYBE_UNUSED void elewise_sub_scl_int8(mllm_int8_t* out, const mllm_int8_t* x, const mllm_int8_t y, size_t n) {
  elementwise_scl_impl(out, x, y, n, SubOp{});
}

HWY_NOINLINE HWY_MAYBE_UNUSED void elewise_mul_scl_int8(mllm_int8_t* out, const mllm_int8_t* x, const mllm_int8_t y, size_t n) {
  elementwise_scl_impl(out, x, y, n, MulOp{});
}

HWY_NOINLINE HWY_MAYBE_UNUSED void elewise_div_scl_int8(mllm_int8_t* out, const mllm_int8_t* x, const mllm_int8_t y, size_t n) {
  elementwise_scl_impl(out, x, y, n, DivOp{});
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
