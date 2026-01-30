// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <hwy/highway.h>
#include "mllm/core/DataTypes.hpp"

HWY_BEFORE_NAMESPACE();
namespace mllm::cpu::common {  // NOLINT
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE; 


struct ScalarAddOp { template<typename T> HWY_INLINE T operator()(T a, T b) const { return a + b; } };

struct ScalarSubOp { template<typename T> HWY_INLINE T operator()(T a, T b) const { return a - b; } };

struct ScalarMulOp { template<typename T> HWY_INLINE T operator()(T a, T b) const { return a * b; } };

struct ScalarDivOp { template<typename T> HWY_INLINE T operator()(T a, T b) const { return a / b; } };

struct ScalarMaxOp { template<typename T> HWY_INLINE T operator()(T a, T b) const { return a > b ? a : b; } };

struct ScalarMinOp { template<typename T> HWY_INLINE T operator()(T a, T b) const { return a < b ? a : b; } };

struct VecAddOp {
  template<class D, class V>
  HWY_INLINE V operator()(D d, V a, V b) const { return hn::Add(a, b); }
};

struct VecSubOp {
  template<class D, class V>
  HWY_INLINE V operator()(D d, V a, V b) const { return hn::Sub(a, b); }
};

struct VecMulOp {
  template<class D, class V>
  HWY_INLINE V operator()(D d, V a, V b) const { return hn::Mul(a, b); }
};

struct VecDivOp {
  template<class D, class V>
  HWY_INLINE V operator()(D d, V a, V b) const { return hn::Div(a, b); }
};

struct VecMaxOp {
  template<class D, class V>
  HWY_INLINE V operator()(D d, V a, V b) const { return hn::Max(a, b); }
};

struct VecMinOp {
  template<class D, class V>
  HWY_INLINE V operator()(D d, V a, V b) const { return hn::Min(a, b); }
};

struct VecSumReduce {
  template <class D, class V>
  HWY_INLINE hn::TFromD<D> operator()(D d, V v) const { return hn::ReduceSum(d, v); }
};


template<typename T, typename ScalarOp, typename VectorOp, typename VectorReduceOp>
HWY_INLINE T reduce_impl(const T* HWY_RESTRICT src, size_t src_stride, size_t size, 
                         ScalarOp&& scalar_op, VectorOp&& vec_op, VectorReduceOp&& vec_reduce_op) {
  if (size == 0) return T(0);

  const hn::ScalableTag<T> d;
  const size_t N = hn::Lanes(d);

  // SIMD fast path
  if (src_stride == 1 && size >= N) {
    using V = hn::Vec<decltype(d)>;

    // Init with first vector
    V vec_result = hn::LoadU(d, src);
    size_t i = N;

    // 4x unroll
    for (; i + 4 * N <= size; i += 4 * N) {
      const V v0 = hn::LoadU(d, src + i);
      const V v1 = hn::LoadU(d, src + i + N);
      const V v2 = hn::LoadU(d, src + i + 2 * N);
      const V v3 = hn::LoadU(d, src + i + 3 * N);

      vec_result = vec_op(d, vec_result, v0);
      vec_result = vec_op(d, vec_result, v1);
      vec_result = vec_op(d, vec_result, v2);
      vec_result = vec_op(d, vec_result, v3);
    }

    for (; i + N <= size; i += N) {
      const V v = hn::LoadU(d, src + i);
      vec_result = vec_op(d, vec_result, v);
    }

    if (i < size) {
      const V vt = hn::LoadN(d, src + i, size - i);
      vec_result = vec_op(d, vec_result, vt);
    }

    return vec_reduce_op(d, vec_result);
  }
  
  // Scalar path (stride != 1 or too small)
  T scalar_result = src[0];
  for (size_t i = 1; i < size; ++i) {
    scalar_result = scalar_op(scalar_result, src[i * src_stride]);
  }
  return scalar_result;

}


HWY_NOINLINE HWY_MAYBE_UNUSED void reduce_sum_fp32(mllm_fp32_t* dst,const mllm_fp32_t* src,
size_t src_stride, size_t size, int32_t thread_count) {
  const mllm_fp32_t v = reduce_impl<mllm_fp32_t>(src, src_stride, size,
      ScalarAddOp{}, VecAddOp{}, VecSumReduce{});
  *dst = v;
}


}  // namespace HWY_NAMESPACE
}  // namespace mllm::cpu::common
HWY_AFTER_NAMESPACE();
