// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

// Focused oracle for the single-token grouped-query-attention decode kernel
// that reads native KV-head [B, H, S, D] cache views.
//
// The reference below is an independent scalar implementation that accumulates
// in double precision. It is deliberately not routed through the production
// kernel, so the vectorized QK dot product, the grouped P@V accumulation, and
// the strided cache addressing cannot validate themselves. Output tolerance is
// used for the reference comparison because the vector dot product reorders the
// float32 summation; slice equivalence and repeat stability are compared
// bitwise because the kernel promises a fixed accumulation order per head.

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "mllm/backends/cpu/kernels/common/gqa_decode/fwd_bhsd.hpp"
#include "KernelTestHelper.hpp"

namespace gqa_decode_kernel_test {

using mllm::cpu::gqa_decode::BhsdStrides;
using mllm::cpu::gqa_decode::fwdBhsdFp32;

// Deterministic index-derived fill. No RNG, so every host reproduces the same
// bytes without carrying a seed through the evidence record.
inline float patternValue(std::size_t index, int salt) {
  const auto scaled = static_cast<float>((index * 37U + static_cast<unsigned>(salt) * 11U) % 251U);
  return (scaled - 125.0F) / 64.0F;
}

inline std::vector<float> makeBuffer(std::size_t count, int salt) {
  std::vector<float> buffer(count);
  for (std::size_t index = 0; index < count; ++index) { buffer[index] = patternValue(index, salt); }
  return buffer;
}

struct Geometry {
  int batch;
  int query_heads;
  int kv_heads;
  int kv_sequence;
  int qk_dim;
  int value_dim;
};

inline std::string describe(const Geometry& geometry) {
  return "B=" + std::to_string(geometry.batch) + " Hq=" + std::to_string(geometry.query_heads)
         + " Hkv=" + std::to_string(geometry.kv_heads) + " S=" + std::to_string(geometry.kv_sequence)
         + " D=" + std::to_string(geometry.qk_dim) + " Dv=" + std::to_string(geometry.value_dim);
}

// A float buffer together with the [B, H, S, D] strides the kernel uses to
// address it. The strides are part of the contract under test: the production
// caller hands the kernel views into a larger static cache and transposed
// query/output tensors, never freshly packed contiguous arrays.
struct StridedView {
  std::vector<float> storage;
  BhsdStrides strides;
};

inline std::size_t offset(const BhsdStrides& strides, int batch, int head, int row, int column) {
  return static_cast<std::size_t>(batch) * strides.batch + static_cast<std::size_t>(head) * strides.head
         + static_cast<std::size_t>(row) * strides.sequence + static_cast<std::size_t>(column) * strides.dimension;
}

// Contiguous [B, H, rows, dim] strides.
inline BhsdStrides contiguousStrides(int heads, int rows, int dim) { return {heads * rows * dim, rows * dim, dim, 1}; }

// Single-token query [B, Hq, 1, D] in contiguous memory.
inline StridedView makeContiguousQuery(const Geometry& geometry, int salt) {
  return {makeBuffer(static_cast<std::size_t>(geometry.batch) * geometry.query_heads * geometry.qk_dim, salt),
          contiguousStrides(geometry.query_heads, 1, geometry.qk_dim)};
}

// KV cache [B, Hkv, S, dim] whose sequence extent is exactly kv_sequence.
inline StridedView makeContiguousCache(const Geometry& geometry, int dim, int salt) {
  return {makeBuffer(static_cast<std::size_t>(geometry.batch) * geometry.kv_heads * geometry.kv_sequence * dim, salt),
          contiguousStrides(geometry.kv_heads, geometry.kv_sequence, dim)};
}

// Zero-filled single-token output [B, Hq, 1, Dv] in contiguous memory.
inline StridedView makeContiguousOutput(const Geometry& geometry) {
  return {std::vector<float>(static_cast<std::size_t>(geometry.batch) * geometry.query_heads * geometry.value_dim, 0.0F),
          contiguousStrides(geometry.query_heads, 1, geometry.value_dim)};
}

// The same cache rows placed inside a static cache [B, Hkv, capacity, dim]
// with capacity > kv_sequence, as nn::KVHeadStaticCache exposes them. Rows
// beyond kv_sequence hold unrelated pattern bytes that the kernel must never
// read.
inline StridedView makeStaticCacheView(const Geometry& geometry, int capacity, int dim, const StridedView& contiguous,
                                       int salt) {
  StridedView view{makeBuffer(static_cast<std::size_t>(geometry.batch) * geometry.kv_heads * capacity * dim, salt),
                   contiguousStrides(geometry.kv_heads, capacity, dim)};
  for (int batch = 0; batch < geometry.batch; ++batch) {
    for (int head = 0; head < geometry.kv_heads; ++head) {
      for (int row = 0; row < geometry.kv_sequence; ++row) {
        for (int column = 0; column < dim; ++column) {
          view.storage[offset(view.strides, batch, head, row, column)] =
              contiguous.storage[offset(contiguous.strides, batch, head, row, column)];
        }
      }
    }
  }
  return view;
}

// The same query bytes described as a transposed [B, 1, Hq, D] tensor: head
// stride D, batch and sequence stride Hq * D. With one token the memory order
// coincides with the contiguous query, so only the stride bookkeeping differs.
inline StridedView makeTransposedQueryView(const Geometry& geometry, const StridedView& contiguous_query) {
  return {contiguous_query.storage,
          {geometry.query_heads * geometry.qk_dim, geometry.qk_dim, geometry.query_heads * geometry.qk_dim, 1}};
}

inline StridedView makeTransposedOutputView(const Geometry& geometry) {
  return {std::vector<float>(static_cast<std::size_t>(geometry.batch) * geometry.query_heads * geometry.value_dim, 0.0F),
          {geometry.query_heads * geometry.value_dim, geometry.value_dim, geometry.query_heads * geometry.value_dim, 1}};
}

// The kernel's grouped scratch: group_size * kv_sequence probabilities.
inline std::vector<float> makeScratch(const Geometry& geometry) {
  return std::vector<float>(static_cast<std::size_t>(geometry.query_heads / geometry.kv_heads) * geometry.kv_sequence, 0.0F);
}

// Independent scalar reference: per query head, softmax(q . k / sqrt(D)) @ v
// with double accumulation, honoring the same strided addressing contract.
inline void referenceDecode(const Geometry& geometry, const StridedView& query, const StridedView& key,
                            const StridedView& value, std::vector<double>& output) {
  const int group_size = geometry.query_heads / geometry.kv_heads;
  const double scale = 1.0 / std::sqrt(static_cast<double>(geometry.qk_dim));
  output.assign(static_cast<std::size_t>(geometry.batch) * geometry.query_heads * geometry.value_dim, 0.0);
  std::vector<double> scores(static_cast<std::size_t>(geometry.kv_sequence));
  for (int batch = 0; batch < geometry.batch; ++batch) {
    for (int head = 0; head < geometry.query_heads; ++head) {
      const int kv_head = head / group_size;
      double maximum = -std::numeric_limits<double>::infinity();
      for (int row = 0; row < geometry.kv_sequence; ++row) {
        double score = 0.0;
        for (int column = 0; column < geometry.qk_dim; ++column) {
          score += static_cast<double>(query.storage[offset(query.strides, batch, head, 0, column)])
                   * static_cast<double>(key.storage[offset(key.strides, batch, kv_head, row, column)]);
        }
        scores[static_cast<std::size_t>(row)] = score * scale;
        maximum = std::max(maximum, scores[static_cast<std::size_t>(row)]);
      }
      double denominator = 0.0;
      for (int row = 0; row < geometry.kv_sequence; ++row) {
        scores[static_cast<std::size_t>(row)] = std::exp(scores[static_cast<std::size_t>(row)] - maximum);
        denominator += scores[static_cast<std::size_t>(row)];
      }
      auto* out = output.data() + (static_cast<std::size_t>(batch) * geometry.query_heads + head) * geometry.value_dim;
      for (int row = 0; row < geometry.kv_sequence; ++row) {
        const double probability = scores[static_cast<std::size_t>(row)] / denominator;
        for (int column = 0; column < geometry.value_dim; ++column) {
          out[column] += probability * static_cast<double>(value.storage[offset(value.strides, batch, kv_head, row, column)]);
        }
      }
    }
  }
}

constexpr float kReferenceTolerance = 1.0e-4F;

inline void expectMatchesReference(const Geometry& geometry, const StridedView& output, const std::vector<double>& reference,
                                   const std::string& label) {
  for (int batch = 0; batch < geometry.batch; ++batch) {
    for (int head = 0; head < geometry.query_heads; ++head) {
      for (int column = 0; column < geometry.value_dim; ++column) {
        const auto expected =
            reference[(static_cast<std::size_t>(batch) * geometry.query_heads + head) * geometry.value_dim + column];
        const auto actual = output.storage[offset(output.strides, batch, head, 0, column)];
        ASSERT_NEAR(actual, expected, kReferenceTolerance)
            << label << " " << describe(geometry) << " b=" << batch << " h=" << head << " d=" << column;
      }
    }
  }
}

inline void testMatchesScalarReference(const std::vector<Geometry>& geometries) {
  for (const auto& geometry : geometries) {
    const auto query = makeContiguousQuery(geometry, 1);
    const auto key = makeContiguousCache(geometry, geometry.qk_dim, 2);
    const auto value = makeContiguousCache(geometry, geometry.value_dim, 3);
    auto output = makeContiguousOutput(geometry);
    auto scratch = makeScratch(geometry);

    ASSERT_TRUE(fwdBhsdFp32(geometry.batch, geometry.query_heads, geometry.kv_heads, geometry.kv_sequence, geometry.qk_dim,
                            geometry.value_dim, query.storage.data(), query.strides, key.storage.data(), key.strides,
                            value.storage.data(), value.strides, output.storage.data(), output.strides, scratch.data()))
        << describe(geometry);

    std::vector<double> reference;
    referenceDecode(geometry, query, key, value, reference);
    expectMatchesReference(geometry, output, reference, "contiguous");
  }
}

// The production consumer hands the kernel a KV view inside a larger static
// cache and a transposed query/output view. Both must be addressed through the
// strides, and the result must be bitwise identical to the contiguous
// computation because the same bytes are read in the same order.
inline void testNativeCacheAndTransposedQueryStrides(const Geometry& geometry, int cache_capacity) {
  ASSERT_GT(cache_capacity, geometry.kv_sequence);
  const auto contiguous_query = makeContiguousQuery(geometry, 1);
  const auto contiguous_key = makeContiguousCache(geometry, geometry.qk_dim, 2);
  const auto contiguous_value = makeContiguousCache(geometry, geometry.value_dim, 3);
  auto contiguous_output = makeContiguousOutput(geometry);
  auto scratch = makeScratch(geometry);
  ASSERT_TRUE(fwdBhsdFp32(geometry.batch, geometry.query_heads, geometry.kv_heads, geometry.kv_sequence, geometry.qk_dim,
                          geometry.value_dim, contiguous_query.storage.data(), contiguous_query.strides,
                          contiguous_key.storage.data(), contiguous_key.strides, contiguous_value.storage.data(),
                          contiguous_value.strides, contiguous_output.storage.data(), contiguous_output.strides,
                          scratch.data()))
      << describe(geometry);

  const auto query_view = makeTransposedQueryView(geometry, contiguous_query);
  const auto key_view = makeStaticCacheView(geometry, cache_capacity, geometry.qk_dim, contiguous_key, 7);
  const auto value_view = makeStaticCacheView(geometry, cache_capacity, geometry.value_dim, contiguous_value, 8);
  auto output_view = makeTransposedOutputView(geometry);
  ASSERT_TRUE(fwdBhsdFp32(geometry.batch, geometry.query_heads, geometry.kv_heads, geometry.kv_sequence, geometry.qk_dim,
                          geometry.value_dim, query_view.storage.data(), query_view.strides, key_view.storage.data(),
                          key_view.strides, value_view.storage.data(), value_view.strides, output_view.storage.data(),
                          output_view.strides, scratch.data()))
      << describe(geometry);

  std::vector<double> reference;
  referenceDecode(geometry, query_view, key_view, value_view, reference);
  expectMatchesReference(geometry, output_view, reference, "strided");
  for (int batch = 0; batch < geometry.batch; ++batch) {
    for (int head = 0; head < geometry.query_heads; ++head) {
      for (int column = 0; column < geometry.value_dim; ++column) {
        ASSERT_EQ(output_view.storage[offset(output_view.strides, batch, head, 0, column)],
                  contiguous_output.storage[offset(contiguous_output.strides, batch, head, 0, column)])
            << describe(geometry) << " b=" << batch << " h=" << head << " d=" << column;
      }
    }
  }
}

// Every (batch, kv-head) group is scheduled independently and every query head
// accumulates its keys in increasing order, so a grouped call must reproduce
// the per-head single-KV-head call bitwise. This also proves the grouped
// scratch rows do not leak between heads or batches.
inline void testGroupedSlicesMatchSingleHeadCallsBitwise(const Geometry& geometry) {
  const auto query = makeContiguousQuery(geometry, 1);
  const auto key = makeContiguousCache(geometry, geometry.qk_dim, 2);
  const auto value = makeContiguousCache(geometry, geometry.value_dim, 3);
  auto grouped_output = makeContiguousOutput(geometry);
  auto grouped_scratch = makeScratch(geometry);
  ASSERT_TRUE(fwdBhsdFp32(geometry.batch, geometry.query_heads, geometry.kv_heads, geometry.kv_sequence, geometry.qk_dim,
                          geometry.value_dim, query.storage.data(), query.strides, key.storage.data(), key.strides,
                          value.storage.data(), value.strides, grouped_output.storage.data(), grouped_output.strides,
                          grouped_scratch.data()))
      << describe(geometry);

  const int group_size = geometry.query_heads / geometry.kv_heads;
  const Geometry single{1, 1, 1, geometry.kv_sequence, geometry.qk_dim, geometry.value_dim};
  const auto key_head_size = static_cast<std::size_t>(geometry.kv_sequence) * geometry.qk_dim;
  const auto value_head_size = static_cast<std::size_t>(geometry.kv_sequence) * geometry.value_dim;
  for (int batch = 0; batch < geometry.batch; ++batch) {
    for (int head = 0; head < geometry.query_heads; ++head) {
      const int kv_head = head / group_size;
      // Slice the grouped operands down to one query head and its KV head.
      const auto* query_head = query.storage.data() + offset(query.strides, batch, head, 0, 0);
      const auto* key_head = key.storage.data() + offset(key.strides, batch, kv_head, 0, 0);
      const auto* value_head = value.storage.data() + offset(value.strides, batch, kv_head, 0, 0);
      const StridedView single_query{std::vector<float>(query_head, query_head + geometry.qk_dim),
                                     contiguousStrides(1, 1, geometry.qk_dim)};
      const StridedView single_key{std::vector<float>(key_head, key_head + key_head_size),
                                   contiguousStrides(1, geometry.kv_sequence, geometry.qk_dim)};
      const StridedView single_value{std::vector<float>(value_head, value_head + value_head_size),
                                     contiguousStrides(1, geometry.kv_sequence, geometry.value_dim)};
      auto single_output = makeContiguousOutput(single);
      auto single_scratch = makeScratch(single);
      ASSERT_TRUE(fwdBhsdFp32(single.batch, single.query_heads, single.kv_heads, single.kv_sequence, single.qk_dim,
                              single.value_dim, single_query.storage.data(), single_query.strides, single_key.storage.data(),
                              single_key.strides, single_value.storage.data(), single_value.strides,
                              single_output.storage.data(), single_output.strides, single_scratch.data()))
          << describe(geometry);
      for (int column = 0; column < geometry.value_dim; ++column) {
        ASSERT_EQ(grouped_output.storage[offset(grouped_output.strides, batch, head, 0, column)],
                  single_output.storage[static_cast<std::size_t>(column)])
            << describe(geometry) << " b=" << batch << " h=" << head << " d=" << column;
      }
    }
  }
}

inline void testRepeatedCallsAreBitwiseStable(const Geometry& geometry, int repeats) {
  const auto query = makeContiguousQuery(geometry, 1);
  const auto key = makeContiguousCache(geometry, geometry.qk_dim, 2);
  const auto value = makeContiguousCache(geometry, geometry.value_dim, 3);
  std::vector<float> first;
  for (int repeat = 0; repeat <= repeats; ++repeat) {
    auto output = makeContiguousOutput(geometry);
    auto scratch = makeScratch(geometry);
    ASSERT_TRUE(fwdBhsdFp32(geometry.batch, geometry.query_heads, geometry.kv_heads, geometry.kv_sequence, geometry.qk_dim,
                            geometry.value_dim, query.storage.data(), query.strides, key.storage.data(), key.strides,
                            value.storage.data(), value.strides, output.storage.data(), output.strides, scratch.data()))
        << describe(geometry);
    if (repeat == 0) {
      first = output.storage;
    } else {
      ASSERT_EQ(output.storage, first) << describe(geometry) << " repeat=" << repeat;
    }
  }
}

// Invalid geometry, null buffers, and unsupported strides must be rejected
// before any byte of the output is touched; the backend op then takes the
// reference path.
inline void testRejectsInvalidGeometryAndStrides() {
  const Geometry geometry{1, 4, 2, 3, 8, 8};
  const auto query = makeContiguousQuery(geometry, 1);
  const auto key = makeContiguousCache(geometry, geometry.qk_dim, 2);
  const auto value = makeContiguousCache(geometry, geometry.value_dim, 3);
  auto output = makeContiguousOutput(geometry);
  auto scratch = makeScratch(geometry);
  ASSERT_TRUE(fwdBhsdFp32(geometry.batch, geometry.query_heads, geometry.kv_heads, geometry.kv_sequence, geometry.qk_dim,
                          geometry.value_dim, query.storage.data(), query.strides, key.storage.data(), key.strides,
                          value.storage.data(), value.strides, output.storage.data(), output.strides, scratch.data()));

  constexpr float kSentinel = 123.5F;
  const auto expectRejected = [&](const std::string& label, int query_heads, int kv_sequence, int qk_dim,
                                  const float* query_data, const float* key_data, BhsdStrides key_strides,
                                  const float* value_data, BhsdStrides value_strides, BhsdStrides output_strides,
                                  float* scratch_data) {
    std::vector<float> untouched(output.storage.size(), kSentinel);
    EXPECT_FALSE(fwdBhsdFp32(geometry.batch, query_heads, geometry.kv_heads, kv_sequence, qk_dim, geometry.value_dim,
                             query_data, query.strides, key_data, key_strides, value_data, value_strides, untouched.data(),
                             output_strides, scratch_data))
        << label;
    for (const auto element : untouched) { EXPECT_EQ(element, kSentinel) << label << " touched the output"; }
  };
  const auto* query_data = query.storage.data();
  const auto* key_data = key.storage.data();
  const auto* value_data = value.storage.data();
  BhsdStrides non_unit_dimension = key.strides;
  non_unit_dimension.dimension = 2;
  BhsdStrides non_unit_output_dimension = output.strides;
  non_unit_output_dimension.dimension = 2;
  BhsdStrides zero_sequence = value.strides;
  zero_sequence.sequence = 0;

  expectRejected("query heads not a multiple of kv heads", 3, geometry.kv_sequence, geometry.qk_dim, query_data, key_data,
                 key.strides, value_data, value.strides, output.strides, scratch.data());
  expectRejected("zero kv sequence", geometry.query_heads, 0, geometry.qk_dim, query_data, key_data, key.strides, value_data,
                 value.strides, output.strides, scratch.data());
  expectRejected("zero qk dim", geometry.query_heads, geometry.kv_sequence, 0, query_data, key_data, key.strides, value_data,
                 value.strides, output.strides, scratch.data());
  expectRejected("null query", geometry.query_heads, geometry.kv_sequence, geometry.qk_dim, nullptr, key_data, key.strides,
                 value_data, value.strides, output.strides, scratch.data());
  expectRejected("null key", geometry.query_heads, geometry.kv_sequence, geometry.qk_dim, query_data, nullptr, key.strides,
                 value_data, value.strides, output.strides, scratch.data());
  expectRejected("null value", geometry.query_heads, geometry.kv_sequence, geometry.qk_dim, query_data, key_data, key.strides,
                 nullptr, value.strides, output.strides, scratch.data());
  expectRejected("null scratch", geometry.query_heads, geometry.kv_sequence, geometry.qk_dim, query_data, key_data, key.strides,
                 value_data, value.strides, output.strides, nullptr);
  expectRejected("non-unit key dimension stride", geometry.query_heads, geometry.kv_sequence, geometry.qk_dim, query_data,
                 key_data, non_unit_dimension, value_data, value.strides, output.strides, scratch.data());
  expectRejected("non-unit output dimension stride", geometry.query_heads, geometry.kv_sequence, geometry.qk_dim, query_data,
                 key_data, key.strides, value_data, value.strides, non_unit_output_dimension, scratch.data());
  expectRejected("non-positive value sequence stride", geometry.query_heads, geometry.kv_sequence, geometry.qk_dim, query_data,
                 key_data, key.strides, value_data, zero_sequence, output.strides, scratch.data());
}

}  // namespace gqa_decode_kernel_test

class GqaDecodeKernelTest : public KernelTest {
 public:
  void testMatchesScalarReference(const std::vector<gqa_decode_kernel_test::Geometry>& geometries) {
    gqa_decode_kernel_test::testMatchesScalarReference(geometries);
  }

  void testNativeCacheAndTransposedQueryStrides(const gqa_decode_kernel_test::Geometry& geometry, int cache_capacity) {
    gqa_decode_kernel_test::testNativeCacheAndTransposedQueryStrides(geometry, cache_capacity);
  }

  void testGroupedSlicesMatchSingleHeadCallsBitwise(const gqa_decode_kernel_test::Geometry& geometry) {
    gqa_decode_kernel_test::testGroupedSlicesMatchSingleHeadCallsBitwise(geometry);
  }

  void testRepeatedCallsAreBitwiseStable(const gqa_decode_kernel_test::Geometry& geometry, int repeats) {
    gqa_decode_kernel_test::testRepeatedCallsAreBitwiseStable(geometry, repeats);
  }

  void testRejectsInvalidGeometryAndStrides() { gqa_decode_kernel_test::testRejectsInvalidGeometryAndStrides(); }
};
