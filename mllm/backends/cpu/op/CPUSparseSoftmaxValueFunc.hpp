#ifndef CPU_SPARSE_SOFTMAX_VALUE_FUNC_HPP
#define CPU_SPARSE_SOFTMAX_VALUE_FUNC_HPP

#include "../CPUBackend.hpp"
#include "DataType.hpp"
#include "Tensor.hpp"
#include "Types.hpp"
#include "backends/cpu/third_party/ggml/Quantize.hpp"
#include "backends/cpu/third_party/ggml/VecDotFP16.hpp"
#include "backends/cpu/third_party/ggml/VecDotFP32.hpp"
#include "../AttentionProfiler.hpp"
#include "../HMXInt8Selector.hpp"
#include "CPUAttentionValueLayout.hpp"
#include "../compute/ActivationFunction.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <omp.h>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace mllm {

// Element-wise (unstructured) sparse attention for CPU prefill. QK logits are
// still dense; this op selects the largest causal-valid elements in each row,
// renormalizes them, and directly accumulates V without materializing a dense P.
class CPUSparseSoftmaxValueFunc : public Op {
protected:
    struct ThreadScratch {
        std::vector<float> rank_scores;
        std::vector<float> weights;
        std::vector<int32_t> selected_indices;
        std::vector<uint64_t> random_bits;
        std::vector<mllm_fp16_t> weights_f16;
        std::vector<mllm_fp16_t> query_f16;
        std::vector<float> dense_output;

        void resize(int capacity) {
            rank_scores.resize(capacity);
            weights.resize(capacity);
            selected_indices.resize(capacity);
            random_bits.resize((capacity + 63) / 64);
            weights_f16.resize(capacity);
            query_f16.resize(capacity);
            dense_output.resize(capacity);
        }
    };

    struct PackedValueState {
        struct HeadBuffer {
            std::unique_ptr<mllm_fp16_t[]> data;
            size_t capacity = 0;

            void reserve(size_t required, size_t preserve) {
                if (required <= capacity) return;
                assert(preserve <= capacity);
                const size_t grown_capacity = capacity == 0
                    ? required : capacity + std::max<size_t>(1, capacity / 2);
                const size_t new_capacity = std::max(required, grown_capacity);
                std::unique_ptr<mllm_fp16_t[]> replacement(
                    new mllm_fp16_t[new_capacity]);
                if (data && preserve > 0) {
                    std::memcpy(replacement.get(), data.get(),
                                preserve * sizeof(mllm_fp16_t));
                }
                data = std::move(replacement);
                capacity = new_capacity;
            }
        };

        std::weak_ptr<Tensor> source;
        int batch = 0;
        int heads = 0;
        int head_dim = 0;
        int src_stride = 0;
        const void *logical_base = nullptr;
        int packed_key_len = 0;
        std::vector<HeadBuffer> head_data;

        void reset(const std::shared_ptr<Tensor> &new_source,
                   int new_batch, int new_heads, int new_head_dim,
                   int new_src_stride, const void *new_logical_base) {
            source = new_source;
            batch = new_batch;
            heads = new_heads;
            head_dim = new_head_dim;
            src_stride = new_src_stride;
            logical_base = new_logical_base;
            packed_key_len = 0;
            head_data.clear();
            head_data.resize(static_cast<size_t>(batch) * heads);
        }
    };

    int thread_count_ = 4;
    float sparsity_ = 0.0F;
    bool causal_mask_ = true;
    int topk_sample_size_ = 0;
    int sampled_candidate_keep_ = 0;
    std::vector<float> head_retentions_;
    int pack_reserve_tokens_ = 0;
    int last_key_pack_begin_ = -1;
    int last_key_pack_end_ = -1;
    int scratch_capacity_ = 0;
    std::vector<ThreadScratch> thread_scratch_;
    std::unordered_map<const Tensor *, PackedValueState> packed_values_;
    std::unordered_map<const Tensor *, PackedValueState> packed_keys_;

    struct SparseQualityDiagnosticConfig {
        bool enabled = false;
        int target_begin = -1;
        int target_end = -1;
    };

    static SparseQualityDiagnosticConfig sparseQualityDiagnosticConfig() {
        SparseQualityDiagnosticConfig config;
        const char *active = std::getenv(
            "MLLM_SPARSE_QUALITY_DIAGNOSTICS_ACTIVE");
        config.enabled = active != nullptr && active[0] == '1'
            && active[1] == '\0';
        if (!config.enabled) return config;
        const auto parse_index = [](const char *name) {
            const char *value = std::getenv(name);
            if (value == nullptr || value[0] == '\0') return -1;
            char *end = nullptr;
            const long parsed = std::strtol(value, &end, 10);
            if (end == value || *end != '\0' || parsed < 0
                || parsed > std::numeric_limits<int>::max()) {
                throw std::runtime_error(
                    std::string("invalid sparse quality diagnostic index: ")
                    + name);
            }
            return static_cast<int>(parsed);
        };
        config.target_begin = parse_index(
            "MLLM_SPARSE_DIAGNOSTIC_TARGET_BEGIN");
        config.target_end = parse_index(
            "MLLM_SPARSE_DIAGNOSTIC_TARGET_END");
        if ((config.target_begin < 0) != (config.target_end < 0)
            || (config.target_begin >= 0
                && config.target_end <= config.target_begin)) {
            throw std::runtime_error(
                "sparse quality diagnostic target range is invalid");
        }
        return config;
    }

    static std::mutex &sparseQualityDiagnosticPrintMutex() {
        static auto *mutex = new std::mutex();
        return *mutex;
    }

    static int sampledCandidateKeep(int sample_count, float density) {
        if (sample_count <= 0) return 0;
        if (density <= 0.0F) return 1;
        if (density >= 1.0F) return sample_count;
        // Select the smallest sample rank r for which
        // P[Binomial(sample_count, density) >= r] <= 1%.  If the sampled
        // cutoff is still too aggressive, the row-level exact fallback keeps
        // results identical; this quantile only controls candidate work.
        constexpr long double alpha = 0.01L;
        if (sample_count <= 4096) {
            const long double p = static_cast<long double>(density);
            const int mode = std::min(sample_count, static_cast<int>(
                std::floor((sample_count + 1) * p)));
            std::vector<long double> probabilities(sample_count + 1, 0.0L);
            const long double log_mode_probability =
                std::lgamma(static_cast<long double>(sample_count + 1))
                - std::lgamma(static_cast<long double>(mode + 1))
                - std::lgamma(static_cast<long double>(sample_count - mode + 1))
                + mode * std::log(p)
                + (sample_count - mode) * std::log1p(-p);
            probabilities[mode] = std::exp(log_mode_probability);
            for (int k = mode; k > 0; --k) {
                probabilities[k - 1] = probabilities[k]
                    * static_cast<long double>(k)
                    / static_cast<long double>(sample_count - k + 1)
                    * (1.0L - p) / p;
            }
            for (int k = mode; k < sample_count; ++k) {
                probabilities[k + 1] = probabilities[k]
                    * static_cast<long double>(sample_count - k)
                    / static_cast<long double>(k + 1)
                    * p / (1.0L - p);
            }
            long double total = 0.0L;
            for (const long double probability : probabilities) {
                total += probability;
            }
            long double cumulative = 0.0L;
            for (int k = 0; k <= sample_count; ++k) {
                cumulative += probabilities[k];
                if (cumulative >= (1.0L - alpha) * total) {
                    return std::min(sample_count, k + 1);
                }
            }
            return sample_count;
        }

        // Avoid a large temporary vector for unreasonable sample sizes.
        constexpr float z = 2.3263479F;
        const float n = static_cast<float>(sample_count);
        const float z_squared = z * z;
        const float candidate_density = (
            density + z_squared / (2.0F * n)
            + z * std::sqrt(density * (1.0F - density) / n
                + z_squared / (4.0F * n * n)))
            / (1.0F + z_squared / n);
        return std::max(1, std::min(sample_count, static_cast<int>(
            std::ceil(candidate_density * (n + 1.0F)))));
    }

    void ensureScratchCapacity(int required, int required_threads = 0) {
        if (required_threads <= 0) required_threads = thread_count_;
        if (required <= scratch_capacity_
            && static_cast<int>(thread_scratch_.size()) >= required_threads) {
            return;
        }
        int capacity = std::max(64, scratch_capacity_);
        while (capacity < required) capacity *= 2;
        thread_scratch_.resize(std::max(
            thread_scratch_.size(), static_cast<std::size_t>(required_threads)));
        for (auto &scratch : thread_scratch_) scratch.resize(capacity);
        scratch_capacity_ = capacity;
    }

#if defined(__aarch64__)
    static void transpose8x8F16(const mllm_fp16_t *src,
                                mllm_fp16_t *dst,
                                int src_stride, int dst_stride) {
        const uint16x8_t r0 = vreinterpretq_u16_f16(vld1q_f16(
            reinterpret_cast<const float16_t *>(src + 0 * src_stride)));
        const uint16x8_t r1 = vreinterpretq_u16_f16(vld1q_f16(
            reinterpret_cast<const float16_t *>(src + 1 * src_stride)));
        const uint16x8_t r2 = vreinterpretq_u16_f16(vld1q_f16(
            reinterpret_cast<const float16_t *>(src + 2 * src_stride)));
        const uint16x8_t r3 = vreinterpretq_u16_f16(vld1q_f16(
            reinterpret_cast<const float16_t *>(src + 3 * src_stride)));
        const uint16x8_t r4 = vreinterpretq_u16_f16(vld1q_f16(
            reinterpret_cast<const float16_t *>(src + 4 * src_stride)));
        const uint16x8_t r5 = vreinterpretq_u16_f16(vld1q_f16(
            reinterpret_cast<const float16_t *>(src + 5 * src_stride)));
        const uint16x8_t r6 = vreinterpretq_u16_f16(vld1q_f16(
            reinterpret_cast<const float16_t *>(src + 6 * src_stride)));
        const uint16x8_t r7 = vreinterpretq_u16_f16(vld1q_f16(
            reinterpret_cast<const float16_t *>(src + 7 * src_stride)));

        const uint16x8x2_t t01 = vtrnq_u16(r0, r1);
        const uint16x8x2_t t23 = vtrnq_u16(r2, r3);
        const uint16x8x2_t t45 = vtrnq_u16(r4, r5);
        const uint16x8x2_t t67 = vtrnq_u16(r6, r7);

        const uint32x4x2_t u02_even = vtrnq_u32(
            vreinterpretq_u32_u16(t01.val[0]),
            vreinterpretq_u32_u16(t23.val[0]));
        const uint32x4x2_t u02_odd = vtrnq_u32(
            vreinterpretq_u32_u16(t01.val[1]),
            vreinterpretq_u32_u16(t23.val[1]));
        const uint32x4x2_t u46_even = vtrnq_u32(
            vreinterpretq_u32_u16(t45.val[0]),
            vreinterpretq_u32_u16(t67.val[0]));
        const uint32x4x2_t u46_odd = vtrnq_u32(
            vreinterpretq_u32_u16(t45.val[1]),
            vreinterpretq_u32_u16(t67.val[1]));

        const auto combine_halves = [](uint32x4_t low_rows,
                                       uint32x4_t high_rows,
                                       bool high) {
            const uint64x2_t low64 = vreinterpretq_u64_u32(low_rows);
            const uint64x2_t high64 = vreinterpretq_u64_u32(high_rows);
            return vreinterpretq_u16_u64(high
                ? vcombine_u64(vget_high_u64(low64), vget_high_u64(high64))
                : vcombine_u64(vget_low_u64(low64), vget_low_u64(high64)));
        };

        const uint16x8_t out[8] = {
            combine_halves(u02_even.val[0], u46_even.val[0], false),
            combine_halves(u02_odd.val[0], u46_odd.val[0], false),
            combine_halves(u02_even.val[1], u46_even.val[1], false),
            combine_halves(u02_odd.val[1], u46_odd.val[1], false),
            combine_halves(u02_even.val[0], u46_even.val[0], true),
            combine_halves(u02_odd.val[0], u46_odd.val[0], true),
            combine_halves(u02_even.val[1], u46_even.val[1], true),
            combine_halves(u02_odd.val[1], u46_odd.val[1], true),
        };
        for (int row = 0; row < 8; ++row) {
            vst1q_f16(
                reinterpret_cast<float16_t *>(dst + row * dst_stride),
                vreinterpretq_f16_u16(out[row]));
        }
    }
#endif

    static void packValueHeadRangeF16(const mllm_fp16_t *src,
                                      mllm_fp16_t *dst,
                                      int src_stride, int key_begin,
                                      int key_end, int head_dim) {
        constexpr int block = 8;
        const int full_dims = head_dim / block * block;
        int key = key_begin;
        for (; key < key_end && key % block != 0; ++key) {
            for (int d = 0; d < head_dim; ++d) {
                dst[key * head_dim + d] = src[d * src_stride + key];
            }
        }
        const int full_key_end = key + (key_end - key) / block * block;
        for (; key < full_key_end; key += block) {
            for (int d = 0; d < full_dims; d += block) {
#if defined(__aarch64__)
                transpose8x8F16(src + d * src_stride + key,
                                dst + key * head_dim + d,
                                src_stride, head_dim);
#else
                for (int di = 0; di < block; ++di) {
                    for (int ki = 0; ki < block; ++ki) {
                        dst[(key + ki) * head_dim + d + di] =
                            src[(d + di) * src_stride + key + ki];
                    }
                }
#endif
            }
            for (int d = full_dims; d < head_dim; ++d) {
                for (int ki = 0; ki < block; ++ki) {
                    dst[(key + ki) * head_dim + d] =
                        src[d * src_stride + key + ki];
                }
            }
        }
        for (; key < key_end; ++key) {
            for (int d = 0; d < head_dim; ++d) {
                dst[key * head_dim + d] = src[d * src_stride + key];
            }
        }
    }

    static void packKeyHeadRangeF16(const mllm_fp16_t *src,
                                    mllm_fp16_t *dst,
                                    int src_stride, int key_begin,
                                    int key_end, int head_dim) {
        for (int key = key_begin; key < key_end; ++key) {
            std::memcpy(dst + static_cast<size_t>(key) * head_dim,
                        src + static_cast<size_t>(key) * src_stride,
                        static_cast<size_t>(head_dim)
                            * sizeof(mllm_fp16_t));
        }
    }

    template <bool positions_are_indices>
    static int32_t sparseValueOffset(const int32_t *positions, int selected,
                                     int head_dim) {
        if constexpr (positions_are_indices) {
            return positions[selected] * head_dim;
        }
        return positions[selected];
    }

    template <bool positions_are_indices>
    static void sparseValueF16Impl(const mllm_fp16_t *packed_value,
                                   const int32_t *positions,
                                   const mllm_fp16_t *weights,
                                   int keep, int head_dim,
                                   float *output) {
#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_FML)
        // A 64-dimension block uses 16 FP32 accumulators and two temporary
        // vectors. For a 128-d head this halves visits to the sparse
        // index/weight stream relative to the former 32-d block.
        constexpr int dim_tile = 64;
        int d = 0;
        for (; d + dim_tile <= head_dim; d += dim_tile) {
            float32x4_t accumulators[16] = {
                vdupq_n_f32(0.0F), vdupq_n_f32(0.0F),
                vdupq_n_f32(0.0F), vdupq_n_f32(0.0F),
                vdupq_n_f32(0.0F), vdupq_n_f32(0.0F),
                vdupq_n_f32(0.0F), vdupq_n_f32(0.0F),
                vdupq_n_f32(0.0F), vdupq_n_f32(0.0F),
                vdupq_n_f32(0.0F), vdupq_n_f32(0.0F),
                vdupq_n_f32(0.0F), vdupq_n_f32(0.0F),
                vdupq_n_f32(0.0F), vdupq_n_f32(0.0F),
            };
            for (int selected = 0; selected < keep; ++selected) {
                if (selected + 8 < keep) {
                    __builtin_prefetch(
                        packed_value
                            + sparseValueOffset<positions_are_indices>(
                                  positions, selected + 8, head_dim)
                            + d,
                        0, 1);
                }
                const mllm_fp16_t *value_row = packed_value
                    + sparseValueOffset<positions_are_indices>(
                          positions, selected, head_dim)
                    + d;
                const float16x8_t weight = vdupq_n_f16(weights[selected]);
                for (int block = 0; block < 8; ++block) {
                    const float16x8_t value = vld1q_f16(
                        reinterpret_cast<const float16_t *>(value_row + block * 8));
                    accumulators[block * 2] = vfmlalq_low_f16(
                        accumulators[block * 2], value, weight);
                    accumulators[block * 2 + 1] = vfmlalq_high_f16(
                        accumulators[block * 2 + 1], value, weight);
                }
            }
            for (int block = 0; block < 16; ++block) {
                vst1q_f32(output + d + block * 4, accumulators[block]);
            }
        }
        for (; d + 8 <= head_dim; d += 8) {
            float32x4_t acc_low = vdupq_n_f32(0.0F);
            float32x4_t acc_high = vdupq_n_f32(0.0F);
            for (int selected = 0; selected < keep; ++selected) {
                const mllm_fp16_t *value_row = packed_value
                    + sparseValueOffset<positions_are_indices>(
                          positions, selected, head_dim)
                    + d;
                const float16x8_t value = vld1q_f16(
                    reinterpret_cast<const float16_t *>(value_row));
                const float16x8_t weight = vdupq_n_f16(weights[selected]);
                acc_low = vfmlalq_low_f16(acc_low, value, weight);
                acc_high = vfmlalq_high_f16(acc_high, value, weight);
            }
            vst1q_f32(output + d, acc_low);
            vst1q_f32(output + d + 4, acc_high);
        }
        for (; d < head_dim; ++d) {
            float sum = 0.0F;
            for (int selected = 0; selected < keep; ++selected) {
                const auto value = packed_value[
                    sparseValueOffset<positions_are_indices>(
                        positions, selected, head_dim) + d];
                sum += MLLM_FP16_TO_FP32(weights[selected])
                    * MLLM_FP16_TO_FP32(value);
            }
            output[d] = sum;
        }
#else
        for (int d = 0; d < head_dim; ++d) {
            float sum = 0.0F;
            for (int selected = 0; selected < keep; ++selected) {
                const auto value = packed_value[
                    sparseValueOffset<positions_are_indices>(
                        positions, selected, head_dim) + d];
                sum += MLLM_FP16_TO_FP32(weights[selected])
                    * MLLM_FP16_TO_FP32(value);
            }
            output[d] = sum;
        }
#endif
    }

    static void sparseValueF16(const mllm_fp16_t *packed_value,
                               const int32_t *selected_offsets,
                               const mllm_fp16_t *weights,
                               int keep, int head_dim,
                               float *output) {
        sparseValueF16Impl<false>(packed_value, selected_offsets, weights,
                                  keep, head_dim, output);
    }

    static void sparseValueF16Selected(const mllm_fp16_t *packed_value,
                                       const int32_t *selected_indices,
                                       const mllm_fp16_t *weights,
                                       int keep, int head_dim,
                                       float *output) {
        sparseValueF16Impl<true>(packed_value, selected_indices, weights,
                                 keep, head_dim, output);
    }

    PackedValueState *preparePackedValueF16(
        const shared_ptr<Tensor> &value, int batch, int heads,
        int query_len, int key_len, int head_dim, bool profile_enabled) {
        assert(value->dtype() == MLLM_TYPE_F16);
        assert(value->ctype() == BHDS);
        assert(static_cast<int64_t>(key_len - 1) * head_dim
               <= std::numeric_limits<int32_t>::max());

        const bool has_cache_master = value->masterTensor() != nullptr;
        auto packed_source = value;
        while (auto master = packed_source->masterTensor()) {
            packed_source = master;
        }
        const int src_stride = value->sequenceSkipDim();
        const void *logical_base = static_cast<const void *>(
            value->ptrAt<mllm_fp16_t>(0, 0, 0, 0));
        auto state_it = packed_values_.find(packed_source.get());
        bool inserted = false;
        if (state_it == packed_values_.end()
            || state_it->second.source.expired()) {
            if (state_it != packed_values_.end()) {
                packed_values_.erase(state_it);
            }
            for (auto it = packed_values_.begin(); it != packed_values_.end();) {
                if (it->second.source.expired()) {
                    it = packed_values_.erase(it);
                } else {
                    ++it;
                }
            }
            auto result = packed_values_.try_emplace(packed_source.get());
            state_it = result.first;
            inserted = result.second;
        }
        auto &state = state_it->second;
        const auto previous_source = state.source.lock();
        const bool same_storage = !inserted
            && previous_source.get() == packed_source.get()
            && state.batch == batch && state.heads == heads
            && state.head_dim == head_dim;
        const bool layout_matches = same_storage
            && state.src_stride == src_stride
            && state.logical_base == logical_base;
        if (!layout_matches) {
            state.reset(packed_source, batch, heads, head_dim,
                        src_stride, logical_base);
        }
        // A prompt cache appends one query chunk at a time.  Equal or shorter
        // lengths mean a warmup/new request and must refresh the current prefix.
        const bool append_only = has_cache_master && layout_matches
            && key_len > state.packed_key_len
            && state.packed_key_len == key_len - query_len;
        const int pack_begin = append_only ? state.packed_key_len : 0;
        constexpr int eager_reserve_token_limit = 4096;
        const int source_capacity = std::max(key_len, src_stride);
        const int reserve_hint = pack_reserve_tokens_ > 0
            ? pack_reserve_tokens_
            : std::min(source_capacity, eager_reserve_token_limit);
        const int reserve_tokens = std::max(key_len, reserve_hint);
        const size_t packed_head_capacity =
            static_cast<size_t>(reserve_tokens) * head_dim;
        const size_t preserve_elements = append_only
            ? static_cast<size_t>(pack_begin) * head_dim : 0;
        ScopedAttentionProfile pack_profile(
            AttentionProfileStage::SPARSE_PACK, profile_enabled);
        for (auto &head_data : state.head_data) {
            head_data.reserve(packed_head_capacity, preserve_elements);
        }
#pragma omp parallel for collapse(2) schedule(static) num_threads(thread_count_)
        for (int b = 0; b < batch; ++b) {
            for (int h = 0; h < heads; ++h) {
                const mllm_fp16_t *src =
                    value->ptrAt<mllm_fp16_t>(b, h, 0, 0);
                mllm_fp16_t *dst = state.head_data[
                    static_cast<size_t>(b) * heads + h].data.get();
                packValueHeadRangeF16(
                    src, dst, src_stride, pack_begin, key_len, head_dim);
            }
        }
        state.packed_key_len = key_len;
        return &state;
    }

    PackedValueState *preparePackedKeyF16(
        const shared_ptr<Tensor> &key, int batch, int heads,
        int query_len, int key_len, int head_dim, bool profile_enabled) {
        assert(key->dtype() == MLLM_TYPE_F16);
        assert(key->ctype() == BSHD);
        assert(key->sequenceSkipDim() > head_dim);
        assert(static_cast<int64_t>(key_len - 1) * head_dim
               <= std::numeric_limits<int32_t>::max());

        const bool has_cache_master = key->masterTensor() != nullptr;
        auto packed_source = key;
        while (auto master = packed_source->masterTensor()) {
            packed_source = master;
        }
        const int src_stride = key->sequenceSkipDim();
        const void *logical_base = static_cast<const void *>(
            key->ptrAt<mllm_fp16_t>(0, 0, 0, 0));
        auto state_it = packed_keys_.find(packed_source.get());
        bool inserted = false;
        if (state_it == packed_keys_.end()
            || state_it->second.source.expired()) {
            if (state_it != packed_keys_.end()) packed_keys_.erase(state_it);
            for (auto it = packed_keys_.begin(); it != packed_keys_.end();) {
                if (it->second.source.expired()) {
                    it = packed_keys_.erase(it);
                } else {
                    ++it;
                }
            }
            auto result = packed_keys_.try_emplace(packed_source.get());
            state_it = result.first;
            inserted = result.second;
        }
        auto &state = state_it->second;
        const auto previous_source = state.source.lock();
        const bool same_storage = !inserted
            && previous_source.get() == packed_source.get()
            && state.batch == batch && state.heads == heads
            && state.head_dim == head_dim;
        const bool layout_matches = same_storage
            && state.src_stride == src_stride
            && state.logical_base == logical_base;
        if (!layout_matches) {
            state.reset(packed_source, batch, heads, head_dim,
                        src_stride, logical_base);
        }
        const bool recognized_direct_cache = has_cache_master
            && packed_source->ctype() == BSHD
            && packed_source->name().find(".Cache") != std::string::npos;
        // Incremental packing assumes CPUKVCacheNPU's non-wrapping direct
        // cache view. Generic/master views are conservatively refreshed.
        const bool append_only = recognized_direct_cache && layout_matches
            && key_len > state.packed_key_len
            && state.packed_key_len == key_len - query_len;
        const int pack_begin = append_only ? state.packed_key_len : 0;
        last_key_pack_begin_ = pack_begin;
        last_key_pack_end_ = key_len;
        const int reserve_tokens = std::max(key_len, pack_reserve_tokens_);
        const size_t packed_head_capacity =
            static_cast<size_t>(reserve_tokens) * head_dim;
        const size_t preserve_elements = append_only
            ? static_cast<size_t>(pack_begin) * head_dim : 0;
        ScopedAttentionProfile pack_profile(
            AttentionProfileStage::SPARSE_PACK, profile_enabled);
        for (auto &head_data : state.head_data) {
            head_data.reserve(packed_head_capacity, preserve_elements);
        }
#pragma omp parallel for collapse(2) schedule(static) num_threads(thread_count_)
        for (int b = 0; b < batch; ++b) {
            for (int h = 0; h < heads; ++h) {
                const mllm_fp16_t *src =
                    key->ptrAt<mllm_fp16_t>(b, h, 0, 0);
                mllm_fp16_t *dst = state.head_data[
                    static_cast<size_t>(b) * heads + h].data.get();
                packKeyHeadRangeF16(
                    src, dst, src_stride, pack_begin, key_len, head_dim);
            }
        }
        state.packed_key_len = key_len;
        return &state;
    }

public:
    CPUSparseSoftmaxValueFunc(Backend *bn, string name, int thread_count,
                              float sparsity, bool causal_mask,
                              int topk_sample_size = 0,
                              std::vector<float> head_retentions = {}) :
        Op(bn, name),
        thread_count_(thread_count),
        sparsity_(sparsity),
        causal_mask_(causal_mask),
        topk_sample_size_(topk_sample_size),
        head_retentions_(std::move(head_retentions)) {
        assert(sparsity_ > 0.0F && sparsity_ < 1.0F);
        assert(topk_sample_size_ >= 0);
        sampled_candidate_keep_ = sampledCandidateKeep(
            topk_sample_size_, 1.0F - sparsity_);
    }

    int lastKeyPackBeginForTest() const { return last_key_pack_begin_; }
    int lastKeyPackEndForTest() const { return last_key_pack_end_; }

    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs,
                      vector<shared_ptr<Tensor>> outputs) override {
        assert(inputs.size() == 2 && outputs.size() == 1);
        const auto &logits = inputs[0];
        const auto &value = inputs[1];
        if (value->ctype() == BSHD) {
            transposeAttentionValueChannels(*value);
        }
        if (value->ctype() != BHDS) return ::INVALID_VALUE;
        assert(logits->batch() == value->batch());
        assert(logits->head() == value->head());
        assert(logits->dimension() == value->sequence());
        outputs[0]->setCtype(logits->ctype());
        outputs[0]->setDtype(logits->dtype());
        outputs[0]->reshape(logits->batch(), logits->head(), logits->sequence(),
                            value->dimension());
        return MLLM_NO_ERROR;
    }

    ErrorCode setUp(vector<shared_ptr<Tensor>> inputs,
                    vector<shared_ptr<Tensor>> outputs) override {
        if (inputs[1]->ctype() == BSHD) {
            transposeAttentionValueChannels(*inputs[1]);
        }
        if (inputs[1]->ctype() != BHDS) return ::INVALID_VALUE;
        outputs[0]->setCtype(inputs[0]->ctype());
        outputs[0]->setDtype(inputs[0]->dtype());
        outputs[0]->alloc();
        return MLLM_NO_ERROR;
    }

    ErrorCode execute(vector<shared_ptr<Tensor>> inputs,
                      vector<shared_ptr<Tensor>> outputs) override {
        const bool profile_enabled = CPUAttentionProfiler::enabled();
        ScopedAttentionProfile total_profile(
            AttentionProfileStage::SPARSE_TOTAL, profile_enabled);
        const auto &logits = inputs[0];
        const auto &value = inputs[1];
        const auto &output = outputs[0];

        if (value->ctype() != BHDS) return ::INVALID_VALUE;

        assert(logits->dtype() == MLLM_TYPE_F32);
        assert(value->dtype() == MLLM_TYPE_F16 || value->dtype() == MLLM_TYPE_F32);
        assert(value->ctype() == BHDS);
        assert(output->ctype() == BSHD || output->ctype() == BHSD);

        const int batch = logits->batch();
        const int heads = logits->head();
        const int query_len = logits->sequence();
        const int key_len = logits->dimension();
        const int head_dim = value->dimension();
        const SparseQualityDiagnosticConfig quality_diagnostics =
            sparseQualityDiagnosticConfig();
        assert(query_len > 0 && key_len >= query_len && head_dim > 0);
        if (!head_retentions_.empty()
            && head_retentions_.size() != static_cast<std::size_t>(heads)) {
            return ::INVALID_VALUE;
        }
        const float maximum_density = head_retentions_.empty()
            ? 1.0F - sparsity_
            : *std::max_element(
                  head_retentions_.begin(), head_retentions_.end());
        const int total_rows = batch * heads * query_len;
        ensureScratchCapacity(std::max(key_len, head_dim));
        const bool use_packed_f16 = value->dtype() == MLLM_TYPE_F16
            && static_cast<int>(std::ceil(maximum_density * key_len)) < key_len;
        PackedValueState *packed_state = nullptr;
        if (use_packed_f16) {
            packed_state = preparePackedValueF16(
                value, batch, heads, query_len, key_len, head_dim,
                profile_enabled);
        }
        uint64_t select_work_ns = 0;
        uint64_t pv_work_ns = 0;
        uint64_t eligible_elements = 0;
        uint64_t retained_elements = 0;
        uint64_t sampled_rows = 0;
        uint64_t fallback_rows = 0;
        uint64_t candidate_elements = 0;
        uint64_t diagnostic_rows = 0;
        uint64_t diagnostic_tail_rows = 0;
        uint64_t diagnostic_target_rows = 0;
        uint64_t diagnostic_target_top20_hits = 0;
        double diagnostic_actual_mass_sum = 0.0;
        double diagnostic_top20_mass_sum = 0.0;
        double diagnostic_entropy_sum = 0.0;
        double diagnostic_boundary_gap_sum = 0.0;
        double diagnostic_tail_top20_mass_sum = 0.0;
        double diagnostic_target_rank_sum = 0.0;
        double diagnostic_target_mass_sum = 0.0;
        double diagnostic_tail_relative_l2_sum = 0.0;
        double diagnostic_tail_cosine_sum = 0.0;

#pragma omp parallel num_threads(thread_count_) \
    reduction(+:select_work_ns,pv_work_ns,eligible_elements,retained_elements, \
              sampled_rows,fallback_rows,candidate_elements,diagnostic_rows, \
              diagnostic_tail_rows,diagnostic_target_rows, \
              diagnostic_target_top20_hits,diagnostic_actual_mass_sum, \
              diagnostic_top20_mass_sum,diagnostic_entropy_sum, \
              diagnostic_boundary_gap_sum,diagnostic_tail_top20_mass_sum, \
              diagnostic_target_rank_sum,diagnostic_target_mass_sum, \
              diagnostic_tail_relative_l2_sum,diagnostic_tail_cosine_sum)
        {
            auto &scratch = thread_scratch_[omp_get_thread_num()];
            float *rank_scores = scratch.rank_scores.data();
            float *weights = scratch.weights.data();
            int32_t *selected_indices = scratch.selected_indices.data();
            mllm_fp16_t *weights_f16 = scratch.weights_f16.data();

#pragma omp for schedule(static)
            for (int row = 0; row < total_rows; ++row) {
                const uint64_t select_start_ns = profile_enabled
                    ? CPUAttentionProfiler::nowNs() : 0;
                const int s = row % query_len;
                const int h = (row / query_len) % heads;
                const int b = row / (query_len * heads);
                const float density = head_retentions_.empty()
                    ? 1.0F - sparsity_ : head_retentions_[h];
                const int valid = causal_mask_ && query_len > 1
                    ? std::min(key_len, key_len - query_len + s + 1)
                    : key_len;
                assert(valid > 0);
                const int keep = std::max(1, std::min(valid,
                    static_cast<int>(std::ceil(density * static_cast<float>(valid)))));

                const float *logit_row = logits->ptrAt<float>(b, h, s, 0);
                float row_max = -INFINITY;
                const auto select_exact_full = [&]() {
                    std::memcpy(rank_scores, logit_row,
                                static_cast<size_t>(valid) * sizeof(float));
                    std::nth_element(rank_scores, rank_scores + keep - 1,
                                     rank_scores + valid, std::greater<float>());
                    const float threshold = rank_scores[keep - 1];
                    int greater_count = 0;
                    row_max = -INFINITY;
                    for (int i = 0; i < keep; ++i) {
                        greater_count += rank_scores[i] > threshold;
                        row_max = std::max(row_max, rank_scores[i]);
                    }
                    int ties_left = keep - greater_count;
                    int selected = 0;
                    if (density <= 0.25F) {
                        // At high sparsity this branch is strongly biased and
                        // avoids writing almost every rejected key.
                        for (int key = 0; key < valid; ++key) {
                            const float score = logit_row[key];
                            const bool take_tie = score == threshold
                                && ties_left > 0;
                            if (score > threshold || take_tie) {
                                selected_indices[selected] = key;
                                weights[selected] = score;
                                ++selected;
                                ties_left -= static_cast<int>(take_tie);
                            }
                        }
                    } else {
                        // Near 50% density the predicate is unpredictable;
                        // unconditional compacting is faster than mispredicts.
                        for (int key = 0; key < valid; ++key) {
                            const float score = logit_row[key];
                            const bool take_tie = score == threshold
                                && ties_left > 0;
                            const bool take = score > threshold || take_tie;
                            selected_indices[selected] = key;
                            weights[selected] = score;
                            selected += static_cast<int>(take);
                            ties_left -= static_cast<int>(take_tie);
                        }
                    }
                    assert(selected == keep && ties_left == 0);
                };
                if (keep == valid) {
                    for (int key = 0; key < valid; ++key) {
                        selected_indices[key] = key;
                        weights[key] = logit_row[key];
                        row_max = std::max(row_max, logit_row[key]);
                    }
                } else if (keep == 1) {
                    int best_key = 0;
                    float best_score = logit_row[0];
                    for (int key = 1; key < valid; ++key) {
                        if (logit_row[key] > best_score) {
                            best_score = logit_row[key];
                            best_key = key;
                        }
                    }
                    selected_indices[0] = best_key;
                    weights[0] = best_score;
                    row_max = best_score;
                } else if (topk_sample_size_ > 0
                           && topk_sample_size_ <= valid / 4) {
                    const int sampled_candidate_keep =
                        sampledCandidateKeep(topk_sample_size_, density);
                    ++sampled_rows;
                    uint32_t sample_state = 0x9e3779b9U
                        ^ (static_cast<uint32_t>(row) + 1U) * 0x85ebca6bU
                        ^ static_cast<uint32_t>(key_len) * 0xc2b2ae35U;
                    if (sample_state == 0) sample_state = 0x6d2b79f5U;
                    for (int i = 0; i < topk_sample_size_; ++i) {
                        // Xorshift plus multiply-high gives a deterministic,
                        // uniform sample without a division in the hot loop.
                        sample_state ^= sample_state << 13;
                        sample_state ^= sample_state >> 17;
                        sample_state ^= sample_state << 5;
                        const int key = static_cast<int>(
                            (static_cast<uint64_t>(sample_state)
                                * static_cast<uint32_t>(valid)) >> 32);
                        rank_scores[i] = logit_row[key];
                    }
                    std::nth_element(rank_scores,
                                     rank_scores + sampled_candidate_keep - 1,
                                     rank_scores + topk_sample_size_,
                                     std::greater<float>());
                    const float candidate_threshold =
                        rank_scores[sampled_candidate_keep - 1];
                    int candidate_count = 0;
                    for (int key = 0; key < valid; ++key) {
                        const float score = logit_row[key];
                        if (score >= candidate_threshold) {
                            selected_indices[candidate_count] = key;
                            weights[candidate_count] = score;
                            row_max = std::max(row_max, score);
                            ++candidate_count;
                        }
                    }
                    candidate_elements += static_cast<uint64_t>(candidate_count);
                    if (candidate_count < keep) {
                        // A conservative sampled cutoff should normally contain
                        // every global Top-K element. Fall back to the full
                        // exact path when a row's distribution defeats it.
                        ++fallback_rows;
                        select_exact_full();
                    } else if (candidate_count > keep) {
                        std::memcpy(rank_scores, weights,
                                    static_cast<size_t>(candidate_count)
                                        * sizeof(float));
                        std::nth_element(rank_scores, rank_scores + keep - 1,
                                         rank_scores + candidate_count,
                                         std::greater<float>());
                        const float threshold = rank_scores[keep - 1];
                        int greater_count = 0;
                        row_max = -INFINITY;
                        for (int i = 0; i < keep; ++i) {
                            greater_count += rank_scores[i] > threshold;
                            row_max = std::max(row_max, rank_scores[i]);
                        }
                        int ties_left = keep - greater_count;
                        int selected = 0;
                        for (int candidate = 0; candidate < candidate_count;
                             ++candidate) {
                            const float score = weights[candidate];
                            const bool take_tie = score == threshold
                                && ties_left > 0;
                            const bool take = score > threshold || take_tie;
                            selected_indices[selected] =
                                selected_indices[candidate];
                            weights[selected] = score;
                            selected += static_cast<int>(take);
                            ties_left -= static_cast<int>(take_tie);
                        }
                        assert(selected == keep && ties_left == 0);
                    }
                } else {
                    select_exact_full();
                }
                double diagnostic_dense_sum = 0.0;
                double diagnostic_dense_max = static_cast<double>(row_max);
                if (quality_diagnostics.enabled) {
                    double dense_score_moment = 0.0;
                    for (int key = 0; key < valid; ++key) {
                        const double score = static_cast<double>(logit_row[key]);
                        const double exponential = std::exp(
                            score - diagnostic_dense_max);
                        diagnostic_dense_sum += exponential;
                        dense_score_moment += exponential * score;
                    }
                    double selected_exponential_sum = 0.0;
                    for (int selected = 0; selected < keep; ++selected) {
                        selected_exponential_sum += std::exp(
                            static_cast<double>(weights[selected])
                            - diagnostic_dense_max);
                    }
                    const int fixed_top20_keep = std::max(
                        1, std::min(valid, static_cast<int>(std::ceil(
                            0.2F * static_cast<float>(valid)))));
                    double fixed_top20_exponential_sum = diagnostic_dense_sum;
                    double boundary_gap = 0.0;
                    if (fixed_top20_keep < valid) {
                        std::memcpy(rank_scores, logit_row,
                                    static_cast<size_t>(valid)
                                        * sizeof(float));
                        std::nth_element(
                            rank_scores, rank_scores + fixed_top20_keep,
                            rank_scores + valid, std::greater<float>());
                        fixed_top20_exponential_sum = 0.0;
                        float selected_minimum = INFINITY;
                        for (int i = 0; i < fixed_top20_keep; ++i) {
                            fixed_top20_exponential_sum += std::exp(
                                static_cast<double>(rank_scores[i])
                                - diagnostic_dense_max);
                            selected_minimum = std::min(
                                selected_minimum, rank_scores[i]);
                        }
                        boundary_gap = std::max(
                            0.0, static_cast<double>(selected_minimum)
                                - static_cast<double>(
                                    rank_scores[fixed_top20_keep]));
                    }
                    const double entropy = valid <= 1
                        ? 0.0
                        : (std::log(diagnostic_dense_sum)
                           + diagnostic_dense_max
                           - dense_score_moment / diagnostic_dense_sum)
                            / std::log(static_cast<double>(valid));
                    const double actual_mass = selected_exponential_sum
                        / diagnostic_dense_sum;
                    const double fixed_top20_mass =
                        fixed_top20_exponential_sum / diagnostic_dense_sum;
                    ++diagnostic_rows;
                    diagnostic_actual_mass_sum += actual_mass;
                    diagnostic_top20_mass_sum += fixed_top20_mass;
                    diagnostic_entropy_sum += entropy;
                    diagnostic_boundary_gap_sum += boundary_gap;

                    if (s == query_len - 1) {
                        ++diagnostic_tail_rows;
                        diagnostic_tail_top20_mass_sum += fixed_top20_mass;
                        const int target_begin = std::max(
                            0, quality_diagnostics.target_begin);
                        const int target_end = std::min(
                            valid, quality_diagnostics.target_end);
                        if (quality_diagnostics.target_begin >= 0
                            && target_begin < target_end) {
                            float target_best_score = -INFINITY;
                            double target_exponential_sum = 0.0;
                            for (int key = target_begin; key < target_end;
                                 ++key) {
                                target_best_score = std::max(
                                    target_best_score, logit_row[key]);
                                target_exponential_sum += std::exp(
                                    static_cast<double>(logit_row[key])
                                    - diagnostic_dense_max);
                            }
                            int target_rank = 1;
                            for (int key = 0; key < valid; ++key) {
                                target_rank += static_cast<int>(
                                    logit_row[key] > target_best_score);
                            }
                            ++diagnostic_target_rows;
                            diagnostic_target_rank_sum += target_rank;
                            diagnostic_target_mass_sum +=
                                target_exponential_sum / diagnostic_dense_sum;
                            diagnostic_target_top20_hits +=
                                static_cast<uint64_t>(
                                    target_rank <= fixed_top20_keep);
                        }
                    }
                }
                eligible_elements += static_cast<uint64_t>(valid);
                retained_elements += static_cast<uint64_t>(keep);
                const float row_sum = mllm_vec_soft_max_f32(
                    keep, weights, weights, row_max);
                const float inv_sum = 1.0F / row_sum;

                float *output_row = output->ptrAt<float>(b, h, s, 0);
                if (value->dtype() == MLLM_TYPE_F16) {
                    // The keep-all control follows dense AV exactly: causal
                    // future positions remain explicit zeros and vec_dot sees
                    // the full key length. This isolates kernel-switch error
                    // from actual Top-K pruning error.
                    const int dot_length = keep == valid ? key_len : keep;
                    if (keep == valid) {
                        vec_scale_f32(keep, weights, inv_sum);
                        for (int selected = 0; selected < keep; ++selected) {
                            weights_f16[selected] =
                                MLLM_FP32_TO_FP16(weights[selected]);
                        }
                    } else {
                        for (int selected = 0; selected < keep; ++selected) {
                            weights_f16[selected] = MLLM_FP32_TO_FP16(
                                weights[selected] * inv_sum);
                            selected_indices[selected] *= head_dim;
                        }
                    }
                    std::fill(weights_f16 + keep, weights_f16 + dot_length,
                              MLLM_FP32_TO_FP16(0.0F));
                    const uint64_t pv_start_ns = profile_enabled
                        ? CPUAttentionProfiler::nowNs() : 0;
                    if (profile_enabled) {
                        select_work_ns += pv_start_ns - select_start_ns;
                    }
                    if (keep == valid) {
                        for (int d = 0; d < head_dim; ++d) {
                            const mllm_fp16_t *value_line =
                                value->ptrAt<mllm_fp16_t>(b, h, 0, d);
                            vec_dot_fp16(key_len, output_row + d,
                                         weights_f16, value_line);
                        }
                    } else {
                        assert(use_packed_f16 && packed_state != nullptr);
                        const mllm_fp16_t *packed_head =
                            packed_state->head_data[
                                static_cast<size_t>(b) * heads + h].data.get();
                        sparseValueF16(packed_head, selected_indices,
                                       weights_f16, keep, head_dim, output_row);
                    }
                    if (profile_enabled) {
                        pv_work_ns += CPUAttentionProfiler::nowNs() - pv_start_ns;
                    }
                } else {
                    vec_scale_f32(keep, weights, inv_sum);
                    const uint64_t pv_start_ns = profile_enabled
                        ? CPUAttentionProfiler::nowNs() : 0;
                    if (profile_enabled) {
                        select_work_ns += pv_start_ns - select_start_ns;
                    }
                    int d = 0;
                    for (; d + 3 < head_dim; d += 4) {
                        const float *value_line_0 = value->ptrAt<float>(b, h, 0, d);
                        const float *value_line_1 = value->ptrAt<float>(b, h, 0, d + 1);
                        const float *value_line_2 = value->ptrAt<float>(b, h, 0, d + 2);
                        const float *value_line_3 = value->ptrAt<float>(b, h, 0, d + 3);
                        float sum_0 = 0.0F;
                        float sum_1 = 0.0F;
                        float sum_2 = 0.0F;
                        float sum_3 = 0.0F;
                        for (int selected = 0; selected < keep; ++selected) {
                            const int key = selected_indices[selected];
                            const float weight = weights[selected];
                            sum_0 += weight * value_line_0[key];
                            sum_1 += weight * value_line_1[key];
                            sum_2 += weight * value_line_2[key];
                            sum_3 += weight * value_line_3[key];
                        }
                        output_row[d] = sum_0;
                        output_row[d + 1] = sum_1;
                        output_row[d + 2] = sum_2;
                        output_row[d + 3] = sum_3;
                    }
                    for (; d < head_dim; ++d) {
                        const float *value_line = value->ptrAt<float>(b, h, 0, d);
                        float sum = 0.0F;
                        for (int selected = 0; selected < keep; ++selected) {
                            sum += weights[selected]
                                * value_line[selected_indices[selected]];
                        }
                        output_row[d] = sum;
                    }
                    if (profile_enabled) {
                        pv_work_ns += CPUAttentionProfiler::nowNs() - pv_start_ns;
                    }
                }
                if (quality_diagnostics.enabled && s == query_len - 1) {
                    for (int key = 0; key < valid; ++key) {
                        rank_scores[key] = static_cast<float>(std::exp(
                            static_cast<double>(logit_row[key])
                            - diagnostic_dense_max) / diagnostic_dense_sum);
                    }
                    float *dense_output = scratch.dense_output.data();
                    if (value->dtype() == MLLM_TYPE_F16) {
                        for (int d = 0; d < head_dim; ++d) {
                            const mllm_fp16_t *value_line =
                                value->ptrAt<mllm_fp16_t>(b, h, 0, d);
                            double sum = 0.0;
                            for (int key = 0; key < valid; ++key) {
                                sum += static_cast<double>(rank_scores[key])
                                    * MLLM_FP16_TO_FP32(value_line[key]);
                            }
                            dense_output[d] = static_cast<float>(sum);
                        }
                    } else {
                        for (int d = 0; d < head_dim; ++d) {
                            const float *value_line =
                                value->ptrAt<float>(b, h, 0, d);
                            double sum = 0.0;
                            for (int key = 0; key < valid; ++key) {
                                sum += static_cast<double>(rank_scores[key])
                                    * value_line[key];
                            }
                            dense_output[d] = static_cast<float>(sum);
                        }
                    }
                    double difference_norm_squared = 0.0;
                    double dense_norm_squared = 0.0;
                    double sparse_norm_squared = 0.0;
                    double dot = 0.0;
                    for (int d = 0; d < head_dim; ++d) {
                        const double dense_value = dense_output[d];
                        const double sparse_value = output_row[d];
                        const double difference = sparse_value - dense_value;
                        difference_norm_squared += difference * difference;
                        dense_norm_squared += dense_value * dense_value;
                        sparse_norm_squared += sparse_value * sparse_value;
                        dot += dense_value * sparse_value;
                    }
                    diagnostic_tail_relative_l2_sum += std::sqrt(
                        difference_norm_squared
                        / std::max(dense_norm_squared, 1.0e-30));
                    diagnostic_tail_cosine_sum += dot / std::sqrt(
                        std::max(dense_norm_squared * sparse_norm_squared,
                                 1.0e-30));
                }
            }
        }
        if (profile_enabled) {
            CPUAttentionProfiler::add(
                AttentionProfileStage::SPARSE_SELECT_WORK, select_work_ns);
            CPUAttentionProfiler::add(
                AttentionProfileStage::SPARSE_PV_WORK, pv_work_ns);
        }
        CPUSparseSelectionStats::add(
            eligible_elements, retained_elements, sampled_rows, fallback_rows,
            candidate_elements);
        if (quality_diagnostics.enabled && diagnostic_rows > 0) {
            const auto average = [](double sum, uint64_t count) {
                return count == 0 ? -1.0
                                  : sum / static_cast<double>(count);
            };
            std::lock_guard<std::mutex> lock(
                sparseQualityDiagnosticPrintMutex());
            std::fprintf(
                stdout,
                "SPARSE_QUALITY_DIAGNOSTIC op=%s key_len=%d query_len=%d "
                "heads=%d rows=%llu actual_retention=%.9f "
                "actual_topk_mass_mean=%.9f top20_mass_mean=%.9f "
                "entropy_normalized_mean=%.9f boundary20_gap_mean=%.9f "
                "tail_top20_mass_mean=%.9f target_begin=%d target_end=%d "
                "target_rank_mean=%.9f target_top20_rate=%.9f "
                "target_mass_mean=%.9f tail_relative_l2_mean=%.9f "
                "tail_cosine_mean=%.9f\n",
                name().c_str(), key_len, query_len, heads,
                static_cast<unsigned long long>(diagnostic_rows),
                eligible_elements == 0
                    ? 0.0
                    : static_cast<double>(retained_elements)
                        / static_cast<double>(eligible_elements),
                average(diagnostic_actual_mass_sum, diagnostic_rows),
                average(diagnostic_top20_mass_sum, diagnostic_rows),
                average(diagnostic_entropy_sum, diagnostic_rows),
                average(diagnostic_boundary_gap_sum, diagnostic_rows),
                average(diagnostic_tail_top20_mass_sum, diagnostic_tail_rows),
                quality_diagnostics.target_begin,
                quality_diagnostics.target_end,
                average(diagnostic_target_rank_sum, diagnostic_target_rows),
                average(static_cast<double>(diagnostic_target_top20_hits),
                        diagnostic_target_rows),
                average(diagnostic_target_mass_sum, diagnostic_target_rows),
                average(diagnostic_tail_relative_l2_sum,
                        diagnostic_tail_rows),
                average(diagnostic_tail_cosine_sum, diagnostic_tail_rows));
            std::fflush(stdout);
        }
        return MLLM_NO_ERROR;
    }
};

class CPUSparseSoftmaxValueFuncCreator final : public CPUBackend::Creator {
public:
    Op *create(OpParam op_param, Backend *bn, string name,
               int thread_count) const override {
        return new CPUSparseSoftmaxValueFunc(
            bn, name, thread_count, op_param.at("sparsity"),
            static_cast<bool>(op_param.at("causal_mask")),
            static_cast<int>(op_param.at("topk_sample_size")),
            [&]() {
                const int count = static_cast<int>(
                    op_param.at("head_retention_count"));
                std::vector<float> values;
                values.reserve(count);
                for (int head = 0; head < count; ++head) {
                    values.push_back(op_param.at(
                        "head_retention_" + std::to_string(head)));
                }
                return values;
            }());
    }
};

// Content-independent element-wise sparse prefill attention.  Unlike
// CPUSparseSoftmaxValueFunc, this op never materializes dense QK logits: it
// computes only a deterministic local+uniform or seeded per-attention-row
// random subset of keys, so both QK and P*V work scale with the retained
// element count.
class CPUPatternSparseAttentionFunc final : public CPUSparseSoftmaxValueFunc {
private:
    float local_ratio_ = 0.5F;
    int prefix_tokens_ = 0;
    int dense_tokens_ = 0;
    bool random_pattern_ = false;
    bool hmx_selector_enabled_ = false;
    uint32_t random_seed_ = 1;
    int layer_id_ = -1;
    int pattern_query_len_ = -1;
    int pattern_key_len_ = -1;
    std::vector<int32_t> pattern_offsets_;
    std::vector<int32_t> pattern_indices_;
    std::vector<std::size_t> pattern_head_bases_;
    int pattern_heads_ = -1;
    int max_topk_per_head_ = 0;
    std::unique_ptr<HMXInt8Selector> hmx_int8_selector_;
    std::vector<const float *> hmx_query_f32_heads_;
    std::vector<const HMXInt8Selector::Half *> hmx_key_heads_;

    struct SharedHmxIndexScratch {
        std::recursive_mutex mutex;
        std::unique_ptr<int32_t[]> data;
        std::size_t capacity = 0;
    };

    static SharedHmxIndexScratch &sharedHmxIndexScratch() {
        // Selected indices remain live through this attention operation's
        // sparse QK/softmax/PV, but not across dependent transformer layers.
        static auto *scratch = new SharedHmxIndexScratch();
        return *scratch;
    }

    void reserveHmxIndices(std::size_t required) {
        SharedHmxIndexScratch &scratch = sharedHmxIndexScratch();
        std::lock_guard<std::recursive_mutex> lock(scratch.mutex);
        if (required <= scratch.capacity) return;
        // HMX/Top-k overwrites every live index. Avoid vector::resize()'s
        // value-initialization in the timed path, but pre-fault the one shared
        // arena now so Top-k does not inherit those page faults instead.
        scratch.data.reset(new int32_t[required]);
        std::memset(scratch.data.get(), 0, required * sizeof(int32_t));
        scratch.capacity = required;
    }

    static int32_t *hmxIndicesData() {
        return sharedHmxIndexScratch().data.get();
    }

    static int hmxPipelineGroupHeads(int heads, int key_len) {
        const char *value = std::getenv("MLLM_HMX_PIPELINE_GROUP_HEADS");
        if (value == nullptr || value[0] == '\0') {
            // On the target CPU exact Top-k and sparse QK/PV contend for the
            // same cores and memory bandwidth. The measured resource-aware
            // default keeps the all-head barrier; 4/8 remain experimental
            // overrides for CPUs where the two stages scale independently.
            (void)heads;
            (void)key_len;
            return 0;
        }
        char *end = nullptr;
        const long parsed = std::strtol(value, &end, 10);
        if (end == value || *end != '\0' || parsed < 0 || parsed > heads) {
            throw std::runtime_error(
                "MLLM_HMX_PIPELINE_GROUP_HEADS must be 0 or between 1 and heads");
        }
        if (parsed != 0 && heads % parsed != 0) {
            throw std::runtime_error(
                "MLLM_HMX_PIPELINE_GROUP_HEADS must divide the configured "
                "head count");
        }
        return static_cast<int>(parsed);
    }

    static int sparseAttentionThreads(int model_threads) {
        const char *value = std::getenv("MLLM_SPARSE_ATTENTION_THREADS");
        if (value == nullptr || value[0] == '\0') return model_threads;
        char *end = nullptr;
        const long parsed = std::strtol(value, &end, 10);
        if (end == value || *end != '\0' || parsed <= 0 || parsed > 64) {
            throw std::runtime_error(
                "MLLM_SPARSE_ATTENTION_THREADS must be in [1, 64]");
        }
        return static_cast<int>(parsed);
    }

    static bool fusionEnabled(const char *name, bool default_value) {
        const char *value = std::getenv(name);
        if (value == nullptr || value[0] == '\0') return default_value;
        return !(value[0] == '0' && value[1] == '\0');
    }

    const int32_t *headRowOffsets(int head, int query_len) const {
        return pattern_offsets_.data()
            + static_cast<std::size_t>(head) * (query_len + 1);
    }

    void preparePatternIndices(int heads, int query_len, int key_len) {
        if (heads == pattern_heads_ && query_len == pattern_query_len_
            && key_len == pattern_key_len_) {
            return;
        }
        if (!head_retentions_.empty()
            && head_retentions_.size() != static_cast<std::size_t>(heads)) {
            throw std::runtime_error(
                "pattern sparse attention head-retention count does not "
                "match the runtime head count");
        }
        pattern_offsets_.assign(
            static_cast<std::size_t>(heads) * (query_len + 1), 0);
        pattern_head_bases_.assign(heads + 1, 0);
        max_topk_per_head_ = 0;
        for (int head = 0; head < heads; ++head) {
            const float density = head_retentions_.empty()
                ? 1.0F - sparsity_ : head_retentions_[head];
            int32_t *offsets = pattern_offsets_.data()
                + static_cast<std::size_t>(head) * (query_len + 1);
            for (int s = 0; s < query_len; ++s) {
                const int valid = causal_mask_ && query_len > 1
                    ? std::min(key_len, key_len - query_len + s + 1)
                    : key_len;
                const int keep = valid <= dense_tokens_
                    ? valid
                    : std::max(1, std::min(valid, static_cast<int>(
                          std::ceil(density * static_cast<float>(valid)))));
                offsets[s + 1] = offsets[s] + keep;
            }
            pattern_head_bases_[head + 1] =
                pattern_head_bases_[head]
                + static_cast<std::size_t>(offsets[query_len]);
            max_topk_per_head_ = std::max(
                max_topk_per_head_, static_cast<int>(offsets[query_len]));
        }
        if (random_pattern_ || hmx_selector_enabled_) {
            // Random indices depend on batch/head/query row and are generated
            // into thread-local scratch during execute().  Keeping only the
            // row sizes here avoids a large persistent index table.
            pattern_indices_.clear();
            pattern_heads_ = heads;
            pattern_query_len_ = query_len;
            pattern_key_len_ = key_len;
            return;
        }
        pattern_indices_.resize(pattern_head_bases_.back());
        for (int head = 0; head < heads; ++head) {
            const int32_t *offsets = headRowOffsets(head, query_len);
            for (int s = 0; s < query_len; ++s) {
                const int valid = causal_mask_ && query_len > 1
                    ? std::min(key_len, key_len - query_len + s + 1)
                    : key_len;
                const int keep = offsets[s + 1] - offsets[s];
                int32_t *indices = pattern_indices_.data()
                    + pattern_head_bases_[head] + offsets[s];
                buildPatternIndices(valid, keep, local_ratio_, prefix_tokens_,
                                    indices);
            }
        }
        pattern_heads_ = heads;
        pattern_query_len_ = query_len;
        pattern_key_len_ = key_len;
    }

    template <bool track_max>
    static float selectedQKScoresF16(
        const mllm_fp16_t *query, const mllm_fp16_t *key_base,
        int key_stride, const int32_t *selected, int keep, int head_dim,
        float scale, float *scores) {
        float row_max = -INFINITY;
#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_FML)
        int i = 0;
        // Eight independent dot products use 16 accumulators plus Q/K
        // temporaries, which fits the AArch64 vector register file. This
        // halves Q-vector loads while preserving each score's accumulation
        // order exactly.
        for (; i + 8 <= keep; i += 8) {
            if (i + 16 <= keep) {
                for (int j = 0; j < 8; ++j) {
                    const mllm_fp16_t *future_key =
                        key_base + selected[i + 8 + j] * key_stride;
                    for (int d = 0; d < head_dim; d += 32) {
                        __builtin_prefetch(future_key + d, 0, 1);
                    }
                }
            }
            const mllm_fp16_t *keys[8] = {
                key_base + selected[i] * key_stride,
                key_base + selected[i + 1] * key_stride,
                key_base + selected[i + 2] * key_stride,
                key_base + selected[i + 3] * key_stride,
                key_base + selected[i + 4] * key_stride,
                key_base + selected[i + 5] * key_stride,
                key_base + selected[i + 6] * key_stride,
                key_base + selected[i + 7] * key_stride,
            };
            float32x4_t acc_low[8] = {
                vdupq_n_f32(0.0F), vdupq_n_f32(0.0F),
                vdupq_n_f32(0.0F), vdupq_n_f32(0.0F),
                vdupq_n_f32(0.0F), vdupq_n_f32(0.0F),
                vdupq_n_f32(0.0F), vdupq_n_f32(0.0F),
            };
            float32x4_t acc_high[8] = {
                vdupq_n_f32(0.0F), vdupq_n_f32(0.0F),
                vdupq_n_f32(0.0F), vdupq_n_f32(0.0F),
                vdupq_n_f32(0.0F), vdupq_n_f32(0.0F),
                vdupq_n_f32(0.0F), vdupq_n_f32(0.0F),
            };
            int d = 0;
            for (; d + 8 <= head_dim; d += 8) {
                const float16x8_t q = vld1q_f16(
                    reinterpret_cast<const float16_t *>(query + d));
                for (int j = 0; j < 8; ++j) {
                    const float16x8_t k = vld1q_f16(
                        reinterpret_cast<const float16_t *>(keys[j] + d));
                    acc_low[j] = vfmlalq_low_f16(acc_low[j], q, k);
                    acc_high[j] = vfmlalq_high_f16(acc_high[j], q, k);
                }
            }
            for (int j = 0; j < 8; ++j) {
                float sum = vaddvq_f32(vaddq_f32(acc_low[j], acc_high[j]));
                for (int tail = d; tail < head_dim; ++tail) {
                    sum += MLLM_FP16_TO_FP32(query[tail])
                        * MLLM_FP16_TO_FP32(keys[j][tail]);
                }
                const float score = sum * scale;
                scores[i + j] = score;
                if constexpr (track_max) {
                    row_max = std::max(row_max, score);
                }
            }
        }
        for (; i < keep; ++i) {
            const mllm_fp16_t *key = key_base + selected[i] * key_stride;
            float32x4_t acc_low = vdupq_n_f32(0.0F);
            float32x4_t acc_high = vdupq_n_f32(0.0F);
            int d = 0;
            for (; d + 8 <= head_dim; d += 8) {
                const float16x8_t q = vld1q_f16(
                    reinterpret_cast<const float16_t *>(query + d));
                const float16x8_t k = vld1q_f16(
                    reinterpret_cast<const float16_t *>(key + d));
                acc_low = vfmlalq_low_f16(acc_low, q, k);
                acc_high = vfmlalq_high_f16(acc_high, q, k);
            }
            float sum = vaddvq_f32(vaddq_f32(acc_low, acc_high));
            for (; d < head_dim; ++d) {
                sum += MLLM_FP16_TO_FP32(query[d])
                    * MLLM_FP16_TO_FP32(key[d]);
            }
            const float score = sum * scale;
            scores[i] = score;
            if constexpr (track_max) {
                row_max = std::max(row_max, score);
            }
        }
#else
        for (int i = 0; i < keep; ++i) {
            vec_dot_fp16(head_dim, scores + i, query,
                         key_base + selected[i] * key_stride);
            scores[i] *= scale;
            if constexpr (track_max) {
                row_max = std::max(row_max, scores[i]);
            }
        }
#endif
        return row_max;
    }

    void computeSelectedMatrixRange(
        const std::shared_ptr<Tensor> &query,
        const std::shared_ptr<Tensor> &key,
        const std::shared_ptr<Tensor> &value,
        const std::shared_ptr<Tensor> &output,
        PackedValueState *packed_key_state,
        PackedValueState *packed_value_state,
        int source_key_stride,
        std::size_t matrix_begin,
        std::size_t matrix_count,
        bool profile_enabled,
        int worker_threads = 0,
        int query_row_begin = 0,
        int query_row_count = -1) {
        const int heads = query->head();
        const int query_len = query->sequence();
        const int key_len = key->sequence();
        const int head_dim = query->dimension();
        const int topk_per_head = max_topk_per_head_;
        const float qk_scale = 1.0F
            / std::sqrt(static_cast<float>(head_dim));
        const int key_stride = packed_key_state != nullptr
            ? head_dim : source_key_stride;
        const bool fuse_qk_max = fusionEnabled(
            "MLLM_SPARSE_FUSE_QK_MAX", true);
        const bool fuse_index_offset = fusionEnabled(
            "MLLM_SPARSE_FUSE_INDEX_OFFSET", false);
        const bool detailed_profile_enabled = profile_enabled
            && CPUAttentionProfiler::detailedEnabled();
        struct WorkStats {
            uint64_t qk_softmax_work_ns = 0;
            uint64_t qk_work_ns = 0;
            uint64_t softmax_work_ns = 0;
            uint64_t normalize_prep_work_ns = 0;
            uint64_t pv_work_ns = 0;
            uint64_t eligible_elements = 0;
            uint64_t retained_elements = 0;

            void add(const WorkStats &other) {
                qk_softmax_work_ns += other.qk_softmax_work_ns;
                qk_work_ns += other.qk_work_ns;
                softmax_work_ns += other.softmax_work_ns;
                normalize_prep_work_ns += other.normalize_prep_work_ns;
                pv_work_ns += other.pv_work_ns;
                eligible_elements += other.eligible_elements;
                retained_elements += other.retained_elements;
            }
        };
        const int active_row_begin = std::clamp(
            query_row_begin, 0, query_len);
        const int active_row_count = query_row_count < 0
            ? query_len - active_row_begin
            : std::clamp(query_row_count, 0,
                         query_len - active_row_begin);
        if (active_row_count == 0 || matrix_count == 0) return;
        const std::int64_t range_rows =
            static_cast<std::int64_t>(matrix_count) * active_row_count;
        const int compute_threads = worker_threads > 0
            ? worker_threads : thread_count_;
        const auto compute_row = [&](std::int64_t local_row,
                                     ThreadScratch &scratch,
                                     WorkStats &stats) {
            float *weights = scratch.weights.data();
            int32_t *selected_offsets = scratch.selected_indices.data();
            uint64_t *random_bits = scratch.random_bits.data();
            mllm_fp16_t *weights_f16 = scratch.weights_f16.data();
            mllm_fp16_t *query_f16 = scratch.query_f16.data();
            const uint64_t qk_start_ns = detailed_profile_enabled
                ? CPUAttentionProfiler::nowNs() : 0;
            const std::size_t matrix = matrix_begin
                + static_cast<std::size_t>(local_row / active_row_count);
            const int s = active_row_begin
                + static_cast<int>(local_row % active_row_count);
            const int h = static_cast<int>(matrix
                % static_cast<std::size_t>(heads));
            const int b = static_cast<int>(matrix
                / static_cast<std::size_t>(heads));
            const int valid = causal_mask_ && query_len > 1
                ? std::min(key_len, key_len - query_len + s + 1)
                : key_len;
            const int32_t *offsets = headRowOffsets(h, query_len);
            const int begin = offsets[s];
            const int keep = offsets[s + 1] - begin;
            const int32_t *selected;
            if (hmx_selector_enabled_) {
                selected = hmxIndicesData()
                    + matrix * static_cast<std::size_t>(topk_per_head)
                    + begin;
            } else if (random_pattern_) {
                const int absolute_query = key_len - query_len + s;
                buildRandomPatternIndices(
                    valid, keep,
                    makeRandomRowSeed(random_seed_, absolute_query, b, h),
                    selected_offsets, random_bits);
                selected = selected_offsets;
            } else {
                selected = pattern_indices_.data()
                    + pattern_head_bases_[h] + begin;
            }

            const float *query_row = query->ptrAt<float>(b, h, s, 0);
            float row_max = -INFINITY;
            if (key->dtype() == MLLM_TYPE_F16) {
                mllm_fp32_to_fp16_row(query_row, query_f16, head_dim);
                const mllm_fp16_t *key_base = packed_key_state != nullptr
                    ? packed_key_state->head_data[matrix].data.get()
                    : key->ptrAt<mllm_fp16_t>(b, h, 0, 0);
                if (fuse_qk_max) {
                    row_max = selectedQKScoresF16<true>(
                        query_f16, key_base, key_stride, selected, keep,
                        head_dim, qk_scale, weights);
                } else {
                    selectedQKScoresF16<false>(
                        query_f16, key_base, key_stride, selected, keep,
                        head_dim, qk_scale, weights);
                }
            } else {
                const float *key_base = key->ptrAt<float>(b, h, 0, 0);
                for (int i = 0; i < keep; ++i) {
                    vec_dot_fp32(
                        head_dim, weights + i, query_row,
                        key_base + selected[i] * key_stride);
                    weights[i] *= qk_scale;
                    if (fuse_qk_max) {
                        row_max = std::max(row_max, weights[i]);
                    }
                }
            }
            const uint64_t softmax_start_ns = detailed_profile_enabled
                ? CPUAttentionProfiler::nowNs() : 0;
            if (detailed_profile_enabled) {
                stats.qk_work_ns += softmax_start_ns - qk_start_ns;
            }
            if (!fuse_qk_max) {
                for (int i = 0; i < keep; ++i) {
                    row_max = std::max(row_max, weights[i]);
                }
            }
            const float row_sum = mllm_vec_soft_max_f32(
                keep, weights, weights, row_max);
            const float inv_sum = 1.0F / row_sum;
            const uint64_t pv_start_ns = detailed_profile_enabled
                ? CPUAttentionProfiler::nowNs() : 0;
            if (detailed_profile_enabled) {
                stats.qk_softmax_work_ns += pv_start_ns - qk_start_ns;
                stats.softmax_work_ns += pv_start_ns - softmax_start_ns;
            }

            float *output_row = output->ptrAt<float>(b, h, s, 0);
            if (value->dtype() == MLLM_TYPE_F16) {
                assert(packed_value_state != nullptr);
                const uint64_t normalize_prep_start_ns =
                    detailed_profile_enabled
                    ? CPUAttentionProfiler::nowNs() : 0;
                for (int i = 0; i < keep; ++i) {
                    weights_f16[i] = MLLM_FP32_TO_FP16(weights[i] * inv_sum);
                    if (!fuse_index_offset) {
                        selected_offsets[i] = selected[i] * head_dim;
                    }
                }
                if (detailed_profile_enabled) {
                    stats.normalize_prep_work_ns +=
                        CPUAttentionProfiler::nowNs()
                        - normalize_prep_start_ns;
                }
                const mllm_fp16_t *packed_head =
                    packed_value_state->head_data[matrix].data.get();
                if (fuse_index_offset) {
                    sparseValueF16Selected(
                        packed_head, selected, weights_f16, keep, head_dim,
                        output_row);
                } else {
                    sparseValueF16(
                        packed_head, selected_offsets, weights_f16, keep,
                        head_dim, output_row);
                }
            } else {
                vec_scale_f32(keep, weights, inv_sum);
                for (int d = 0; d < head_dim; ++d) {
                    const float *value_line =
                        value->ptrAt<float>(b, h, 0, d);
                    float sum = 0.0F;
                    for (int i = 0; i < keep; ++i) {
                        sum += weights[i] * value_line[selected[i]];
                    }
                    output_row[d] = sum;
                }
            }
            if (detailed_profile_enabled) {
                stats.pv_work_ns +=
                    CPUAttentionProfiler::nowNs() - pv_start_ns;
            }
            stats.eligible_elements += static_cast<uint64_t>(valid);
            stats.retained_elements += static_cast<uint64_t>(keep);
        };

        WorkStats total_stats;
        if (compute_threads == 1) {
            // Pipeline parallelism lives in the process-wide stage pool. A
            // stage task must not create a nested OpenMP team.
            static thread_local ThreadScratch pipeline_scratch;
            pipeline_scratch.resize(std::max(key_len, head_dim));
            for (std::int64_t local_row = 0; local_row < range_rows;
                 ++local_row) {
                compute_row(local_row, pipeline_scratch, total_stats);
            }
        } else {
            std::vector<WorkStats> worker_stats(
                static_cast<std::size_t>(compute_threads));
#pragma omp parallel num_threads(compute_threads)
            {
                const int worker = omp_get_thread_num();
                auto &scratch = thread_scratch_[worker];
#pragma omp for schedule(static)
                for (std::int64_t local_row = 0; local_row < range_rows;
                     ++local_row) {
                    compute_row(local_row, scratch, worker_stats[worker]);
                }
            }
            for (const auto &stats : worker_stats) total_stats.add(stats);
        }
        if (detailed_profile_enabled) {
            CPUAttentionProfiler::add(
                AttentionProfileStage::PATTERN_QK_SOFTMAX_WORK,
                total_stats.qk_softmax_work_ns);
            CPUAttentionProfiler::add(
                AttentionProfileStage::PATTERN_QK_WORK,
                total_stats.qk_work_ns);
            CPUAttentionProfiler::add(
                AttentionProfileStage::PATTERN_SOFTMAX_WORK,
                total_stats.softmax_work_ns);
            CPUAttentionProfiler::add(
                AttentionProfileStage::PATTERN_NORMALIZE_PREP_WORK,
                total_stats.normalize_prep_work_ns);
            CPUAttentionProfiler::add(
                AttentionProfileStage::SPARSE_PV_WORK,
                total_stats.pv_work_ns);
        }
        CPUSparseSelectionStats::add(
            total_stats.eligible_elements, total_stats.retained_elements);
    }

public:
    CPUPatternSparseAttentionFunc(Backend *bn, string name, int thread_count,
                                  float sparsity, bool causal_mask,
                                  float local_ratio = 0.5F,
                                  int prefix_tokens = 0,
                                  int dense_tokens = 0,
                                  bool random_pattern = false,
                                  uint32_t random_seed = 1,
                                  int pack_reserve_tokens = 0,
                                  bool hmx_selector = false,
                                  std::vector<float> head_retentions = {},
                                  int layer_id = -1) :
        CPUSparseSoftmaxValueFunc(bn, std::move(name), thread_count, sparsity,
                                  causal_mask, 0, std::move(head_retentions)),
        local_ratio_(local_ratio),
        prefix_tokens_(prefix_tokens),
        dense_tokens_(dense_tokens),
        random_pattern_(random_pattern),
        hmx_selector_enabled_(hmx_selector),
        random_seed_(random_seed),
        layer_id_(layer_id) {
        assert(local_ratio_ >= 0.0F && local_ratio_ <= 1.0F);
        assert(prefix_tokens_ >= 0);
        assert(dense_tokens_ >= 0);
        assert(pack_reserve_tokens >= 0);
        assert(!(random_pattern_ && hmx_selector_enabled_));
        pack_reserve_tokens_ = pack_reserve_tokens;
        if (hmx_selector_enabled_) {
            hmx_int8_selector_ = std::make_unique<HMXInt8Selector>(
                static_cast<size_t>(pack_reserve_tokens));
        }
    }

    // Floyd sampling selects exactly `keep` keys from the whole unmasked
    // interval without replacement in O(keep + valid/64), then enumerates the
    // bitset in key order for cache-friendlier QK/PV access.
    static void buildRandomPatternIndices(int valid, int keep, uint64_t seed,
                                          int32_t *indices) {
        std::vector<uint64_t> random_bits((valid + 63) / 64, 0);
        buildRandomPatternIndices(valid, keep, seed, indices,
                                  random_bits.data());
    }

    static void buildRandomPatternIndices(int valid, int keep, uint64_t seed,
                                          int32_t *indices,
                                          uint64_t *random_bits) {
        assert(valid > 0 && keep > 0 && keep <= valid);
        assert(random_bits != nullptr);
        if (keep == valid) {
            for (int i = 0; i < valid; ++i) indices[i] = i;
            return;
        }
        uint32_t random = static_cast<uint32_t>(seed)
            ^ static_cast<uint32_t>(seed >> 32) ^ 0x6d2b79f5U;
        if (random == 0) random = 0x9e3779b9U;
        const auto next_random = [&random]() {
            random ^= random << 13;
            random ^= random >> 17;
            random ^= random << 5;
            return random;
        };
        const auto uniform_bounded = [&next_random](uint32_t bound) {
            assert(bound > 0);
            while (true) {
                const uint32_t value = next_random();
                const uint64_t product =
                    static_cast<uint64_t>(value) * bound;
                const uint32_t low = static_cast<uint32_t>(product);
                if (low < bound) {
                    const uint32_t threshold =
                        static_cast<uint32_t>(-bound) % bound;
                    if (low < threshold) continue;
                }
                return static_cast<uint32_t>(product >> 32);
            }
        };
#ifndef NDEBUG
        for (int word = 0; word < (valid + 63) / 64; ++word) {
            assert(random_bits[word] == 0);
        }
#endif
        for (int upper = valid - keep; upper < valid; ++upper) {
            const int candidate = static_cast<int>(
                uniform_bounded(static_cast<uint32_t>(upper + 1)));
            const uint64_t candidate_mask =
                uint64_t{1} << (candidate & 63);
            const bool candidate_already_selected =
                (random_bits[candidate >> 6] & candidate_mask) != 0;
            const int selected = candidate_already_selected
                ? upper : candidate;
            const uint64_t selected_mask = uint64_t{1} << (selected & 63);
            assert((random_bits[selected >> 6] & selected_mask) == 0);
            random_bits[selected >> 6] |= selected_mask;
        }
        int output_index = 0;
        const int word_count = (valid + 63) / 64;
        for (int word_index = 0; word_index < word_count; ++word_index) {
            uint64_t word = random_bits[word_index];
            random_bits[word_index] = 0;
            while (word != 0) {
                const int bit = __builtin_ctzll(word);
                indices[output_index++] =
                    static_cast<int32_t>(word_index * 64 + bit);
                word &= word - 1;
            }
        }
        assert(output_index == keep);
#ifndef NDEBUG
        for (int i = 0; i < keep; ++i) {
            assert(indices[i] >= 0 && indices[i] < valid);
            assert(i == 0 || indices[i - 1] < indices[i]);
        }
#endif
    }

    static uint64_t makeRandomRowSeed(uint32_t seed, int absolute_query,
                                      int batch = 0, int head = 0) {
        assert(absolute_query >= 0 && batch >= 0 && head >= 0);
        uint64_t mixed = (static_cast<uint64_t>(seed) << 32)
            ^ (static_cast<uint64_t>(
                   static_cast<uint32_t>(absolute_query + 1))
               * 0x9e3779b97f4a7c15ULL)
            ^ (static_cast<uint64_t>(static_cast<uint32_t>(batch + 1))
               * 0xbf58476d1ce4e5b9ULL)
            ^ (static_cast<uint64_t>(static_cast<uint32_t>(head + 1))
               * 0x94d049bb133111ebULL);
        mixed = (mixed ^ (mixed >> 30)) * 0xbf58476d1ce4e5b9ULL;
        mixed = (mixed ^ (mixed >> 27)) * 0x94d049bb133111ebULL;
        return mixed ^ (mixed >> 31);
    }

    static void buildPatternIndices(int valid, int keep, float local_ratio,
                                    int prefix_tokens, int32_t *indices) {
        assert(valid > 0 && keep > 0 && keep <= valid);
        assert(local_ratio >= 0.0F && local_ratio <= 1.0F);
        assert(prefix_tokens >= 0);
        const int local_keep = std::max(0, std::min(keep, static_cast<int>(
            std::ceil(local_ratio * static_cast<float>(keep)))));
        const int history_end = valid - local_keep;
        const int nonlocal_keep = keep - local_keep;
        const int prefix_keep = std::min(
            prefix_tokens, std::min(nonlocal_keep, history_end));
        for (int i = 0; i < prefix_keep; ++i) {
            indices[i] = i;
        }
        const int uniform_keep = nonlocal_keep - prefix_keep;
        const int uniform_begin = prefix_keep;
        const int uniform_len = history_end - uniform_begin;
        if (uniform_keep == 1) {
            indices[prefix_keep] = uniform_begin;
        } else if (uniform_keep > 1) {
            for (int i = 0; i < uniform_keep; ++i) {
                indices[prefix_keep + i] = static_cast<int32_t>(
                    uniform_begin + static_cast<int64_t>(i) * (uniform_len - 1)
                    / (uniform_keep - 1));
            }
        }
        for (int i = 0; i < local_keep; ++i) {
            indices[nonlocal_keep + i] = history_end + i;
        }
#ifndef NDEBUG
        for (int i = 0; i < keep; ++i) {
            assert(indices[i] >= 0 && indices[i] < valid);
            assert(i == 0 || indices[i - 1] < indices[i]);
        }
#endif
    }

    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs,
                      vector<shared_ptr<Tensor>> outputs) override {
        assert(inputs.size() == 3 && outputs.size() == 1);
        const auto &query = inputs[0];
        const auto &key = inputs[1];
        const auto &value = inputs[2];
        if (value->ctype() == BSHD) {
            transposeAttentionValueChannels(*value);
        }
        if ((query->ctype() != BSHD && query->ctype() != BHSD)
            || (key->ctype() != BSHD && key->ctype() != BHSD)
            || value->ctype() != BHDS) {
            return ::INVALID_VALUE;
        }
        if (query->batch() != key->batch()
            || query->batch() != value->batch()
            || query->head() != key->head()
            || query->head() != value->head()
            || query->dimension() != key->dimension()
            || query->dimension() != value->dimension()
            || key->sequence() != value->sequence()
            || key->sequence() < query->sequence()) {
            return ::INVALID_VALUE;
        }
        outputs[0]->setCtype(query->ctype());
        outputs[0]->setDtype(MLLM_TYPE_F32);
        outputs[0]->reshape(query->batch(), query->head(), query->sequence(),
                            query->dimension());
        return MLLM_NO_ERROR;
    }

    ErrorCode setUp(vector<shared_ptr<Tensor>> inputs,
                    vector<shared_ptr<Tensor>> outputs) override {
        if (inputs[2]->ctype() == BSHD) {
            transposeAttentionValueChannels(*inputs[2]);
        }
        if (inputs[2]->ctype() != BHDS) return ::INVALID_VALUE;
        outputs[0]->setCtype(inputs[0]->ctype());
        outputs[0]->setDtype(MLLM_TYPE_F32);
        outputs[0]->alloc();
        if (hmx_selector_enabled_) {
            if (inputs[0]->dtype() != MLLM_TYPE_F32
                || inputs[1]->dtype() != MLLM_TYPE_F16) {
                throw std::runtime_error(
                    "INT8 HMX attention selector requires F32 Q and F16 K");
            }
            const int reserve_tokens = std::max(
                inputs[1]->sequence(), pack_reserve_tokens_);
            const std::size_t matrix_count =
                static_cast<std::size_t>(inputs[0]->batch())
                * static_cast<std::size_t>(inputs[0]->head());
            // Build the maximum-shape row plan during setup. Runtime calls
            // with shorter K lengths will rewrite only the small offset table.
            preparePatternIndices(
                inputs[0]->head(), inputs[0]->sequence(), reserve_tokens);
            reserveHmxIndices(
                matrix_count * static_cast<std::size_t>(max_topk_per_head_));
            hmx_key_heads_.resize(matrix_count);
            hmx_query_f32_heads_.resize(matrix_count);
            std::string error;
            if (!hmx_int8_selector_->reserve(
                    inputs[0]->batch(), inputs[0]->head(),
                    inputs[0]->sequence(), reserve_tokens,
                    inputs[0]->dimension(), &error)) {
                throw std::runtime_error(
                    "failed to reserve INT8 HMX attention selector: "
                    + error);
            }
        }
        return MLLM_NO_ERROR;
    }

    ErrorCode execute(vector<shared_ptr<Tensor>> inputs,
                      vector<shared_ptr<Tensor>> outputs) override {
        std::unique_lock<std::recursive_mutex> hmx_index_scratch_lock;
        if (hmx_selector_enabled_) {
            hmx_index_scratch_lock = std::unique_lock<std::recursive_mutex>(
                sharedHmxIndexScratch().mutex);
        }
        const bool profile_enabled = CPUAttentionProfiler::enabled();
        ScopedAttentionProfile total_profile(
            AttentionProfileStage::SPARSE_TOTAL, profile_enabled);
        const auto &query = inputs[0];
        const auto &key = inputs[1];
        const auto &value = inputs[2];
        const auto &output = outputs[0];

        if (query->dtype() != MLLM_TYPE_F32
            || (key->dtype() != MLLM_TYPE_F16
                && key->dtype() != MLLM_TYPE_F32)
            || (value->dtype() != MLLM_TYPE_F16
                && value->dtype() != MLLM_TYPE_F32)
            || value->ctype() != BHDS) {
            return ::INVALID_VALUE;
        }

        const int batch = query->batch();
        const int heads = query->head();
        const int query_len = query->sequence();
        const int key_len = key->sequence();
        const int head_dim = query->dimension();
        if (query_len <= 0 || key_len < query_len || head_dim <= 0) {
            return ::INVALID_VALUE;
        }
        const int source_key_stride = key->sequenceSkipDim();
        if (source_key_stride <= 0) return ::INVALID_VALUE;
        if (key->masterTensor() != nullptr) {
            const auto offset = key->shapeOffset();
            const auto master_shape = key->shapeMaster();
            if (offset.size() != 4 || master_shape.size() != 4
                || offset[2] < 0 || master_shape[2] < 0
                || static_cast<int64_t>(offset[2]) + key_len
                    > master_shape[2]) {
                return ::INVALID_VALUE;
            }
        }
        const int sparse_threads = sparseAttentionThreads(thread_count_);
        ensureScratchCapacity(std::max(key_len, head_dim), sparse_threads);
        preparePatternIndices(heads, query_len, key_len);
        PackedValueState *packed_key_state = nullptr;
        if (key->dtype() == MLLM_TYPE_F16 && key->ctype() == BSHD
            && source_key_stride > head_dim) {
            packed_key_state = preparePackedKeyF16(
                key, batch, heads, query_len, key_len, head_dim,
                profile_enabled);
        }
        PackedValueState *packed_state = nullptr;
        if (value->dtype() == MLLM_TYPE_F16) {
            packed_state = preparePackedValueF16(
                value, batch, heads, query_len, key_len, head_dim,
                profile_enabled);
        }

        const int topk_per_head = max_topk_per_head_;
        bool sparse_pipeline_complete = false;
        if (hmx_selector_enabled_) {
            if (key->dtype() != MLLM_TYPE_F16 || hmx_int8_selector_ == nullptr) {
                throw std::runtime_error(
                    "INT8 HMX attention selector received a non-F16 K cache");
            }
            ScopedAttentionProfile hmx_profile(
                AttentionProfileStage::HMX_SELECT, profile_enabled);
            const size_t matrix_count = static_cast<size_t>(batch) * heads;
            {
                ScopedAttentionProfile prepare_q_profile(
                    AttentionProfileStage::HMX_PREPARE_Q, profile_enabled);
                hmx_key_heads_.resize(matrix_count);
                reserveHmxIndices(
                    matrix_count * static_cast<size_t>(topk_per_head));
                hmx_query_f32_heads_.resize(matrix_count);
                for (int matrix = 0;
                     matrix < static_cast<int>(matrix_count); ++matrix) {
                    const int b = matrix / heads;
                    const int h = matrix % heads;
                    hmx_query_f32_heads_[matrix] =
                        query->ptrAt<float>(b, h, 0, 0);
                    const mllm_fp16_t *packed_key =
                        packed_key_state != nullptr
                        ? packed_key_state->head_data[matrix].data.get()
                        : key->ptrAt<mllm_fp16_t>(b, h, 0, 0);
                    hmx_key_heads_[matrix] = reinterpret_cast<
                        const HMXInt8Selector::Half *>(packed_key);
                }
            }

            HMXInt8Selector::Request request;
            const int query_row_stride = query->sequenceSkipDim();
            if (query_row_stride < head_dim) return ::INVALID_VALUE;
            request.query_f32_heads = hmx_query_f32_heads_.data();
            request.query_f32_row_stride = static_cast<std::size_t>(
                query_row_stride);
            request.key_heads = hmx_key_heads_.data();
            request.batch = batch;
            request.heads = heads;
            request.query_len = query_len;
            request.key_len = key_len;
            request.head_dim = head_dim;
            request.row_offsets = pattern_offsets_.data();
            request.row_offsets_head_stride =
                static_cast<std::size_t>(query_len + 1);
            request.causal_prefix_tokens = causal_mask_
                ? key_len - query_len : key_len - 1;
            request.topk_indices = hmxIndicesData();
            request.topk_head_stride = static_cast<size_t>(topk_per_head);
            request.thread_count = thread_count_;
            request.layer_id = layer_id_;
            request.head_retentions = head_retentions_.empty()
                ? nullptr : head_retentions_.data();
            request.head_retention_count = head_retentions_.size();
            request.diagnostic_label = name();
            if (!head_retentions_.empty()) {
                request.pipeline_group_heads =
                    hmxPipelineGroupHeads(heads, key_len);
                request.topk_group_ready =
                    [&, packed_key_state, packed_state](
                        std::size_t matrix_begin,
                        std::size_t matrix_count) {
                        computeSelectedMatrixRange(
                            query, key, value, output, packed_key_state,
                            packed_state, source_key_stride, matrix_begin,
                            matrix_count, profile_enabled, 1);
                    };
                request.topk_group_rows_ready =
                    [&, packed_key_state, packed_state](
                        std::size_t matrix_begin,
                        std::size_t matrix_count,
                        int query_row_begin,
                        int query_row_count) {
                        computeSelectedMatrixRange(
                            query, key, value, output, packed_key_state,
                            packed_state, source_key_stride, matrix_begin,
                            matrix_count, profile_enabled, 1,
                            query_row_begin, query_row_count);
                    };
            }
            std::string error;
            if (!hmx_int8_selector_->select(request, &error)) {
                throw std::runtime_error(
                    "INT8 HMX attention selection failed: " + error);
            }
            sparse_pipeline_complete = request.pipeline_group_heads > 0;
        }
        if (!sparse_pipeline_complete) {
            const std::uint64_t serial_sparse_begin_ns =
                profile_enabled && hmx_selector_enabled_
                ? CPUAttentionProfiler::nowNs() : 0;
            computeSelectedMatrixRange(
                query, key, value, output, packed_key_state, packed_state,
                source_key_stride, 0,
                static_cast<std::size_t>(batch) * heads, profile_enabled,
                sparse_threads);
            if (profile_enabled && hmx_selector_enabled_) {
                CPUAttentionProfiler::add(
                    AttentionProfileStage::HMX_SPARSE_STAGE_WALL,
                    CPUAttentionProfiler::nowNs()
                        - serial_sparse_begin_ns);
            }
        }
        return MLLM_NO_ERROR;
    }
};

class CPUPatternSparseAttentionFuncCreator final : public CPUBackend::Creator {
public:
    Op *create(OpParam op_param, Backend *bn, string name,
               int thread_count) const override {
        const int head_retention_count = static_cast<int>(
            op_param.at("head_retention_count"));
        std::vector<float> head_retentions;
        head_retentions.reserve(head_retention_count);
        for (int head = 0; head < head_retention_count; ++head) {
            head_retentions.push_back(
                op_param.at("head_retention_" + std::to_string(head)));
        }
        return new CPUPatternSparseAttentionFunc(
            bn, std::move(name), thread_count, op_param.at("sparsity"),
            static_cast<bool>(op_param.at("causal_mask")),
            op_param.at("local_ratio"),
            static_cast<int>(op_param.at("prefix_tokens")),
            static_cast<int>(op_param.at("dense_tokens")),
            static_cast<bool>(op_param.at("random_pattern")),
            static_cast<uint32_t>(op_param.at("random_seed")),
            static_cast<int>(op_param.at("pack_reserve_tokens")),
            static_cast<bool>(op_param.at("hmx_selector")),
            std::move(head_retentions),
            static_cast<int>(op_param.at("layer_id")));
    }
};

} // namespace mllm

#endif // CPU_SPARSE_SOFTMAX_VALUE_FUNC_HPP
