#include "CPUTest.hpp"
#include "backends/cpu/AttentionProfiler.hpp"
#include "backends/cpu/op/CPUKVCacheNPU.hpp"
#include "backends/cpu/op/CPUSparseSoftmaxValueFunc.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <initializer_list>
#include <string>
#include <vector>

namespace {

class ScopedEnvironmentOverride {
public:
    ScopedEnvironmentOverride(const char *name, const char *value) :
        name_(name) {
        if (const char *previous = std::getenv(name)) {
            had_previous_ = true;
            previous_ = previous;
        }
        EXPECT_EQ(setenv(name_.c_str(), value, 1), 0);
    }

    ~ScopedEnvironmentOverride() {
        if (had_previous_) {
            setenv(name_.c_str(), previous_.c_str(), 1);
        } else {
            unsetenv(name_.c_str());
        }
    }

    void set(const char *value) {
        EXPECT_EQ(setenv(name_.c_str(), value, 1), 0);
    }

private:
    std::string name_;
    std::string previous_;
    bool had_previous_ = false;
};

std::shared_ptr<Tensor> makeF32Query(Backend *backend, int query_len,
                                     int head_dim,
                                     const std::vector<float> &data) {
    EXPECT_EQ(data.size(), static_cast<size_t>(query_len * head_dim));
    auto query = std::make_shared<Tensor>(backend);
    query->setCtype(BSHD);
    query->setDtype(MLLM_TYPE_F32);
    query->reshape(1, 1, query_len, head_dim);
    query->alloc();
    for (int s = 0; s < query_len; ++s) {
        for (int d = 0; d < head_dim; ++d) {
            query->setDataAt<float>(
                0, 0, s, d, data[static_cast<size_t>(s) * head_dim + d]);
        }
    }
    return query;
}

std::shared_ptr<Tensor> makeF16Sequence(Backend *backend, ChlType ctype,
                                        int sequence, int dimension,
                                        const std::vector<float> &data) {
    EXPECT_EQ(data.size(), static_cast<size_t>(sequence * dimension));
    auto tensor = std::make_shared<Tensor>(backend);
    tensor->setCtype(ctype);
    tensor->setDtype(MLLM_TYPE_F16);
    tensor->reshape(1, 1, sequence, dimension);
    tensor->alloc();
    for (int s = 0; s < sequence; ++s) {
        for (int d = 0; d < dimension; ++d) {
            tensor->setDataAt<mllm_fp16_t>(
                0, 0, s, d,
                MLLM_FP32_TO_FP16(
                    data[static_cast<size_t>(s) * dimension + d]));
        }
    }
    return tensor;
}

float selectedSoftmaxValue(const std::vector<float> &scores,
                           const std::vector<float> &values,
                           std::initializer_list<int> selected_keys) {
    float maximum = -INFINITY;
    for (const int key : selected_keys) {
        maximum = std::max(maximum, scores[key]);
    }
    float denominator = 0.0F;
    float numerator = 0.0F;
    for (const int key : selected_keys) {
        const float weight = std::exp(scores[key] - maximum);
        denominator += weight;
        numerator += weight * values[key];
    }
    return numerator / denominator;
}

void runPatternAttention(CPUPatternSparseAttentionFunc &op,
                         const std::shared_ptr<Tensor> &query,
                         const std::shared_ptr<Tensor> &key,
                         const std::shared_ptr<Tensor> &value,
                         const std::shared_ptr<Tensor> &output) {
    ASSERT_EQ(op.reshape({query, key, value}, {output}), MLLM_NO_ERROR);
    ASSERT_EQ(op.setUp({query, key, value}, {output}), MLLM_NO_ERROR);
    ASSERT_EQ(op.execute({query, key, value}, {output}), MLLM_NO_ERROR);
}

} // namespace

TEST_F(CPUTest, CPUPatternSparseAttentionCausalOldPrefixMasksFutureKeys) {
    // key_len - query_len == 2, so the two rows have valid lengths 3 and 4.
    // A very large future score at key 3 must not affect row 0.
    auto query = makeF32Query(bn_, 2, 1, {1.0F, 1.0F});
    const std::vector<float> key_data = {0.0F, 1.0F, 2.0F, 100.0F};
    const std::vector<float> value_data = {10.0F, 20.0F, 30.0F, 40.0F};
    auto key = makeF16Sequence(bn_, BSHD, 4, 1, key_data);
    auto value = makeF16Sequence(bn_, BHDS, 4, 1, value_data);
    auto output = std::make_shared<Tensor>(bn_);

    // keep(row 0)=ceil(0.5*3)=2 -> local keys {1,2};
    // keep(row 1)=ceil(0.5*4)=2 -> local keys {2,3}.
    CPUPatternSparseAttentionFunc op(
        bn_, "pattern-causal-old-prefix", 2, 0.5F, true, 1.0F);
    CPUSparseSelectionStats::reset();
    runPatternAttention(op, query, key, value, output);
    const auto stats = CPUSparseSelectionStats::snapshot();

    EXPECT_NEAR(output->dataAt<float>(0, 0, 0, 0),
                selectedSoftmaxValue(key_data, value_data, {1, 2}), 3.0e-2F);
    EXPECT_NEAR(output->dataAt<float>(0, 0, 1, 0),
                selectedSoftmaxValue(key_data, value_data, {2, 3}), 3.0e-2F);
    EXPECT_EQ(stats.eligible, 7U);
    EXPECT_EQ(stats.retained, 4U);
}

TEST_F(CPUTest, CPUPatternSparseAttentionUsesLocalAndUniformExpectedKeys) {
    auto query = makeF32Query(bn_, 1, 1, {1.0F});
    auto key = makeF16Sequence(
        bn_, BSHD, 12, 1, std::vector<float>(12, 0.0F));
    const std::vector<float> value_data = {
        1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F,
        7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F};
    auto value = makeF16Sequence(bn_, BHDS, 12, 1, value_data);
    auto output = std::make_shared<Tensor>(bn_);

    // keep=ceil(0.5*12)=6 and local_keep=ceil(0.5*6)=3.
    // The history is [0,9); its three endpoint-linspace keys are {0,4,8},
    // followed by the local suffix {9,10,11}. Equal QK scores make the result
    // the arithmetic mean of exactly those six values.
    CPUPatternSparseAttentionFunc op(
        bn_, "pattern-local-uniform", 2, 0.5F, false, 0.5F);
    CPUSparseSelectionStats::reset();
    runPatternAttention(op, query, key, value, output);
    const auto stats = CPUSparseSelectionStats::snapshot();

    const float expected =
        (value_data[0] + value_data[4] + value_data[8]
         + value_data[9] + value_data[10] + value_data[11]) / 6.0F;
    EXPECT_NEAR(output->dataAt<float>(0, 0, 0, 0), expected, 1.0e-2F);
    EXPECT_EQ(stats.eligible, 12U);
    EXPECT_EQ(stats.retained, 6U);
}

TEST_F(CPUTest, CPUPatternSparseAttentionReadsF16KeyAndValue) {
    auto query = makeF32Query(bn_, 1, 2, {1.0F, 2.0F});
    // At 50% sparsity with local_ratio=0.5, keys {0,3} are retained. Give
    // those keys equal K vectors; huge unselected K values must not turn the
    // fixed pattern into score-based Top-K selection.
    auto key = makeF16Sequence(
        bn_, BSHD, 4, 2,
        {0.5F, -0.25F, 100.0F, 100.0F,
         200.0F, 200.0F, 0.5F, -0.25F});
    auto value = makeF16Sequence(
        bn_, BHDS, 4, 2,
        {1.5F, -2.25F, 100.0F, 200.0F,
         300.0F, 400.0F, 4.5F, 6.25F});
    auto output = std::make_shared<Tensor>(bn_);

    CPUPatternSparseAttentionFunc op(
        bn_, "pattern-f16-key-value", 2, 0.5F, false, 0.5F);
    runPatternAttention(op, query, key, value, output);

    EXPECT_EQ(key->dtype(), MLLM_TYPE_F16);
    EXPECT_EQ(value->dtype(), MLLM_TYPE_F16);
    EXPECT_EQ(value->ctype(), BHDS);
    EXPECT_EQ(output->dtype(), MLLM_TYPE_F32);
    EXPECT_EQ(output->ctype(), BSHD);
    EXPECT_NEAR(output->dataAt<float>(0, 0, 0, 0), 3.0F, 1.0e-3F);
    EXPECT_NEAR(output->dataAt<float>(0, 0, 0, 1), 2.0F, 1.0e-3F);
}

TEST_F(CPUTest, CPUPatternSparseAttentionFusedAddressAndMaxMatchLegacy) {
    constexpr int query_len = 2;
    constexpr int key_len = 6;
    constexpr int head_dim = 8;
    const std::vector<float> query_data = {
        0.25F, -0.50F, 0.75F, 1.00F, -1.25F, 1.50F, -1.75F, 2.00F,
        -0.10F, 0.20F, -0.30F, 0.40F, -0.50F, 0.60F, -0.70F, 0.80F,
    };
    std::vector<float> key_data(key_len * head_dim);
    std::vector<float> value_data(key_len * head_dim);
    for (int s = 0; s < key_len; ++s) {
        for (int d = 0; d < head_dim; ++d) {
            key_data[s * head_dim + d] =
                static_cast<float>((s + 1) * (d - 3)) / 17.0F;
            value_data[s * head_dim + d] =
                static_cast<float>((s - 2) * (d + 1)) / 11.0F;
        }
    }
    auto query = makeF32Query(bn_, query_len, head_dim, query_data);
    auto key = makeF16Sequence(
        bn_, BSHD, key_len, head_dim, key_data);
    auto value = makeF16Sequence(
        bn_, BHDS, key_len, head_dim, value_data);
    auto legacy_output = std::make_shared<Tensor>(bn_);
    auto fused_output = std::make_shared<Tensor>(bn_);
    ScopedEnvironmentOverride qk_max(
        "MLLM_SPARSE_FUSE_QK_MAX", "0");
    ScopedEnvironmentOverride index_offset(
        "MLLM_SPARSE_FUSE_INDEX_OFFSET", "0");

    CPUPatternSparseAttentionFunc legacy(
        bn_, "pattern-legacy-address-max", 1, 0.5F, false, 0.5F);
    runPatternAttention(legacy, query, key, value, legacy_output);

    qk_max.set("1");
    index_offset.set("1");
    CPUPatternSparseAttentionFunc fused(
        bn_, "pattern-fused-address-max", 1, 0.5F, false, 0.5F);
    runPatternAttention(fused, query, key, value, fused_output);

    for (int s = 0; s < query_len; ++s) {
        for (int d = 0; d < head_dim; ++d) {
            EXPECT_FLOAT_EQ(
                fused_output->dataAt<float>(0, 0, s, d),
                legacy_output->dataAt<float>(0, 0, s, d));
        }
    }
}

TEST_F(CPUTest, CPUPatternSparseAttentionAppliesInverseSqrtHeadDimScale) {
    constexpr int head_dim = 2;
    auto query = makeF32Query(bn_, 1, head_dim, {1.0F, 0.0F});
    // The fixed pattern retains keys {0,3}. Their unscaled QK scores are 0
    // and 2, while keys 1 and 2 deliberately have much larger scores.
    auto key = makeF16Sequence(
        bn_, BSHD, 4, head_dim,
        {0.0F, 0.0F, 100.0F, 0.0F,
         200.0F, 0.0F, 2.0F, 0.0F});
    auto value = makeF16Sequence(
        bn_, BHDS, 4, head_dim,
        {0.0F, 0.0F, 100.0F, 200.0F,
         300.0F, 400.0F, 10.0F, 20.0F});
    auto output = std::make_shared<Tensor>(bn_);

    CPUPatternSparseAttentionFunc op(
        bn_, "pattern-qk-scale", 2, 0.5F, false, 0.5F);
    runPatternAttention(op, query, key, value, output);

    const float scaled_score = 2.0F / std::sqrt(2.0F);
    const float key3_weight =
        std::exp(scaled_score) / (1.0F + std::exp(scaled_score));
    EXPECT_NEAR(output->dataAt<float>(0, 0, 0, 0),
                key3_weight * 10.0F, 2.0e-2F);
    EXPECT_NEAR(output->dataAt<float>(0, 0, 0, 1),
                key3_weight * 20.0F, 2.0e-2F);
}

TEST_F(CPUTest, CPUPatternSparseAttentionReadsTwoChunkKVCacheOldPrefix) {
    CPUKVCacheNPU key_cache(bn_, "test.k_cache", 1, 8, 2);
    CPUKVCacheNPU value_cache(bn_, "test.v_cache", 1, 8, 2);
    CPUPatternSparseAttentionFunc op(
        bn_, "pattern-cached-kv", 2, 0.5F, true, 1.0F);
    auto cached_key = std::make_shared<Tensor>(bn_);
    auto cached_value = std::make_shared<Tensor>(bn_);
    auto output = std::make_shared<Tensor>(bn_);

    auto run_chunk = [&](const std::vector<float> &keys,
                         const std::vector<float> &values,
                         int expected_key_len) {
        auto query = makeF32Query(bn_, 2, 1, {1.0F, 1.0F});
        auto key_input = makeF16Sequence(bn_, BSHD, 2, 1, keys);
        auto value_input = makeF16Sequence(bn_, BHDS, 2, 1, values);

        ASSERT_EQ(key_cache.reshape({key_input}, {cached_key}), MLLM_NO_ERROR);
        ASSERT_EQ(value_cache.reshape({value_input}, {cached_value}),
                  MLLM_NO_ERROR);
        ASSERT_EQ(cached_key->sequence(), expected_key_len);
        ASSERT_EQ(cached_value->sequence(), expected_key_len);
        ASSERT_EQ(op.reshape({query, cached_key, cached_value}, {output}),
                  MLLM_NO_ERROR);

        ASSERT_EQ(key_cache.setUp({key_input}, {cached_key}), MLLM_NO_ERROR);
        ASSERT_EQ(value_cache.setUp({value_input}, {cached_value}),
                  MLLM_NO_ERROR);
        ASSERT_NE(cached_key->masterTensor(), nullptr);
        ASSERT_NE(cached_value->masterTensor(), nullptr);
        ASSERT_EQ(cached_key->masterTensor()->ctype(), BSHD);
        ASSERT_EQ(cached_value->masterTensor()->ctype(), BHDS);
        ASSERT_EQ(op.setUp({query, cached_key, cached_value}, {output}),
                  MLLM_NO_ERROR);

        ASSERT_EQ(key_cache.execute({key_input}, {cached_key}), MLLM_NO_ERROR);
        ASSERT_EQ(value_cache.execute({value_input}, {cached_value}),
                  MLLM_NO_ERROR);
        ASSERT_EQ(op.execute({query, cached_key, cached_value}, {output}),
                  MLLM_NO_ERROR);
    };

    // First chunk has no old prefix. With density 0.5 and a fully local
    // pattern, its two causal rows retain keys 0 and 1 respectively.
    run_chunk({0.0F, 1.0F}, {10.0F, 20.0F}, 2);
    EXPECT_FLOAT_EQ(output->dataAt<float>(0, 0, 0, 0), 10.0F);
    EXPECT_FLOAT_EQ(output->dataAt<float>(0, 0, 1, 0), 20.0F);

    // Second chunk sees the two cached tokens as an old prefix. Row 0 has
    // valid=3 and retains local keys {1,2}; row 1 retains {2,3}.
    CPUSparseSelectionStats::reset();
    run_chunk({2.0F, 3.0F}, {30.0F, 40.0F}, 4);
    const auto stats = CPUSparseSelectionStats::snapshot();
    const std::vector<float> all_keys = {0.0F, 1.0F, 2.0F, 3.0F};
    const std::vector<float> all_values = {10.0F, 20.0F, 30.0F, 40.0F};
    EXPECT_NEAR(output->dataAt<float>(0, 0, 0, 0),
                selectedSoftmaxValue(all_keys, all_values, {1, 2}), 3.0e-2F);
    EXPECT_NEAR(output->dataAt<float>(0, 0, 1, 0),
                selectedSoftmaxValue(all_keys, all_values, {2, 3}), 3.0e-2F);
    EXPECT_EQ(stats.eligible, 7U);
    EXPECT_EQ(stats.retained, 4U);
}

TEST_F(CPUTest, CPUPatternSparseAttentionKeepOneUsesUniformKeyZero) {
    auto query = makeF32Query(bn_, 1, 1, {1.0F});
    auto key = makeF16Sequence(
        bn_, BSHD, 5, 1, {0.0F, 10.0F, 20.0F, 30.0F, 40.0F});
    auto value = makeF16Sequence(
        bn_, BHDS, 5, 1, {7.0F, 20.0F, 30.0F, 40.0F, 50.0F});
    auto output = std::make_shared<Tensor>(bn_);

    // keep=ceil(0.1*5)=1. local_ratio=0 makes this the R==1 uniform-history
    // case, whose specified endpoint rule chooses key 0.
    CPUPatternSparseAttentionFunc op(
        bn_, "pattern-keep-one", 1, 0.9F, false, 0.0F);
    CPUSparseSelectionStats::reset();
    runPatternAttention(op, query, key, value, output);
    const auto stats = CPUSparseSelectionStats::snapshot();

    EXPECT_FLOAT_EQ(output->dataAt<float>(0, 0, 0, 0), 7.0F);
    EXPECT_EQ(stats.eligible, 5U);
    EXPECT_EQ(stats.retained, 1U);
}

TEST_F(CPUTest, CPUPatternSparseAttentionRandomRowsAreExactUniformAndRepeatable) {
    constexpr int valid = 100;
    constexpr int keep = 20;
    std::vector<int32_t> first(keep);
    std::vector<int32_t> repeated(keep);
    std::vector<int32_t> different(keep);
    std::vector<int32_t> different_query(keep);
    std::vector<int32_t> different_head(keep);
    CPUPatternSparseAttentionFunc::buildRandomPatternIndices(
        valid, keep,
        CPUPatternSparseAttentionFunc::makeRandomRowSeed(1234, 17, 0, 2),
        first.data());
    CPUPatternSparseAttentionFunc::buildRandomPatternIndices(
        valid, keep,
        CPUPatternSparseAttentionFunc::makeRandomRowSeed(1234, 17, 0, 2),
        repeated.data());
    CPUPatternSparseAttentionFunc::buildRandomPatternIndices(
        valid, keep,
        CPUPatternSparseAttentionFunc::makeRandomRowSeed(5678, 17, 0, 2),
        different.data());
    CPUPatternSparseAttentionFunc::buildRandomPatternIndices(
        valid, keep,
        CPUPatternSparseAttentionFunc::makeRandomRowSeed(1234, 18, 0, 2),
        different_query.data());
    CPUPatternSparseAttentionFunc::buildRandomPatternIndices(
        valid, keep,
        CPUPatternSparseAttentionFunc::makeRandomRowSeed(1234, 17, 0, 3),
        different_head.data());

    EXPECT_EQ(first, repeated);
    EXPECT_NE(first, different);
    EXPECT_NE(first, different_query);
    EXPECT_NE(first, different_head);
    for (int i = 0; i < keep; ++i) {
        EXPECT_GE(first[i], 0);
        EXPECT_LT(first[i], valid);
        if (i > 0) EXPECT_LT(first[i - 1], first[i]);
    }

    // Across many independently seeded rows, every key in the whole valid
    // interval has the same 20% marginal selection probability.  This also
    // detects the old one-key-per-five-token-stratum policy.
    constexpr int trials = 4096;
    std::vector<int> selected_count(valid, 0);
    bool observed_non_stratified_row = false;
    for (int trial = 0; trial < trials; ++trial) {
        std::vector<int32_t> selected(keep);
        CPUPatternSparseAttentionFunc::buildRandomPatternIndices(
            valid, keep,
            CPUPatternSparseAttentionFunc::makeRandomRowSeed(
                static_cast<uint32_t>(trial), 17, 0, 2),
            selected.data());
        std::array<int, valid / 5> per_old_stratum{};
        for (const int index : selected) {
            ++selected_count[index];
            ++per_old_stratum[index / 5];
        }
        observed_non_stratified_row |= std::any_of(
            per_old_stratum.begin(), per_old_stratum.end(),
            [](int count) { return count != 1; });
    }
    EXPECT_TRUE(observed_non_stratified_row);
    for (const int count : selected_count) {
        EXPECT_GT(count, 700);
        EXPECT_LT(count, 940);
    }

    std::vector<int32_t> keep_all(7);
    CPUPatternSparseAttentionFunc::buildRandomPatternIndices(
        7, 7, CPUPatternSparseAttentionFunc::makeRandomRowSeed(1234, 6),
        keep_all.data());
    EXPECT_EQ(keep_all,
              (std::vector<int32_t>{0, 1, 2, 3, 4, 5, 6}));
}

TEST_F(CPUTest, CPUPatternSparseAttentionExecutesRandomSelectedKeysOnly) {
    constexpr int valid = 10;
    constexpr int keep = 2;
    constexpr uint32_t seed = 37;
    std::vector<int32_t> selected(keep);
    const uint64_t row_seed =
        CPUPatternSparseAttentionFunc::makeRandomRowSeed(seed, valid - 1);
    CPUPatternSparseAttentionFunc::buildRandomPatternIndices(
        valid, keep, row_seed, selected.data());

    auto query = makeF32Query(bn_, 1, 1, {1.0F});
    auto key = makeF16Sequence(
        bn_, BSHD, valid, 1, std::vector<float>(valid, 0.0F));
    std::vector<float> value_data(valid);
    for (int i = 0; i < valid; ++i) value_data[i] = i + 1.0F;
    auto value = makeF16Sequence(bn_, BHDS, valid, 1, value_data);
    auto output = std::make_shared<Tensor>(bn_);

    CPUPatternSparseAttentionFunc op(
        bn_, "pattern-random", 2, 0.8F, false, 0.5F, 0, 0, true, seed);
    CPUSparseSelectionStats::reset();
    runPatternAttention(op, query, key, value, output);
    const auto stats = CPUSparseSelectionStats::snapshot();

    const float expected =
        (value_data[selected[0]] + value_data[selected[1]]) / 2.0F;
    EXPECT_NEAR(output->dataAt<float>(0, 0, 0, 0), expected, 1.0e-3F);
    EXPECT_EQ(stats.eligible, static_cast<uint64_t>(valid));
    EXPECT_EQ(stats.retained, static_cast<uint64_t>(keep));
}

TEST_F(CPUTest, CPUPatternSparseAttentionRandomPatternHonorsDenseTokens) {
    constexpr int valid = 4;
    auto query = makeF32Query(bn_, 1, 1, {1.0F});
    auto key = makeF16Sequence(
        bn_, BSHD, valid, 1, std::vector<float>(valid, 0.0F));
    auto value = makeF16Sequence(
        bn_, BHDS, valid, 1, {10.0F, 20.0F, 30.0F, 40.0F});
    auto output = std::make_shared<Tensor>(bn_);

    CPUPatternSparseAttentionFunc op(
        bn_, "pattern-random-dense-prefix", 2, 0.8F, false,
        0.5F, 0, valid, true, 43);
    CPUSparseSelectionStats::reset();
    runPatternAttention(op, query, key, value, output);
    const auto stats = CPUSparseSelectionStats::snapshot();

    EXPECT_NEAR(output->dataAt<float>(0, 0, 0, 0), 25.0F, 1.0e-3F);
    EXPECT_EQ(stats.eligible, static_cast<uint64_t>(valid));
    EXPECT_EQ(stats.retained, static_cast<uint64_t>(valid));
}

TEST_F(CPUTest, CPUPatternSparseAttentionRandomPatternRespectsCausalOldPrefix) {
    constexpr uint32_t seed = 91;
    auto query = makeF32Query(bn_, 2, 1, {1.0F, 1.0F});
    const std::vector<float> key_data = {0.0F, 1.0F, 2.0F, 100.0F};
    const std::vector<float> value_data = {10.0F, 20.0F, 30.0F, 1000.0F};
    auto key = makeF16Sequence(bn_, BSHD, 4, 1, key_data);
    auto value = makeF16Sequence(bn_, BHDS, 4, 1, value_data);
    auto output = std::make_shared<Tensor>(bn_);

    CPUPatternSparseAttentionFunc op(
        bn_, "pattern-random-causal", 2, 0.5F, true,
        0.5F, 0, 0, true, seed);
    CPUSparseSelectionStats::reset();
    runPatternAttention(op, query, key, value, output);
    const auto stats = CPUSparseSelectionStats::snapshot();

    for (int s = 0; s < 2; ++s) {
        const int valid = 3 + s;
        const int keep = (valid + 1) / 2;
        std::vector<int32_t> selected(keep);
        CPUPatternSparseAttentionFunc::buildRandomPatternIndices(
            valid, keep,
            CPUPatternSparseAttentionFunc::makeRandomRowSeed(seed, 2 + s),
            selected.data());
        float maximum = -INFINITY;
        for (const int key_index : selected) {
            maximum = std::max(maximum, key_data[key_index]);
        }
        float numerator = 0.0F;
        float denominator = 0.0F;
        for (const int key_index : selected) {
            const float weight = std::exp(key_data[key_index] - maximum);
            numerator += weight * value_data[key_index];
            denominator += weight;
        }
        EXPECT_NEAR(output->dataAt<float>(0, 0, s, 0),
                    numerator / denominator, 3.0e-2F);
    }
    EXPECT_EQ(stats.eligible, 7U);
    EXPECT_EQ(stats.retained, 4U);
}

TEST_F(CPUTest, CPUPatternSparseAttentionPacksMultiHeadBSHDKeys) {
    constexpr int heads = 2;
    constexpr int valid = 10;
    constexpr int keep = 2;
    constexpr uint32_t seed = 53;
    auto query = std::make_shared<Tensor>(bn_);
    query->setCtype(BSHD);
    query->setDtype(MLLM_TYPE_F32);
    query->reshape(1, heads, 1, 1);
    query->alloc();
    auto key = std::make_shared<Tensor>(bn_);
    key->setCtype(BSHD);
    key->setDtype(MLLM_TYPE_F16);
    key->reshape(1, heads, valid, 1);
    key->alloc();
    auto value = std::make_shared<Tensor>(bn_);
    value->setCtype(BHDS);
    value->setDtype(MLLM_TYPE_F16);
    value->reshape(1, heads, valid, 1);
    value->alloc();
    std::vector<std::vector<float>> key_data(
        heads, std::vector<float>(valid));
    std::vector<std::vector<float>> value_data(
        heads, std::vector<float>(valid));
    for (int h = 0; h < heads; ++h) {
        query->setDataAt<float>(0, h, 0, 0, 1.0F);
        for (int s = 0; s < valid; ++s) {
            key_data[h][s] = h == 0 ? 0.125F * s : -0.0625F * s;
            value_data[h][s] = h * 100.0F + s + 1.0F;
            key->setDataAt<mllm_fp16_t>(
                0, h, s, 0, MLLM_FP32_TO_FP16(key_data[h][s]));
            value->setDataAt<mllm_fp16_t>(
                0, h, s, 0, MLLM_FP32_TO_FP16(value_data[h][s]));
        }
    }
    ASSERT_NE(key->sequenceSkipDim(), 1);

    auto output = std::make_shared<Tensor>(bn_);
    CPUPatternSparseAttentionFunc op(
        bn_, "pattern-random-packed-key", 2, 0.8F, false,
        0.5F, 0, 0, true, seed);
    runPatternAttention(op, query, key, value, output);

    for (int h = 0; h < heads; ++h) {
        std::vector<int32_t> selected(keep);
        CPUPatternSparseAttentionFunc::buildRandomPatternIndices(
            valid, keep,
            CPUPatternSparseAttentionFunc::makeRandomRowSeed(
                seed, valid - 1, 0, h),
            selected.data());
        const float maximum = std::max(
            key_data[h][selected[0]], key_data[h][selected[1]]);
        float numerator = 0.0F;
        float denominator = 0.0F;
        for (const int key_index : selected) {
            const float weight =
                std::exp(key_data[h][key_index] - maximum);
            numerator += weight * value_data[h][key_index];
            denominator += weight;
        }
        const float expected = numerator / denominator;
        EXPECT_NEAR(output->dataAt<float>(0, h, 0, 0), expected, 5.0e-2F);
    }
}

TEST_F(CPUTest, CPUPatternSparseAttentionUsesIndependentHeadRetentions) {
    constexpr int heads = 2;
    constexpr int valid = 10;
    auto query = std::make_shared<Tensor>(bn_);
    query->setCtype(BSHD);
    query->setDtype(MLLM_TYPE_F32);
    query->reshape(1, heads, 1, 1);
    query->alloc();
    auto key = std::make_shared<Tensor>(bn_);
    key->setCtype(BSHD);
    key->setDtype(MLLM_TYPE_F16);
    key->reshape(1, heads, valid, 1);
    key->alloc();
    auto value = std::make_shared<Tensor>(bn_);
    value->setCtype(BHDS);
    value->setDtype(MLLM_TYPE_F16);
    value->reshape(1, heads, valid, 1);
    value->alloc();
    for (int h = 0; h < heads; ++h) {
        query->setDataAt<float>(0, h, 0, 0, 0.0F);
        for (int s = 0; s < valid; ++s) {
            key->setDataAt<mllm_fp16_t>(
                0, h, s, 0, MLLM_FP32_TO_FP16(0.0F));
            value->setDataAt<mllm_fp16_t>(
                0, h, s, 0, MLLM_FP32_TO_FP16(
                    h * 100.0F + s + 1.0F));
        }
    }
    auto output = std::make_shared<Tensor>(bn_);
    CPUPatternSparseAttentionFunc op(
        bn_, "pattern-per-head-retention", 2, 0.8F, false,
        1.0F, 0, 0, false, 1, 0, false, {0.2F, 0.6F});
    CPUSparseSelectionStats::reset();
    runPatternAttention(op, query, key, value, output);
    const auto stats = CPUSparseSelectionStats::snapshot();

    // With a fully local pattern, head 0 keeps values 9,10 and head 1 keeps
    // values 105..110. Equal QK scores reduce each result to their mean.
    EXPECT_NEAR(output->dataAt<float>(0, 0, 0, 0), 9.5F, 1.0e-3F);
    EXPECT_NEAR(output->dataAt<float>(0, 1, 0, 0), 107.5F, 5.0e-2F);
    EXPECT_EQ(stats.eligible, 20U);
    EXPECT_EQ(stats.retained, 8U);
}

TEST_F(CPUTest, CPUPatternSparseAttentionPackedKeyAppendsAndResetsCache) {
    constexpr int heads = 2;
    constexpr int chunk = 4;
    constexpr uint32_t seed = 67;
    CPUKVCacheNPU key_cache(bn_, "packed.test.k_cache", 1, 12, 2);
    CPUKVCacheNPU value_cache(bn_, "packed.test.v_cache", 1, 12, 2);
    CPUPatternSparseAttentionFunc op(
        bn_, "pattern-random-packed-cache", 2, 0.8F, true,
        0.5F, 0, 0, true, seed, 12);
    auto cached_key = std::make_shared<Tensor>(bn_);
    auto cached_value = std::make_shared<Tensor>(bn_);
    auto output = std::make_shared<Tensor>(bn_);

    auto make_chunk = [&](ChlType ctype,
                          const std::vector<std::vector<float>> &data) {
        auto tensor = std::make_shared<Tensor>(bn_);
        tensor->setCtype(ctype);
        tensor->setDtype(MLLM_TYPE_F16);
        tensor->reshape(1, heads, chunk, 1);
        tensor->alloc();
        for (int h = 0; h < heads; ++h) {
            for (int s = 0; s < chunk; ++s) {
                tensor->setDataAt<mllm_fp16_t>(
                    0, h, s, 0, MLLM_FP32_TO_FP16(data[h][s]));
            }
        }
        return tensor;
    };
    auto run_chunk = [&](const std::vector<std::vector<float>> &keys,
                         const std::vector<std::vector<float>> &values) {
        auto query = std::make_shared<Tensor>(bn_);
        query->setCtype(BSHD);
        query->setDtype(MLLM_TYPE_F32);
        query->reshape(1, heads, chunk, 1);
        query->alloc();
        for (int h = 0; h < heads; ++h) {
            for (int s = 0; s < chunk; ++s) {
                query->setDataAt<float>(0, h, s, 0, 1.0F);
            }
        }
        auto key_input = make_chunk(BSHD, keys);
        auto value_input = make_chunk(BHDS, values);
        ASSERT_EQ(key_cache.reshape({key_input}, {cached_key}), MLLM_NO_ERROR);
        ASSERT_EQ(value_cache.reshape({value_input}, {cached_value}),
                  MLLM_NO_ERROR);
        ASSERT_EQ(op.reshape({query, cached_key, cached_value}, {output}),
                  MLLM_NO_ERROR);
        ASSERT_EQ(key_cache.setUp({key_input}, {cached_key}), MLLM_NO_ERROR);
        ASSERT_EQ(value_cache.setUp({value_input}, {cached_value}),
                  MLLM_NO_ERROR);
        ASSERT_EQ(op.setUp({query, cached_key, cached_value}, {output}),
                  MLLM_NO_ERROR);
        ASSERT_EQ(key_cache.execute({key_input}, {cached_key}), MLLM_NO_ERROR);
        ASSERT_EQ(value_cache.execute({value_input}, {cached_value}),
                  MLLM_NO_ERROR);
        ASSERT_EQ(op.execute({query, cached_key, cached_value}, {output}),
                  MLLM_NO_ERROR);
    };
    auto check_output = [&](const std::vector<std::vector<float>> &all_keys,
                            const std::vector<std::vector<float>> &all_values,
                            int old_prefix) {
        for (int h = 0; h < heads; ++h) {
            for (int s = 0; s < chunk; ++s) {
                const int valid = old_prefix + s + 1;
                const int keep = (valid + 4) / 5;
                std::vector<int32_t> selected(keep);
                CPUPatternSparseAttentionFunc::buildRandomPatternIndices(
                    valid, keep,
                    CPUPatternSparseAttentionFunc::makeRandomRowSeed(
                        seed, old_prefix + s, 0, h),
                    selected.data());
                float maximum = -INFINITY;
                for (const int key_index : selected) {
                    maximum = std::max(maximum, all_keys[h][key_index]);
                }
                float numerator = 0.0F;
                float denominator = 0.0F;
                for (const int key_index : selected) {
                    const float weight =
                        std::exp(all_keys[h][key_index] - maximum);
                    numerator += weight * all_values[h][key_index];
                    denominator += weight;
                }
                const float expected = numerator / denominator;
                EXPECT_NEAR(output->dataAt<float>(0, h, s, 0), expected,
                            1.0e-3F * std::max(1.0F, std::fabs(expected)));
            }
        }
    };

    const std::vector<std::vector<float>> keys0 = {
        {1.5F, 0.2F, 0.4F, 0.6F},
        {-1.5F, -0.1F, -0.2F, -0.3F},
    };
    const std::vector<std::vector<float>> values0 = {
        {10.0F, 20.0F, 30.0F, 40.0F},
        {110.0F, 120.0F, 130.0F, 140.0F},
    };
    const std::vector<std::vector<float>> keys1 = {
        {0.8F, 1.0F, 1.2F, 1.4F},
        {-0.4F, -0.5F, -0.6F, -0.7F},
    };
    const std::vector<std::vector<float>> values1 = {
        {50.0F, 60.0F, 70.0F, 80.0F},
        {150.0F, 160.0F, 170.0F, 180.0F},
    };
    run_chunk(keys0, values0);
    EXPECT_EQ(op.lastKeyPackBeginForTest(), 0);
    EXPECT_EQ(op.lastKeyPackEndForTest(), chunk);
    run_chunk(keys1, values1);
    EXPECT_EQ(op.lastKeyPackBeginForTest(), chunk);
    EXPECT_EQ(op.lastKeyPackEndForTest(), 2 * chunk);
    std::vector<std::vector<float>> all_keys = keys0;
    std::vector<std::vector<float>> all_values = values0;
    for (int h = 0; h < heads; ++h) {
        all_keys[h].insert(all_keys[h].end(), keys1[h].begin(), keys1[h].end());
        all_values[h].insert(
            all_values[h].end(), values1[h].begin(), values1[h].end());
    }
    check_output(all_keys, all_values, chunk);

    key_cache.clearCache();
    value_cache.clearCache();
    const std::vector<std::vector<float>> reset_keys = {
        {2.0F, 2.2F, 2.4F, 2.6F},
        {-2.0F, -2.1F, -2.2F, -2.3F},
    };
    const std::vector<std::vector<float>> reset_values = {
        {210.0F, 220.0F, 230.0F, 240.0F},
        {310.0F, 320.0F, 330.0F, 340.0F},
    };
    run_chunk(reset_keys, reset_values);
    EXPECT_EQ(op.lastKeyPackBeginForTest(), 0);
    EXPECT_EQ(op.lastKeyPackEndForTest(), chunk);
    const std::vector<std::vector<float>> reset_keys1 = {
        {2.8F, 3.0F, 3.2F, 3.4F},
        {-2.4F, -2.5F, -2.6F, -2.7F},
    };
    const std::vector<std::vector<float>> reset_values1 = {
        {250.0F, 260.0F, 270.0F, 280.0F},
        {350.0F, 360.0F, 370.0F, 380.0F},
    };
    run_chunk(reset_keys1, reset_values1);
    EXPECT_EQ(op.lastKeyPackBeginForTest(), chunk);
    EXPECT_EQ(op.lastKeyPackEndForTest(), 2 * chunk);
    std::vector<std::vector<float>> reset_all_keys = reset_keys;
    std::vector<std::vector<float>> reset_all_values = reset_values;
    for (int h = 0; h < heads; ++h) {
        reset_all_keys[h].insert(reset_all_keys[h].end(),
                                 reset_keys1[h].begin(), reset_keys1[h].end());
        reset_all_values[h].insert(reset_all_values[h].end(),
                                   reset_values1[h].begin(),
                                   reset_values1[h].end());
    }
    check_output(reset_all_keys, reset_all_values, chunk);
}
