#include "CPUTest.hpp"
#include "backends/cpu/op/CPUKVCacheNPU.hpp"
#include "backends/cpu/op/CPUSoftMax.hpp"
#include "backends/cpu/op/CPUSparseSoftmaxValueFunc.hpp"
#include "backends/cpu/compute/Matmul.hpp"

#include <cmath>

TEST_F(CPUTest, CPUSparseSoftmaxValueUsesCausalRowWiseTopK) {
    auto logits = std::make_shared<Tensor>(bn_);
    logits->setCtype(BHSD);
    logits->setDtype(MLLM_TYPE_F32);
    logits->reshape(1, 1, 2, 3);
    logits->alloc();
    const float scores[2][3] = {{0.0F, 2.0F, 100.0F},
                                {3.0F, 1.0F, 2.0F}};
    for (int s = 0; s < 2; ++s) {
        for (int d = 0; d < 3; ++d) {
            logits->setDataAt<float>(0, 0, s, d, scores[s][d]);
        }
    }

    auto value = std::make_shared<Tensor>(bn_);
    value->setCtype(BHDS);
    value->setDtype(MLLM_TYPE_F32);
    value->reshape(1, 1, 3, 2);
    value->alloc();
    const float values[3][2] = {{1.0F, 10.0F},
                                {2.0F, 20.0F},
                                {3.0F, 30.0F}};
    for (int s = 0; s < 3; ++s) {
        for (int d = 0; d < 2; ++d) {
            value->setDataAt<float>(0, 0, s, d, values[s][d]);
        }
    }

    auto output = std::make_shared<Tensor>(bn_);
    CPUSparseSoftmaxValueFunc op(bn_, "sparse-attention", 2, 0.5F, true);
    ASSERT_EQ(op.reshape({logits, value}, {output}), MLLM_NO_ERROR);
    ASSERT_EQ(op.setUp({logits, value}, {output}), MLLM_NO_ERROR);
    ASSERT_EQ(op.execute({logits, value}, {output}), MLLM_NO_ERROR);

    // Row 0 sees only keys 0..1 and retains key 1. Row 1 sees all keys and
    // retains keys 0 and 2, with a stable softmax over scores 3 and 2.
    const float key0_weight = std::exp(3.0F) / (std::exp(3.0F) + std::exp(2.0F));
    const float expected_row1 = key0_weight * 1.0F + (1.0F - key0_weight) * 3.0F;
    EXPECT_NEAR(output->dataAt<float>(0, 0, 0, 0), 2.0F, 1.0e-6F);
    EXPECT_NEAR(output->dataAt<float>(0, 0, 0, 1), 20.0F, 1.0e-5F);
    EXPECT_NEAR(output->dataAt<float>(0, 0, 1, 0), expected_row1, 1.0e-5F);
    EXPECT_NEAR(output->dataAt<float>(0, 0, 1, 1), expected_row1 * 10.0F, 1.0e-4F);
}

TEST_F(CPUTest, CPUSparseSoftmaxValueUsesPerHeadRetention) {
    auto logits = std::make_shared<Tensor>(bn_);
    logits->setCtype(BHSD);
    logits->setDtype(MLLM_TYPE_F32);
    logits->reshape(1, 2, 1, 4);
    logits->alloc();
    for (int head = 0; head < 2; ++head) {
        for (int key = 0; key < 4; ++key) {
            logits->setDataAt<float>(
                0, head, 0, key, static_cast<float>(key));
        }
    }

    auto value = std::make_shared<Tensor>(bn_);
    value->setCtype(BHDS);
    value->setDtype(MLLM_TYPE_F32);
    value->reshape(1, 2, 4, 1);
    value->alloc();
    for (int head = 0; head < 2; ++head) {
        for (int key = 0; key < 4; ++key) {
            value->setDataAt<float>(
                0, head, key, 0, static_cast<float>(key * 10));
        }
    }

    auto output = std::make_shared<Tensor>(bn_);
    CPUSparseSoftmaxValueFunc op(
        bn_, "sparse-attention-per-head", 2, 0.5F, false, 0,
        {0.25F, 0.5F});
    ASSERT_EQ(op.reshape({logits, value}, {output}), MLLM_NO_ERROR);
    ASSERT_EQ(op.setUp({logits, value}, {output}), MLLM_NO_ERROR);
    ASSERT_EQ(op.execute({logits, value}, {output}), MLLM_NO_ERROR);

    EXPECT_FLOAT_EQ(output->dataAt<float>(0, 0, 0, 0), 30.0F);
    const float weight2 = std::exp(2.0F) / (std::exp(2.0F) + std::exp(3.0F));
    EXPECT_NEAR(output->dataAt<float>(0, 1, 0, 0),
                weight2 * 20.0F + (1.0F - weight2) * 30.0F, 1.0e-5F);
}

TEST_F(CPUTest, CPUSparseSoftmaxValueF32CausalKeepAllUsesValidKey) {
    auto logits = std::make_shared<Tensor>(bn_);
    logits->setCtype(BHSD);
    logits->setDtype(MLLM_TYPE_F32);
    logits->reshape(1, 2, 2, 2);
    logits->alloc();
    const float scores[2][2][2] = {
        {{1.0F, 100.0F}, {0.0F, 5.0F}},
        {{2.0F, 100.0F}, {4.0F, 3.0F}},
    };
    for (int h = 0; h < 2; ++h) {
        for (int s = 0; s < 2; ++s) {
            for (int key = 0; key < 2; ++key) {
                logits->setDataAt<float>(0, h, s, key, scores[h][s][key]);
            }
        }
    }

    auto value = std::make_shared<Tensor>(bn_);
    value->setCtype(BHDS);
    value->setDtype(MLLM_TYPE_F32);
    value->reshape(1, 2, 2, 1);
    value->alloc();
    value->setDataAt<float>(0, 0, 0, 0, 10.0F);
    value->setDataAt<float>(0, 0, 1, 0, 20.0F);
    value->setDataAt<float>(0, 1, 0, 0, 100.0F);
    value->setDataAt<float>(0, 1, 1, 0, 200.0F);

    auto output = std::make_shared<Tensor>(bn_);
    // One thread deliberately reuses the same scratch across heads. Head 0's
    // final row selects key 1; head 1's first causal row must nevertheless use
    // its sole valid key 0 when keep == valid == 1.
    CPUSparseSoftmaxValueFunc op(
        bn_, "sparse-attention-f32-keep-all", 1, 0.5F, true);
    ASSERT_EQ(op.reshape({logits, value}, {output}), MLLM_NO_ERROR);
    ASSERT_EQ(op.setUp({logits, value}, {output}), MLLM_NO_ERROR);
    ASSERT_EQ(op.execute({logits, value}, {output}), MLLM_NO_ERROR);

    EXPECT_FLOAT_EQ(output->dataAt<float>(0, 0, 0, 0), 10.0F);
    EXPECT_FLOAT_EQ(output->dataAt<float>(0, 0, 1, 0), 20.0F);
    EXPECT_FLOAT_EQ(output->dataAt<float>(0, 1, 0, 0), 100.0F);
}

TEST_F(CPUTest, CPUSparseSoftmaxValueReadsF16BHDSCacheWithOldPrefix) {
    auto logits = std::make_shared<Tensor>(bn_);
    logits->setCtype(BHSD);
    logits->setDtype(MLLM_TYPE_F32);
    logits->reshape(1, 1, 2, 4);
    logits->alloc();
    const float scores[2][4] = {{0.0F, 1.0F, 3.0F, 100.0F},
                                {5.0F, 4.0F, 3.0F, 2.0F}};
    for (int s = 0; s < 2; ++s) {
        for (int d = 0; d < 4; ++d) {
            logits->setDataAt<float>(0, 0, s, d, scores[s][d]);
        }
    }

    auto value = std::make_shared<Tensor>(bn_);
    value->setCtype(BHDS);
    value->setDtype(MLLM_TYPE_F16);
    value->reshape(1, 1, 4, 2);
    value->alloc();
    for (int s = 0; s < 4; ++s) {
        value->setDataAt<mllm_fp16_t>(0, 0, s, 0,
                                      MLLM_FP32_TO_FP16(10.0F + s));
        value->setDataAt<mllm_fp16_t>(0, 0, s, 1,
                                      MLLM_FP32_TO_FP16(20.0F + s));
    }

    auto output = std::make_shared<Tensor>(bn_);
    CPUSparseSoftmaxValueFunc op(bn_, "sparse-attention-f16", 2, 0.75F, true);
    ASSERT_EQ(op.reshape({logits, value}, {output}), MLLM_NO_ERROR);
    ASSERT_EQ(op.setUp({logits, value}, {output}), MLLM_NO_ERROR);
    ASSERT_EQ(op.execute({logits, value}, {output}), MLLM_NO_ERROR);

    // q0 has a two-token old prefix, hence valid keys 0..2 and key 2 wins;
    // q1 sees all four keys and key 0 wins. Future key 3's score must not leak.
    EXPECT_NEAR(output->dataAt<float>(0, 0, 0, 0), 12.0F, 1.0e-6F);
    EXPECT_NEAR(output->dataAt<float>(0, 0, 0, 1), 22.0F, 1.0e-6F);
    EXPECT_NEAR(output->dataAt<float>(0, 0, 1, 0), 10.0F, 1.0e-6F);
    EXPECT_NEAR(output->dataAt<float>(0, 0, 1, 1), 20.0F, 1.0e-6F);
}

TEST_F(CPUTest, CPUSparseSoftmaxValueBreaksCutoffTiesByLowestKeyIndex) {
    auto logits = std::make_shared<Tensor>(bn_);
    logits->setCtype(BHSD);
    logits->setDtype(MLLM_TYPE_F32);
    logits->reshape(1, 1, 1, 4);
    logits->alloc();
    for (int key = 0; key < 4; ++key) {
        logits->setDataAt<float>(0, 0, 0, key, 1.0F);
    }

    auto value = std::make_shared<Tensor>(bn_);
    value->setCtype(BHDS);
    value->setDtype(MLLM_TYPE_F32);
    value->reshape(1, 1, 4, 1);
    value->alloc();
    for (int key = 0; key < 4; ++key) {
        value->setDataAt<float>(0, 0, key, 0,
                                static_cast<float>((key + 1) * 10));
    }

    auto output = std::make_shared<Tensor>(bn_);
    CPUSparseSoftmaxValueFunc op(bn_, "sparse-attention-ties", 2, 0.5F, false);
    ASSERT_EQ(op.reshape({logits, value}, {output}), MLLM_NO_ERROR);
    ASSERT_EQ(op.setUp({logits, value}, {output}), MLLM_NO_ERROR);
    ASSERT_EQ(op.execute({logits, value}, {output}), MLLM_NO_ERROR);

    // Equal scores retain keys 0 and 1, so their equal-weight mean is 15.
    EXPECT_FLOAT_EQ(output->dataAt<float>(0, 0, 0, 0), 15.0F);
}

TEST_F(CPUTest, CPUSparseSoftmaxValuePacksCompleteF16EightByEightBlock) {
    auto logits = std::make_shared<Tensor>(bn_);
    logits->setCtype(BHSD);
    logits->setDtype(MLLM_TYPE_F32);
    logits->reshape(1, 1, 1, 8);
    logits->alloc();
    const float scores[8] = {-4.0F, 4.0F, -3.0F, 3.0F,
                             2.0F, -2.0F, -1.0F, 1.0F};
    for (int key = 0; key < 8; ++key) {
        logits->setDataAt<float>(0, 0, 0, key, scores[key]);
    }

    auto value = std::make_shared<Tensor>(bn_);
    value->setCtype(BHDS);
    value->setDtype(MLLM_TYPE_F16);
    value->reshape(1, 1, 8, 8);
    value->alloc();
    for (int key = 0; key < 8; ++key) {
        for (int d = 0; d < 8; ++d) {
            value->setDataAt<mllm_fp16_t>(
                0, 0, key, d,
                MLLM_FP32_TO_FP16(static_cast<float>(key * 10 + d)));
        }
    }

    auto output = std::make_shared<Tensor>(bn_);
    CPUSparseSoftmaxValueFunc op(bn_, "sparse-attention-pack8", 2, 0.5F, false);
    ASSERT_EQ(op.reshape({logits, value}, {output}), MLLM_NO_ERROR);
    ASSERT_EQ(op.setUp({logits, value}, {output}), MLLM_NO_ERROR);
    ASSERT_EQ(op.execute({logits, value}, {output}), MLLM_NO_ERROR);

    const int selected_keys[4] = {1, 3, 4, 7};
    float denominator = 0.0F;
    for (const int key : selected_keys) denominator += std::exp(scores[key] - 4.0F);
    for (int d = 0; d < 8; ++d) {
        float expected = 0.0F;
        for (const int key : selected_keys) {
            const auto weight_f16 = MLLM_FP32_TO_FP16(
                std::exp(scores[key] - 4.0F) / denominator);
            expected += MLLM_FP16_TO_FP32(weight_f16)
                * static_cast<float>(key * 10 + d);
        }
        EXPECT_NEAR(output->dataAt<float>(0, 0, 0, d), expected, 1.0e-3F);
    }
}

TEST_F(CPUTest, CPUSparseSoftmaxValueSampledPrefilterMatchesDirectExactTopK) {
    auto logits = std::make_shared<Tensor>(bn_);
    logits->setCtype(BHSD);
    logits->setDtype(MLLM_TYPE_F32);
    logits->reshape(1, 1, 1, 32);
    logits->alloc();
    for (int key = 0; key < 32; ++key) {
        const float score = static_cast<float>((key * 17) % 37) * 0.1F
            + static_cast<float>(key) * 1.0e-4F;
        logits->setDataAt<float>(0, 0, 0, key, score);
    }

    auto value = std::make_shared<Tensor>(bn_);
    value->setCtype(BHDS);
    value->setDtype(MLLM_TYPE_F32);
    value->reshape(1, 1, 32, 3);
    value->alloc();
    for (int key = 0; key < 32; ++key) {
        for (int d = 0; d < 3; ++d) {
            value->setDataAt<float>(0, 0, key, d,
                                    static_cast<float>(key * 3 + d));
        }
    }

    auto exact_output = std::make_shared<Tensor>(bn_);
    auto sampled_output = std::make_shared<Tensor>(bn_);
    CPUSparseSoftmaxValueFunc exact_op(
        bn_, "sparse-attention-exact", 2, 0.75F, false, 0);
    CPUSparseSoftmaxValueFunc sampled_op(
        bn_, "sparse-attention-sampled", 2, 0.75F, false, 8);
    for (const auto &pair : {
             std::pair<CPUSparseSoftmaxValueFunc *, std::shared_ptr<Tensor>>(
                 &exact_op, exact_output),
             std::pair<CPUSparseSoftmaxValueFunc *, std::shared_ptr<Tensor>>(
                 &sampled_op, sampled_output)}) {
        ASSERT_EQ(pair.first->reshape({logits, value}, {pair.second}),
                  MLLM_NO_ERROR);
        ASSERT_EQ(pair.first->setUp({logits, value}, {pair.second}),
                  MLLM_NO_ERROR);
        ASSERT_EQ(pair.first->execute({logits, value}, {pair.second}),
                  MLLM_NO_ERROR);
    }
    for (int d = 0; d < 3; ++d) {
        EXPECT_FLOAT_EQ(sampled_output->dataAt<float>(0, 0, 0, d),
                        exact_output->dataAt<float>(0, 0, 0, d));
    }
}

TEST_F(CPUTest, CPUSparseSoftmaxValueSamplesAtFourTimesSampleSizeWithTies) {
    constexpr int sample_size = 4;
    constexpr int key_len = 4 * sample_size;
    auto logits = std::make_shared<Tensor>(bn_);
    logits->setCtype(BHSD);
    logits->setDtype(MLLM_TYPE_F32);
    logits->reshape(1, 1, 1, key_len);
    logits->alloc();
    for (int key = 0; key < key_len; ++key) {
        logits->setDataAt<float>(0, 0, 0, key, 1.0F);
    }

    auto value = std::make_shared<Tensor>(bn_);
    value->setCtype(BHDS);
    value->setDtype(MLLM_TYPE_F32);
    value->reshape(1, 1, key_len, 1);
    value->alloc();
    for (int key = 0; key < key_len; ++key) {
        value->setDataAt<float>(0, 0, key, 0,
                                static_cast<float>(key + 1));
    }

    auto exact_output = std::make_shared<Tensor>(bn_);
    auto sampled_output = std::make_shared<Tensor>(bn_);
    CPUSparseSoftmaxValueFunc exact_op(
        bn_, "sparse-attention-four-s-exact", 1, 0.5F, false, 0);
    CPUSparseSoftmaxValueFunc sampled_op(
        bn_, "sparse-attention-four-s-sampled", 1, 0.5F, false,
        sample_size);
    ASSERT_EQ(exact_op.reshape({logits, value}, {exact_output}), MLLM_NO_ERROR);
    ASSERT_EQ(exact_op.setUp({logits, value}, {exact_output}), MLLM_NO_ERROR);
    ASSERT_EQ(exact_op.execute({logits, value}, {exact_output}), MLLM_NO_ERROR);

    CPUSparseSelectionStats::reset();
    ASSERT_EQ(sampled_op.reshape({logits, value}, {sampled_output}),
              MLLM_NO_ERROR);
    ASSERT_EQ(sampled_op.setUp({logits, value}, {sampled_output}),
              MLLM_NO_ERROR);
    ASSERT_EQ(sampled_op.execute({logits, value}, {sampled_output}),
              MLLM_NO_ERROR);
    const auto stats = CPUSparseSelectionStats::snapshot();

    EXPECT_FLOAT_EQ(sampled_output->dataAt<float>(0, 0, 0, 0),
                    exact_output->dataAt<float>(0, 0, 0, 0));
    EXPECT_FLOAT_EQ(sampled_output->dataAt<float>(0, 0, 0, 0), 4.5F);
    EXPECT_EQ(stats.sampled_rows, 1U);
    EXPECT_EQ(stats.fallback_rows, 0U);
    EXPECT_EQ(stats.candidate_elements, static_cast<uint64_t>(key_len));
}

TEST_F(CPUTest, CPUSparseSoftmaxValueDoesNotSampleBelowFourTimesSampleSize) {
    constexpr int sample_size = 4;
    constexpr int key_len = 4 * sample_size - 1;
    auto logits = std::make_shared<Tensor>(bn_);
    logits->setCtype(BHSD);
    logits->setDtype(MLLM_TYPE_F32);
    logits->reshape(1, 1, 1, key_len);
    logits->alloc();
    for (int key = 0; key < key_len; ++key) {
        const float score = static_cast<float>((key * 7) % 17)
            + static_cast<float>(key) * 1.0e-3F;
        logits->setDataAt<float>(0, 0, 0, key, score);
    }

    auto value = std::make_shared<Tensor>(bn_);
    value->setCtype(BHDS);
    value->setDtype(MLLM_TYPE_F32);
    value->reshape(1, 1, key_len, 2);
    value->alloc();
    for (int key = 0; key < key_len; ++key) {
        value->setDataAt<float>(0, 0, key, 0,
                                static_cast<float>(key + 1));
        value->setDataAt<float>(0, 0, key, 1,
                                static_cast<float>(key * key + 1));
    }

    auto exact_output = std::make_shared<Tensor>(bn_);
    auto boundary_output = std::make_shared<Tensor>(bn_);
    CPUSparseSoftmaxValueFunc exact_op(
        bn_, "sparse-attention-below-four-s-exact", 1, 0.5F, false, 0);
    CPUSparseSoftmaxValueFunc boundary_op(
        bn_, "sparse-attention-below-four-s", 1, 0.5F, false, sample_size);
    ASSERT_EQ(exact_op.reshape({logits, value}, {exact_output}), MLLM_NO_ERROR);
    ASSERT_EQ(exact_op.setUp({logits, value}, {exact_output}), MLLM_NO_ERROR);
    ASSERT_EQ(exact_op.execute({logits, value}, {exact_output}), MLLM_NO_ERROR);

    CPUSparseSelectionStats::reset();
    ASSERT_EQ(boundary_op.reshape({logits, value}, {boundary_output}),
              MLLM_NO_ERROR);
    ASSERT_EQ(boundary_op.setUp({logits, value}, {boundary_output}),
              MLLM_NO_ERROR);
    ASSERT_EQ(boundary_op.execute({logits, value}, {boundary_output}),
              MLLM_NO_ERROR);
    const auto stats = CPUSparseSelectionStats::snapshot();

    for (int d = 0; d < 2; ++d) {
        EXPECT_FLOAT_EQ(boundary_output->dataAt<float>(0, 0, 0, d),
                        exact_output->dataAt<float>(0, 0, 0, d));
    }
    EXPECT_EQ(stats.sampled_rows, 0U);
    EXPECT_EQ(stats.fallback_rows, 0U);
    EXPECT_EQ(stats.candidate_elements, 0U);
}

TEST_F(CPUTest, CPUSparseSoftmaxValueKeepAllMatchesDenseFromBSHDValue) {
    auto logits = std::make_shared<Tensor>(bn_);
    logits->setCtype(BSHD);
    logits->setDtype(MLLM_TYPE_F32);
    logits->reshape(1, 1, 2, 4);
    logits->alloc();
    const float scores[2][4] = {{0.0F, 1.0F, 3.0F, 100.0F},
                                {5.0F, 4.0F, 3.0F, 2.0F}};
    for (int s = 0; s < 2; ++s) {
        for (int d = 0; d < 4; ++d) {
            logits->setDataAt<float>(0, 0, s, d, scores[s][d]);
        }
    }

    auto dense_value = std::make_shared<Tensor>(bn_);
    auto sparse_value = std::make_shared<Tensor>(bn_);
    for (const auto &value : {dense_value, sparse_value}) {
        value->setCtype(BSHD);
        value->setDtype(MLLM_TYPE_F16);
        value->reshape(1, 1, 4, 3);
        value->alloc();
        for (int s = 0; s < 4; ++s) {
            for (int d = 0; d < 3; ++d) {
                value->setDataAt<mllm_fp16_t>(
                    0, 0, s, d, MLLM_FP32_TO_FP16(10.0F * s + d + 1.0F));
            }
        }
    }

    auto probabilities = std::make_shared<Tensor>(bn_);
    CPUSoftMax softmax(bn_, "dense-softmax", DIMENSION, true, 2);
    ASSERT_EQ(softmax.reshape({logits}, {probabilities}), MLLM_NO_ERROR);
    ASSERT_EQ(softmax.setUp({logits}, {probabilities}), MLLM_NO_ERROR);
    ASSERT_EQ(softmax.execute({logits}, {probabilities}), MLLM_NO_ERROR);

    transposeAttentionValueChannels(*dense_value);
    auto dense_output = std::make_shared<Tensor>(bn_);
    dense_output->setCtype(BSHD);
    dense_output->setDtype(MLLM_TYPE_F32);
    dense_output->reshape(1, 1, 2, 3);
    dense_output->alloc();
    ASSERT_EQ(mat_mul(probabilities.get(), dense_value.get(), dense_output.get(),
                      false, nullptr, false, false, 2),
              MLLM_NO_ERROR);

    auto sparse_output = std::make_shared<Tensor>(bn_);
    CPUSparseSoftmaxValueFunc sparse_op(
        bn_, "keep-all-sparse-value", 2, 1.0e-6F, true);
    ASSERT_EQ(sparse_op.reshape({logits, sparse_value}, {sparse_output}),
              MLLM_NO_ERROR);
    ASSERT_EQ(sparse_op.setUp({logits, sparse_value}, {sparse_output}),
              MLLM_NO_ERROR);
    ASSERT_EQ(sparse_op.execute({logits, sparse_value}, {sparse_output}),
              MLLM_NO_ERROR);

    EXPECT_EQ(dense_value->ctype(), BHDS);
    EXPECT_EQ(sparse_value->ctype(), BHDS);
    for (int s = 0; s < 2; ++s) {
        for (int d = 0; d < 3; ++d) {
            EXPECT_FLOAT_EQ(dense_output->dataAt<float>(0, 0, s, d),
                            sparse_output->dataAt<float>(0, 0, s, d));
        }
    }
}

TEST_F(CPUTest, CPUSparseSoftmaxValuePropagatesBHDSLayoutAcrossCacheChunks) {
    CPUKVCacheNPU value_cache(bn_, "test.v_cache", 1, 8, 2);
    CPUSparseSoftmaxValueFunc sparse_op(
        bn_, "cached-sparse-value", 2, 0.5F, true);
    auto cached_value = std::make_shared<Tensor>(bn_);
    auto output = std::make_shared<Tensor>(bn_);

    auto run_chunk = [&](const std::vector<float> &values,
                         const std::vector<float> &scores,
                         int expected_key_len) {
        auto input = std::make_shared<Tensor>(bn_);
        input->setCtype(BHDS);
        input->setDtype(MLLM_TYPE_F16);
        input->reshape(1, 1, 2, 1);
        input->alloc();
        for (int s = 0; s < 2; ++s) {
            input->setDataAt<mllm_fp16_t>(
                0, 0, s, 0, MLLM_FP32_TO_FP16(values[s]));
        }

        auto logits = std::make_shared<Tensor>(bn_);
        logits->setCtype(BSHD);
        logits->setDtype(MLLM_TYPE_F32);
        logits->reshape(1, 1, 2, expected_key_len);
        logits->alloc();
        for (int s = 0; s < 2; ++s) {
            for (int key = 0; key < expected_key_len; ++key) {
                logits->setDataAt<float>(
                    0, 0, s, key, scores[s * expected_key_len + key]);
            }
        }

        ASSERT_EQ(value_cache.reshape({input}, {cached_value}), MLLM_NO_ERROR);
        ASSERT_EQ(sparse_op.reshape({logits, cached_value}, {output}),
                  MLLM_NO_ERROR);
        ASSERT_EQ(value_cache.setUp({input}, {cached_value}), MLLM_NO_ERROR);
        ASSERT_EQ(sparse_op.setUp({logits, cached_value}, {output}),
                  MLLM_NO_ERROR);
        ASSERT_EQ(cached_value->ctype(), BHDS);
        ASSERT_NE(cached_value->masterTensor(), nullptr);
        ASSERT_EQ(cached_value->masterTensor()->ctype(), BHDS);
        ASSERT_EQ(value_cache.execute({input}, {cached_value}), MLLM_NO_ERROR);
        ASSERT_EQ(sparse_op.execute({logits, cached_value}, {output}),
                  MLLM_NO_ERROR);
    };

    // First chunk: row 0 retains key 0; row 1 retains key 1.
    run_chunk({10.0F, 20.0F}, {5.0F, 100.0F,
                                1.0F, 3.0F}, 2);
    EXPECT_FLOAT_EQ(output->dataAt<float>(0, 0, 0, 0), 10.0F);
    EXPECT_FLOAT_EQ(output->dataAt<float>(0, 0, 1, 0), 20.0F);

    // Second chunk includes the old prefix. Top-K must read both old cache
    // values and newly appended values from the still-BHDS master cache.
    run_chunk({30.0F, 40.0F}, {9.0F, 1.0F, 8.0F, 100.0F,
                                1.0F, 9.0F, 2.0F, 8.0F}, 4);
    const float row0_expected =
        (std::exp(9.0F) * 10.0F + std::exp(8.0F) * 30.0F)
        / (std::exp(9.0F) + std::exp(8.0F));
    const float row1_expected =
        (std::exp(9.0F) * 20.0F + std::exp(8.0F) * 40.0F)
        / (std::exp(9.0F) + std::exp(8.0F));
    EXPECT_NEAR(output->dataAt<float>(0, 0, 0, 0), row0_expected, 2.0e-2F);
    EXPECT_NEAR(output->dataAt<float>(0, 0, 1, 0), row1_expected, 2.0e-2F);
}
