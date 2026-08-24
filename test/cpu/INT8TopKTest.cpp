#include "backends/cpu/INT8TopK.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

namespace {

using Score = mllm::INT8TopK::Score;
using Index = mllm::INT8TopK::Index;

std::vector<Index> referenceTopK(const std::vector<Score> &scores,
                                 std::size_t keep) {
    std::vector<Index> order(scores.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](Index left, Index right) {
        if (scores[left] != scores[right]) {
            return scores[left] > scores[right];
        }
        return left < right;
    });
    order.resize(keep);
    std::sort(order.begin(), order.end());
    return order;
}

TEST(INT8TopKTest, HandlesTiesAndSignedExtremes) {
    const std::vector<Score> scores = {
        -128, 127, 0, 127, -1, 0, -128, 42};
    for (std::size_t keep = 1; keep <= scores.size(); ++keep) {
        std::vector<Index> actual(keep, -1);
        ASSERT_TRUE(mllm::INT8TopK::selectRow(
            scores.data(), scores.size(), keep, actual.data()));
        EXPECT_EQ(actual, referenceTopK(scores, keep));
        EXPECT_TRUE(std::is_sorted(actual.begin(), actual.end()));
    }
}

TEST(INT8TopKTest, RandomRowsMatchStableReference) {
    std::mt19937 generator(0x18a8U);
    std::uniform_int_distribution<int> distribution(-128, 127);
    for (const std::size_t length :
         {1U, 2U, 31U, 127U, 128U, 129U, 1020U, 1021U, 2048U,
          4097U}) {
        std::vector<Score> scores(length);
        for (Score &score : scores) {
            score = static_cast<Score>(distribution(generator));
        }
        for (const std::size_t keep : std::vector<std::size_t>{
                 1U, (length + 4U) / 5U, length}) {
            std::vector<Index> actual(keep, -1);
            ASSERT_TRUE(mllm::INT8TopK::selectRow(
                scores.data(), scores.size(), keep, actual.data()));
            EXPECT_EQ(actual, referenceTopK(scores, keep));
        }
    }
}

TEST(INT8TopKTest, RepeatedScoresRemainExactAcrossCounterBoundary) {
    for (const std::size_t length :
         {1024U, 2040U, 2041U, 4160U, 65535U, 65536U}) {
        const std::vector<Score> scores(length, static_cast<Score>(42));
        const std::size_t keep = (length + 4U) / 5U;
        std::vector<Index> actual(keep, -1);
        ASSERT_TRUE(mllm::INT8TopK::selectRow(
            scores.data(), scores.size(), keep, actual.data()));
        for (std::size_t index = 0; index < keep; ++index) {
            EXPECT_EQ(actual[index], static_cast<Index>(index));
        }
    }
}

TEST(INT8TopKTest, CausalRowBlocksMatchStableReference) {
    constexpr std::size_t row_count = 128;
    constexpr std::size_t row_stride = 1024;
    constexpr std::size_t first_valid = 513;
    constexpr std::size_t maximum_valid = 1024;
    std::mt19937 generator(0xca05a1U);
    std::uniform_int_distribution<int> distribution(-128, 127);
    std::vector<Score> scores(row_count * row_stride);
    for (Score &score : scores) {
        score = static_cast<Score>(distribution(generator));
    }
    std::vector<Index> offsets(row_count + 1U, 0);
    for (std::size_t row = 0; row < row_count; ++row) {
        const std::size_t valid = std::min(
            maximum_valid, first_valid + row);
        const std::size_t keep = std::max<std::size_t>(1U,
            (valid * 14U + 99U) / 100U);
        offsets[row + 1U] = offsets[row] + static_cast<Index>(keep);
    }
    std::vector<Index> actual(static_cast<std::size_t>(offsets.back()), -1);
    ASSERT_TRUE(mllm::INT8TopK::selectCausalRows(
        scores.data(), row_count, row_stride, first_valid, maximum_valid,
        offsets.data(), actual.data()));
    for (std::size_t row = 0; row < row_count; ++row) {
        const std::size_t valid = std::min(
            maximum_valid, first_valid + row);
        const std::size_t begin = static_cast<std::size_t>(offsets[row]);
        const std::size_t keep = static_cast<std::size_t>(
            offsets[row + 1U] - offsets[row]);
        const std::vector<Score> row_scores(
            scores.begin() + static_cast<std::ptrdiff_t>(row * row_stride),
            scores.begin() + static_cast<std::ptrdiff_t>(
                row * row_stride + valid));
        const std::vector<Index> row_actual(
            actual.begin() + static_cast<std::ptrdiff_t>(begin),
            actual.begin() + static_cast<std::ptrdiff_t>(begin + keep));
        EXPECT_EQ(row_actual, referenceTopK(row_scores, keep));
    }
}

TEST(INT8TopKTest, CausalRowBlocksHandleTiesAndRemainder) {
    constexpr std::size_t row_count = 10;
    constexpr std::size_t row_stride = 32;
    std::vector<Score> scores(row_count * row_stride,
                              static_cast<Score>(7));
    std::vector<Index> offsets(row_count + 1U, 0);
    for (std::size_t row = 0; row < row_count; ++row) {
        offsets[row + 1U] = offsets[row]
            + static_cast<Index>((row + 2U) / 2U);
    }
    std::vector<Index> actual(static_cast<std::size_t>(offsets.back()), -1);
    ASSERT_TRUE(mllm::INT8TopK::selectCausalRows(
        scores.data(), row_count, row_stride, 1U, row_stride,
        offsets.data(), actual.data()));
    for (std::size_t row = 0; row < row_count; ++row) {
        const std::size_t begin = static_cast<std::size_t>(offsets[row]);
        const std::size_t keep = static_cast<std::size_t>(
            offsets[row + 1U] - offsets[row]);
        for (std::size_t selected = 0; selected < keep; ++selected) {
            EXPECT_EQ(actual[begin + selected],
                      static_cast<Index>(selected));
        }
    }
}

TEST(INT8TopKTest, DisjointCausalRowRangesMatchWholeHead) {
    constexpr std::size_t row_count = 128;
    constexpr std::size_t row_stride = 1024;
    constexpr std::size_t first_valid = 897;
    std::mt19937 generator(0xc001abU);
    std::uniform_int_distribution<int> distribution(-128, 127);
    std::vector<Score> scores(row_count * row_stride);
    for (std::size_t index = 0; index < scores.size(); ++index) {
        scores[index] = index % 29U == 0U
            ? static_cast<Score>(17)
            : static_cast<Score>(distribution(generator));
    }
    for (const std::size_t retention_percent : {10U, 20U, 55U, 100U}) {
        std::vector<Index> offsets(row_count + 1U, 0);
        for (std::size_t row = 0; row < row_count; ++row) {
            const std::size_t valid = std::min(
                row_stride, first_valid + row);
            const std::size_t keep = std::max<std::size_t>(
                1U, (valid * retention_percent + 99U) / 100U);
            offsets[row + 1U] = offsets[row]
                + static_cast<Index>(keep);
        }
        std::vector<Index> whole(
            static_cast<std::size_t>(offsets.back()), -1);
        std::vector<Index> split(whole.size(), -1);
        ASSERT_TRUE(mllm::INT8TopK::selectCausalRows(
            scores.data(), row_count, row_stride, first_valid, row_stride,
            offsets.data(), whole.data()));
        ASSERT_TRUE(mllm::INT8TopK::selectCausalRowRange(
            scores.data(), row_count, 0, 64, row_stride, first_valid,
            row_stride, offsets.data(), split.data()));
        ASSERT_TRUE(mllm::INT8TopK::selectCausalRowRange(
            scores.data(), row_count, 64, 64, row_stride, first_valid,
            row_stride, offsets.data(), split.data()));
        EXPECT_EQ(split, whole)
            << "retention_percent=" << retention_percent;
    }
}

TEST(INT8TopKTest, RejectsInvalidArguments) {
    const Score score = 0;
    Index index = -1;
    EXPECT_FALSE(mllm::INT8TopK::selectRow(nullptr, 1, 1, &index));
    EXPECT_FALSE(mllm::INT8TopK::selectRow(&score, 1, 1, nullptr));
    EXPECT_FALSE(mllm::INT8TopK::selectRow(&score, 0, 1, &index));
    EXPECT_FALSE(mllm::INT8TopK::selectRow(&score, 1, 0, &index));
    EXPECT_FALSE(mllm::INT8TopK::selectRow(&score, 1, 2, &index));
}

} // namespace
