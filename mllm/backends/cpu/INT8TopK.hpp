#ifndef MLLM_CPU_INT8_TOP_K_HPP
#define MLLM_CPU_INT8_TOP_K_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace mllm {

// Exact stable Top-k for signed INT8 scores. Scores rank from greatest to
// least, equal scores prefer the smaller key index, and output indices are
// stored in increasing key order. Since INT8 has only 256 possible values,
// one histogram pass plus one ordered emission pass is sufficient.
class INT8TopK final {
public:
    using Score = std::int8_t;
    using Index = std::int32_t;

    INT8TopK() = delete;

    static bool selectRow(const Score *scores, std::size_t score_count,
                          std::size_t keep, Index *indices) noexcept {
        if (scores == nullptr || indices == nullptr || score_count == 0
            || keep == 0 || keep > score_count
            || score_count
                > static_cast<std::size_t>(
                      std::numeric_limits<Index>::max()) + 1U) {
            return false;
        }
        if (keep == score_count) {
            for (std::size_t column = 0; column < score_count; ++column) {
                indices[column] = static_cast<Index>(column);
            }
            return true;
        }

        if (score_count
            <= static_cast<std::size_t>(
                std::numeric_limits<std::uint16_t>::max())) {
            return selectRowImpl<std::uint16_t, 8>(
                scores, score_count, keep, indices);
        }
        return selectRowImpl<std::uint32_t, 4>(
            scores, score_count, keep, indices);
    }

    static bool selectCausalRows(const Score *scores,
                                 std::size_t row_count,
                                 std::size_t score_row_stride,
                                 std::size_t first_valid_count,
                                 std::size_t maximum_valid_count,
                                 const Index *output_row_offsets,
                                 Index *indices) noexcept {
        return selectCausalRowRange(
            scores, row_count, 0, row_count, score_row_stride,
            first_valid_count, maximum_valid_count, output_row_offsets,
            indices);
    }

    // Select a disjoint range of causal rows without rebasing either the score
    // tensor or its output offsets.  This is the primitive used by the
    // cooperative pipeline executor: workers can split one head by rows and
    // write directly into non-overlapping parts of the final index buffer.
    static bool selectCausalRowRange(
        const Score *scores, std::size_t total_row_count,
        std::size_t row_begin, std::size_t row_count,
        std::size_t score_row_stride, std::size_t first_valid_count,
        std::size_t maximum_valid_count,
        const Index *output_row_offsets, Index *indices) noexcept {
        if (scores == nullptr || output_row_offsets == nullptr
            || indices == nullptr || total_row_count == 0 || row_count == 0
            || row_begin > total_row_count
            || row_count > total_row_count - row_begin
            || score_row_stride == 0 || first_valid_count == 0
            || maximum_valid_count == 0
            || maximum_valid_count > score_row_stride
            || maximum_valid_count
                > static_cast<std::size_t>(
                      std::numeric_limits<std::uint16_t>::max())
            || total_row_count - 1U
                > (std::numeric_limits<std::size_t>::max()
                   - maximum_valid_count) / score_row_stride
            || output_row_offsets[0] != 0) {
            return false;
        }
        constexpr std::size_t rows_per_block = 4;
        const std::size_t row_end = row_begin + row_count;
        std::size_t row = row_begin;
        for (; row + rows_per_block <= row_end;
             row += rows_per_block) {
            std::array<const Score *, rows_per_block> row_scores{};
            std::array<std::size_t, rows_per_block> valid_counts{};
            std::array<std::size_t, rows_per_block> keep_counts{};
            std::size_t common_valid = maximum_valid_count;
            for (std::size_t lane = 0; lane < rows_per_block; ++lane) {
                const std::size_t current_row = row + lane;
                const std::size_t valid = causalValidCount(
                    first_valid_count, maximum_valid_count, current_row);
                const Index begin = output_row_offsets[current_row];
                const Index end = output_row_offsets[current_row + 1U];
                if (begin < 0 || end <= begin
                    || static_cast<std::size_t>(end - begin) > valid) {
                    return false;
                }
                row_scores[lane]
                    = scores + current_row * score_row_stride;
                valid_counts[lane] = valid;
                keep_counts[lane] = static_cast<std::size_t>(end - begin);
                common_valid = std::min(common_valid, valid);
            }

            std::array<std::array<std::uint16_t, 256>, rows_per_block>
                histograms{};
            for (std::size_t column = 0; column < common_valid; ++column) {
                for (std::size_t lane = 0; lane < rows_per_block; ++lane) {
                    ++histograms[lane][bucket(row_scores[lane][column])];
                }
            }
            for (std::size_t lane = 0; lane < rows_per_block; ++lane) {
                for (std::size_t column = common_valid;
                     column < valid_counts[lane]; ++column) {
                    ++histograms[lane][bucket(row_scores[lane][column])];
                }
            }
            for (std::size_t lane = 0; lane < rows_per_block; ++lane) {
                std::size_t greater_count = 0;
                unsigned threshold_bucket = 0;
                for (unsigned candidate = 256; candidate-- > 0;) {
                    const std::size_t candidate_count
                        = histograms[lane][candidate];
                    if (greater_count + candidate_count
                        >= keep_counts[lane]) {
                        threshold_bucket = candidate;
                        break;
                    }
                    greater_count += candidate_count;
                }
                const Score threshold = static_cast<Score>(
                    static_cast<int>(threshold_bucket) - 128);
                if (!emitThreshold(
                        row_scores[lane], valid_counts[lane],
                        keep_counts[lane],
                        indices + output_row_offsets[row + lane], threshold,
                        greater_count)) {
                    return false;
                }
            }
        }
        for (; row < row_end; ++row) {
            const std::size_t valid = causalValidCount(
                first_valid_count, maximum_valid_count, row);
            const Index begin = output_row_offsets[row];
            const Index end = output_row_offsets[row + 1U];
            if (begin < 0 || end <= begin
                || static_cast<std::size_t>(end - begin) > valid
                || !selectRow(scores + row * score_row_stride, valid,
                              static_cast<std::size_t>(end - begin),
                              indices + begin)) {
                return false;
            }
        }
        return true;
    }

private:
    static std::size_t causalValidCount(std::size_t first_valid_count,
                                        std::size_t maximum_valid_count,
                                        std::size_t row) noexcept {
        if (first_valid_count >= maximum_valid_count
            || row >= maximum_valid_count - first_valid_count) {
            return maximum_valid_count;
        }
        return first_valid_count + row;
    }

    template <typename Counter, std::size_t BankCount>
    static bool selectRowImpl(const Score *scores, std::size_t score_count,
                              std::size_t keep, Index *indices) noexcept {
        static_assert(BankCount == 4 || BankCount == 8);

        /* A single histogram has a loop-carried read/modify/write dependency,
         * especially for the heavily repeated central INT8 buckets.  The
         * eight banks expose independent updates for the normal attention
         * path; very long generic rows retain a four-bank uint32 fallback. */
        std::array<std::array<Counter, 256>, BankCount> banks{};
        std::size_t column = 0;
        if constexpr (BankCount == 8) {
            for (; column + 8 <= score_count; column += 8) {
                ++banks[0][bucket(scores[column])];
                ++banks[1][bucket(scores[column + 1])];
                ++banks[2][bucket(scores[column + 2])];
                ++banks[3][bucket(scores[column + 3])];
                ++banks[4][bucket(scores[column + 4])];
                ++banks[5][bucket(scores[column + 5])];
                ++banks[6][bucket(scores[column + 6])];
                ++banks[7][bucket(scores[column + 7])];
            }
        } else {
            for (; column + 4 <= score_count; column += 4) {
                ++banks[0][bucket(scores[column])];
                ++banks[1][bucket(scores[column + 1])];
                ++banks[2][bucket(scores[column + 2])];
                ++banks[3][bucket(scores[column + 3])];
            }
        }
        for (; column < score_count; ++column) {
            ++banks[column & (BankCount - 1U)][bucket(scores[column])];
        }
        std::array<Counter, 256> histogram{};
        for (std::size_t value = 0; value < histogram.size(); ++value) {
            std::size_t total = 0;
            for (std::size_t bank = 0; bank < BankCount; ++bank) {
                total += banks[bank][value];
            }
            histogram[value] = static_cast<Counter>(total);
        }

        std::size_t greater_count = 0;
        unsigned int threshold_bucket = 0;
        for (unsigned int candidate = 256; candidate-- > 0;) {
            const std::size_t candidate_count = histogram[candidate];
            if (greater_count + candidate_count >= keep) {
                threshold_bucket = candidate;
                break;
            }
            greater_count += candidate_count;
        }
        const Score threshold = static_cast<Score>(
            static_cast<int>(threshold_bucket) - 128);
        return emitThreshold(scores, score_count, keep, indices, threshold,
                             greater_count);
    }

    static bool emitThreshold(const Score *scores,
                              std::size_t score_count,
                              std::size_t keep,
                              Index *indices,
                              Score threshold,
                              std::size_t greater_count) noexcept {
        const std::size_t threshold_ties_to_take = keep - greater_count;
        std::size_t threshold_ties_taken = 0;
        std::size_t selected = 0;
        for (std::size_t column = 0; column < score_count; ++column) {
            const Score score = scores[column];
            const bool take_threshold = score == threshold
                && threshold_ties_taken < threshold_ties_to_take;
            if (score > threshold || take_threshold) {
                indices[selected++] = static_cast<Index>(column);
                threshold_ties_taken +=
                    static_cast<std::size_t>(take_threshold);
            }
        }
        return selected == keep;
    }
    static constexpr unsigned int bucket(Score score) noexcept {
        return static_cast<unsigned int>(static_cast<int>(score) + 128);
    }
};

} // namespace mllm

#endif // MLLM_CPU_INT8_TOP_K_HPP
