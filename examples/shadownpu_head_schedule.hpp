#ifndef MLLM_EXAMPLES_SHADOWNPU_HEAD_SCHEDULE_HPP
#define MLLM_EXAMPLES_SHADOWNPU_HEAD_SCHEDULE_HPP

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mllm::examples {

struct ShadowNPUHeadSchedule {
    std::vector<std::vector<float>> retentions;
    float average_retention = 0.0F;
    int dense_heads = 0;
    std::string normalized_dense_layers = "none";
};

inline std::string denseLayerRanges(
    const std::vector<std::vector<float>> &retentions);

inline std::string denseLayerRanges(
    const std::vector<std::vector<float>> &retentions) {
    std::string result;
    int run_begin = -1;
    const auto append_run = [&](int begin, int end, std::string &out) {
        if (!out.empty()) out += ',';
        out += std::to_string(begin);
        if (end != begin) out += '-' + std::to_string(end);
    };
    for (int layer = 0; layer <= static_cast<int>(retentions.size()); ++layer) {
        const bool dense = layer < static_cast<int>(retentions.size())
            && !retentions[layer].empty()
            && std::all_of(retentions[layer].begin(), retentions[layer].end(),
                           [](float retention) { return retention == 1.0F; });
        if (dense && run_begin < 0) run_begin = layer;
        if (!dense && run_begin >= 0) {
            append_run(run_begin, layer - 1, result);
            run_begin = -1;
        }
    }
    return result.empty() ? "none" : result;
}

inline ShadowNPUHeadSchedule loadShadowNPUHeadSchedule(
    const std::string &path, int layers, int heads) {
    if (layers <= 0 || heads <= 0) {
        throw std::invalid_argument(
            "ShadowNPU head schedule requires positive layer/head counts");
    }
    std::ifstream input(path);
    if (!input) {
        throw std::invalid_argument(
            "cannot open ShadowNPU head-retention profile: " + path);
    }
    ShadowNPUHeadSchedule schedule;
    schedule.retentions.assign(layers, std::vector<float>(heads));
    double sum = 0.0;
    for (int layer = 0; layer < layers; ++layer) {
        for (int head = 0; head < heads; ++head) {
            float retention = 0.0F;
            if (!(input >> retention)) {
                throw std::invalid_argument(
                    "ShadowNPU profile has fewer entries than the model: "
                    + path);
            }
            if (!(retention > 0.0F && retention <= 1.0F)
                || !std::isfinite(retention)) {
                throw std::invalid_argument(
                    "ShadowNPU head retention must be finite and in (0, 1]");
            }
            schedule.retentions[layer][head] = retention;
            sum += retention;
            if (retention == 1.0F) ++schedule.dense_heads;
        }
    }
    float trailing = 0.0F;
    if (input >> trailing) {
        throw std::invalid_argument(
            "ShadowNPU profile has more entries than the model: " + path);
    }
    schedule.average_retention = static_cast<float>(
        sum / static_cast<double>(layers * heads));
    schedule.normalized_dense_layers = denseLayerRanges(schedule.retentions);
    return schedule;
}

} // namespace mllm::examples

#endif // MLLM_EXAMPLES_SHADOWNPU_HEAD_SCHEDULE_HPP
