#ifndef MLLM_CPU_HMX_PIPELINE_LATENCY_RECORDER_HPP
#define MLLM_CPU_HMX_PIPELINE_LATENCY_RECORDER_HPP

#include "AttentionProfiler.hpp"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>

namespace mllm::hmx_pipeline {

// Raw device-side timing sink used by the offline profiler. The file contains
// one sample per line; aggregation, warm-up removal, and metadata binding are
// intentionally performed by the host profiler.
class LatencyRecorder final {
public:
    static bool enabled() noexcept {
        const char *path = std::getenv(
            "MLLM_HMX_PIPELINE_RAW_PROFILE_OUTPUT");
        return path != nullptr && path[0] != '\0';
    }

    static void npu(int key_len, int heads, float q_scale, float k_scale,
                    uint64_t elapsed_ns) {
        if (!stageEnabled("npu")) return;
        write("npu\t" + std::to_string(key_len) + "\t"
              + std::to_string(heads) + "\t" + floatString(q_scale) + "\t"
              + floatString(k_scale) + "\t"
              + std::to_string(elapsed_ns / 1000.0));
    }

    static void head(const char *stage, int key_len, int layer, int head,
                     float retention, uint64_t elapsed_ns) {
        if (!stageEnabled("heads")) return;
        write(std::string(stage) + "\t" + std::to_string(key_len) + "\t"
              + std::to_string(layer) + "\t" + std::to_string(head) + "\t"
              + floatString(retention) + "\t"
              + std::to_string(elapsed_ns / 1000.0));
    }

private:
    static bool stageEnabled(const char *category) noexcept {
        const char *value = std::getenv(
            "MLLM_HMX_PIPELINE_RAW_PROFILE_STAGE");
        return value == nullptr || value[0] == '\0'
            || std::strcmp(value, "all") == 0
            || std::strcmp(value, category) == 0;
    }

    static std::string floatString(float value) {
        std::ostringstream output;
        output << std::setprecision(9) << value;
        return output.str();
    }

    static void write(const std::string &line) {
        const char *path = std::getenv(
            "MLLM_HMX_PIPELINE_RAW_PROFILE_OUTPUT");
        if (path == nullptr || path[0] == '\0') return;
        std::lock_guard<std::mutex> lock(mutex());
        auto &state = stream();
        if (state.path != path) {
            if (state.output.is_open()) state.output.close();
            state.path = path;
            state.output.open(path, std::ios::out | std::ios::app);
        }
        if (state.output) {
            state.output << line << '\n';
            state.output.flush();
        }
    }

    struct StreamState {
        std::string path;
        std::ofstream output;
    };

    static StreamState &stream() {
        static auto *state = new StreamState();
        return *state;
    }

    static std::mutex &mutex() {
        static auto *value = new std::mutex();
        return *value;
    }
};

} // namespace mllm::hmx_pipeline

#endif // MLLM_CPU_HMX_PIPELINE_LATENCY_RECORDER_HPP
