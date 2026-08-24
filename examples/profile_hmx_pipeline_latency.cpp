#include "backends/cpu/CPUBackend.hpp"
#include "backends/cpu/op/CPUSparseSoftmaxValueFunc.hpp"
#include "memory/SystemMemoryManager.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace mllm;

namespace {

constexpr int kHeads = 12;
constexpr int kHeadDim = 128;
constexpr int kCapacity = 4160;
int gQueryLen = 512;

struct ScalePair {
    float q = 0.0F;
    float k = 0.0F;
};

struct ManifestProfile {
    std::vector<ScalePair> scales;
    std::vector<int> head_counts;
};

std::vector<std::string> splitTabs(const std::string &line) {
    std::vector<std::string> fields;
    std::size_t begin = 0;
    while (true) {
        const std::size_t tab = line.find('\t', begin);
        fields.push_back(line.substr(begin, tab - begin));
        if (tab == std::string::npos) break;
        begin = tab + 1;
    }
    return fields;
}

ManifestProfile loadManifestProfile(const std::string &manifest) {
    std::ifstream input(manifest);
    if (!input) throw std::invalid_argument("cannot open manifest");
    std::map<int, std::set<std::pair<float, float>>> by_heads;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') continue;
        const auto fields = splitTabs(line);
        if (fields.empty() || fields[0] == "model") continue;
        if (fields.size() < 11) {
            throw std::invalid_argument("invalid manifest row");
        }
        const int heads = std::stoi(fields[1]);
        const ScalePair pair{std::stof(fields[7]), std::stof(fields[8])};
        if (!by_heads[heads].insert({pair.q, pair.k}).second) {
            throw std::invalid_argument("duplicate manifest scale pair");
        }
    }
    if (by_heads.empty()) {
        throw std::invalid_argument("manifest contains no operators");
    }
    const auto &expected = by_heads.begin()->second;
    if (expected.size() != 9U) {
        throw std::invalid_argument("profiler requires exactly 9 scale pairs");
    }
    ManifestProfile result;
    for (const auto &entry : by_heads) {
        if (entry.second != expected) {
            throw std::invalid_argument(
                "manifest head counts have inconsistent scale pairs");
        }
        result.head_counts.push_back(entry.first);
    }
    for (const auto &pair : expected) {
        result.scales.push_back({pair.first, pair.second});
    }
    return result;
}

std::vector<std::vector<float>> loadRetentions(const std::string &path) {
    std::ifstream input(path);
    if (!input) throw std::invalid_argument("cannot open head profile");
    std::vector<std::vector<float>> result;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        std::istringstream values(line);
        std::vector<float> row;
        float value = 0.0F;
        while (values >> value) row.push_back(value);
        if (row.size() != kHeads) {
            throw std::invalid_argument("head profile row must have 12 values");
        }
        result.push_back(std::move(row));
    }
    if (result.size() != 28U) {
        throw std::invalid_argument("head profile must have 28 layers");
    }
    return result;
}

std::shared_ptr<Tensor> makeTensor(Backend *backend, ChlType ctype,
                                   DataType dtype, int heads,
                                   int sequence, int dimension) {
    auto tensor = std::make_shared<Tensor>(backend);
    tensor->setCtype(ctype);
    tensor->setDtype(dtype);
    tensor->reshape(1, heads, sequence, dimension);
    tensor->alloc();
    return tensor;
}

void fillInputs(const std::shared_ptr<Tensor> &query,
                const std::shared_ptr<Tensor> &key,
                const std::shared_ptr<Tensor> &value,
                const std::vector<ScalePair> &head_scales) {
    const int heads = query->head();
    const int key_len = key->sequence();
    for (int head = 0; head < heads; ++head) {
        const float q_peak = head_scales[head].q * 127.0F;
        const float k_peak = head_scales[head].k * 127.0F;
        for (int row = 0; row < gQueryLen; ++row) {
            float *destination = query->ptrAt<float>(0, head, row, 0);
            for (int dimension = 0; dimension < kHeadDim; ++dimension) {
                const float fraction = static_cast<float>(
                    ((head * 17 + row * 13 + dimension * 7) % 201) - 100)
                    / 100.0F;
                destination[dimension] = q_peak * fraction;
            }
        }
        query->setDataAt<float>(0, head, 0, 0, q_peak);
        for (int token = 0; token < key_len; ++token) {
            auto *key_row = key->ptrAt<mllm_fp16_t>(0, head, token, 0);
            for (int dimension = 0; dimension < kHeadDim; ++dimension) {
                const float k_fraction = static_cast<float>(
                    ((head * 29 + token * 11 + dimension * 5) % 181) - 90)
                    / 90.0F;
                const float v = static_cast<float>(
                    ((head * 31 + token * 3 + dimension * 19) % 109) - 54)
                    / 64.0F;
                key_row[dimension] = MLLM_FP32_TO_FP16(
                    k_peak * k_fraction);
                value->setDataAt<mllm_fp16_t>(
                    0, head, token, dimension, MLLM_FP32_TO_FP16(v));
            }
        }
        key->setDataAt<mllm_fp16_t>(
            0, head, 0, 0, MLLM_FP32_TO_FP16(k_peak));
    }
}

void executeSamples(CPUPatternSparseAttentionFunc &op,
                    const std::shared_ptr<Tensor> &query,
                    const std::shared_ptr<Tensor> &key,
                    const std::shared_ptr<Tensor> &value,
                    const std::shared_ptr<Tensor> &output,
                    const std::string &raw_output,
                    const char *stage, int warmup, int repetitions) {
    if (op.reshape({query, key, value}, {output}) != MLLM_NO_ERROR
        || op.setUp({query, key, value}, {output}) != MLLM_NO_ERROR) {
        throw std::runtime_error("latency profiler setup failed");
    }
    unsetenv("MLLM_HMX_PIPELINE_RAW_PROFILE_OUTPUT");
    for (int iteration = 0; iteration < warmup; ++iteration) {
        if (op.execute({query, key, value}, {output}) != MLLM_NO_ERROR) {
            throw std::runtime_error("latency profiler warm-up failed");
        }
    }
    setenv("MLLM_HMX_PIPELINE_RAW_PROFILE_STAGE", stage, 1);
    setenv("MLLM_HMX_PIPELINE_RAW_PROFILE_OUTPUT", raw_output.c_str(), 1);
    for (int iteration = 0; iteration < repetitions; ++iteration) {
        if (op.execute({query, key, value}, {output}) != MLLM_NO_ERROR) {
            throw std::runtime_error("latency profiler measurement failed");
        }
    }
    unsetenv("MLLM_HMX_PIPELINE_RAW_PROFILE_OUTPUT");
}

void executeNpuSequence(
    CPUPatternSparseAttentionFunc &op,
    const std::shared_ptr<Tensor> &query,
    const std::shared_ptr<Tensor> &key,
    const std::shared_ptr<Tensor> &value,
    const std::shared_ptr<Tensor> &output,
    const std::vector<int> &lengths) {
    for (const int key_len : lengths) {
        key->reshape(1, query->head(), key_len, kHeadDim);
        value->reshape(1, query->head(), key_len, kHeadDim);
        if (op.reshape({query, key, value}, {output}) != MLLM_NO_ERROR
            || op.setUp({query, key, value}, {output}) != MLLM_NO_ERROR
            || op.execute({query, key, value}, {output}) != MLLM_NO_ERROR) {
            throw std::runtime_error(
                "incremental NPU latency sequence failed");
        }
    }
}

std::vector<int> keyLengths(int maximum) {
    std::vector<int> lengths;
    for (int length = gQueryLen; length <= std::min(4096, maximum);
         length += gQueryLen) {
        lengths.push_back(length);
    }
    if (maximum == kCapacity) lengths.push_back(kCapacity);
    return lengths;
}

void profileNpu(Backend *backend, const std::vector<ScalePair> &buckets,
                const std::vector<int> &head_counts,
                const std::string &raw_output, int maximum_key_len,
                int warmup, int repetitions) {
    setenv("MLLM_HMX_PIPELINE_PROFILE_NPU_ONLY", "1", 1);
    // Profile the producer before Top-k. With no ready callback the H12
    // CPU-packed call stays one synchronous RPC and LatencyRecorder's NPU row
    // excludes all consumer work.
    setenv("MLLM_HMX_PIPELINE_GROUP_HEADS", "0", 1);
    std::size_t completed = 0;
    const std::size_t total = buckets.size() * head_counts.size();
    const std::vector<int> lengths = keyLengths(maximum_key_len);
    for (std::size_t bucket = 0; bucket < buckets.size(); ++bucket) {
        for (const int fused_heads : head_counts) {
            std::vector<float> retentions(
                static_cast<std::size_t>(fused_heads), 0.2F);
            CPUPatternSparseAttentionFunc op(
                backend, "offline.npu", 1, 0.8F, true, 0.5F, 0, 0,
                false, 1, kCapacity, true, retentions, 2);
            const std::vector<ScalePair> scales(
                static_cast<std::size_t>(fused_heads), buckets[bucket]);
            auto query = makeTensor(
                backend, BSHD, MLLM_TYPE_F32, fused_heads,
                gQueryLen, kHeadDim);
            std::array<std::shared_ptr<Tensor>, 2> keys;
            std::array<std::shared_ptr<Tensor>, 2> values;
            for (std::size_t source = 0; source < keys.size(); ++source) {
                keys[source] = makeTensor(
                    backend, BSHD, MLLM_TYPE_F16, fused_heads,
                    maximum_key_len, kHeadDim);
                values[source] = makeTensor(
                    backend, BHDS, MLLM_TYPE_F16, fused_heads,
                    maximum_key_len, kHeadDim);
                fillInputs(query, keys[source], values[source], scales);
            }
            auto output = std::make_shared<Tensor>(backend);
            unsetenv("MLLM_HMX_PIPELINE_RAW_PROFILE_OUTPUT");
            for (int iteration = 0; iteration < warmup; ++iteration) {
                const std::size_t source = static_cast<std::size_t>(
                    iteration) % keys.size();
                executeNpuSequence(
                    op, query, keys[source], values[source], output, lengths);
            }
            setenv("MLLM_HMX_PIPELINE_RAW_PROFILE_STAGE", "npu", 1);
            setenv("MLLM_HMX_PIPELINE_RAW_PROFILE_OUTPUT",
                   raw_output.c_str(), 1);
            for (int iteration = 0; iteration < repetitions; ++iteration) {
                const std::size_t source = static_cast<std::size_t>(
                    warmup + iteration) % keys.size();
                executeNpuSequence(
                    op, query, keys[source], values[source], output, lengths);
            }
            unsetenv("MLLM_HMX_PIPELINE_RAW_PROFILE_OUTPUT");
            ++completed;
            if (completed % 12U == 0U || completed == total) {
                std::cerr << "PROFILE_NPU " << completed << '/' << total
                          << std::endl;
            }
        }
    }
    unsetenv("MLLM_HMX_PIPELINE_PROFILE_NPU_ONLY");
    setenv("MLLM_HMX_PIPELINE_GROUP_HEADS", "1", 1);
}

void profileHeads(Backend *backend,
                  const std::vector<std::vector<float>> &retentions,
                  const ScalePair &scale, const std::string &raw_output,
                  int maximum_key_len, int warmup, int repetitions) {
    unsetenv("MLLM_HMX_PIPELINE_PROFILE_NPU_ONLY");
    // Measure intrinsic one-head consumer costs on the production CPU
    // resources. The direct executor intentionally overlaps CPU4 sparse
    // assistance with the next Top-k head, which would make a fixed FIFO
    // head's recorded wall time depend on its position. The two-stage
    // profiler uses the same persistent cooperative Top-k and sparse workers
    // but records each head before applying the offline permutation.
    setenv("MLLM_HMX_PIPELINE_GROUP_HEADS", "1", 1);
    setenv("MLLM_HMX_PIPELINE_READY_GROUP_HEADS", "1", 1);
    setenv("MLLM_HMX_PIPELINE_EXECUTOR", "two-stage", 1);
    setenv("MLLM_HMX_PIPELINE_DIRECT_SPARSE_ASSIST_PERCENT", "0", 1);
    std::size_t completed = 0;
    const std::size_t total = keyLengths(maximum_key_len).size() * 26U;
    for (int layer = 2; layer < 28; ++layer) {
        CPUPatternSparseAttentionFunc op(
            backend, "offline.heads", 1, 0.8F, true, 0.5F, 0, 0,
            false, 1 + layer * 4099, kCapacity, true,
            retentions[static_cast<std::size_t>(layer)], layer);
        for (const int key_len : keyLengths(maximum_key_len)) {
            auto query = makeTensor(
                backend, BSHD, MLLM_TYPE_F32, kHeads, gQueryLen, kHeadDim);
            auto key = makeTensor(
                backend, BSHD, MLLM_TYPE_F16, kHeads, key_len, kHeadDim);
            auto value = makeTensor(
                backend, BHDS, MLLM_TYPE_F16, kHeads, key_len, kHeadDim);
            auto output = std::make_shared<Tensor>(backend);
            fillInputs(query, key, value,
                       std::vector<ScalePair>(kHeads, scale));
            executeSamples(op, query, key, value, output, raw_output,
                           "heads", warmup, repetitions);
            ++completed;
            if (completed % 16U == 0U || completed == total) {
                std::cerr << "PROFILE_HEADS " << completed << '/' << total
                          << std::endl;
            }
        }
    }
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 4 || argc > 8) {
        std::cerr << "usage: " << argv[0]
                  << " MANIFEST HEAD_PROFILE RAW_OUTPUT"
                     " [MAX_KEY_LEN] [WARMUP] [REPETITIONS] [QUERY_LEN]\n";
        return 2;
    }
    try {
        const std::string manifest = argv[1];
        const std::string head_profile = argv[2];
        const std::string raw_output = argv[3];
        const int maximum_key_len = argc >= 5 ? std::atoi(argv[4]) : kCapacity;
        const int warmup = argc >= 6 ? std::atoi(argv[5]) : 3;
        const int repetitions = argc >= 7 ? std::atoi(argv[6]) : 20;
        gQueryLen = argc >= 8 ? std::atoi(argv[7]) : 512;
        if (gQueryLen <= 0 || maximum_key_len < gQueryLen
            || maximum_key_len > kCapacity
            || (maximum_key_len != kCapacity
                && maximum_key_len % gQueryLen != 0)
            || warmup < 0 || repetitions <= 0) {
            throw std::invalid_argument("invalid profiling arguments");
        }
        const auto manifest_profile = loadManifestProfile(manifest);
        const auto &buckets = manifest_profile.scales;
        const auto retentions = loadRetentions(head_profile);
        const char *stages_value = std::getenv(
            "MLLM_HMX_PIPELINE_PROFILE_STAGES");
        const std::string stages = stages_value == nullptr
            ? "all" : std::string(stages_value);
        if (stages != "all" && stages != "npu" && stages != "heads") {
            throw std::invalid_argument(
                "MLLM_HMX_PIPELINE_PROFILE_STAGES must be all, npu, or heads");
        }
        std::ofstream(raw_output, std::ios::trunc).close();
        setenv("MLLM_HMX_PIPELINE_GROUP_HEADS", "1", 1);
        setenv("MLLM_HMX_PIPELINE_SCHEDULE", "fifo", 1);
        // Keep the paper's cooperative two-core mode as the default, but
        // allow a resource-matched calibration run to select the same Top-k
        // execution mode as the benchmark under test.  Overwriting an
        // explicit setting here silently produced profiles for the wrong CPU
        // topology.
        const char *topk_mode = std::getenv(
            "MLLM_HMX_PIPELINE_TOPK_MODE");
        if (topk_mode == nullptr || topk_mode[0] == '\0') {
            setenv("MLLM_HMX_PIPELINE_TOPK_MODE", "cooperative", 1);
        }
        setenv("MLLM_HMX_INT8_TOPK_OVERSAMPLE", "1", 1);
        setenv("MLLM_HMX_INT8_OUTLIER_FALLBACK", "0", 1);
        unsetenv("MLLM_HMX_INT8_DYNAMIC_QK_SCALE");
        unsetenv("MLLM_HMX_INT8_DYNAMIC_OUTPUT_SCALE");
        unsetenv("MLLM_HMX_INT8_DSP_INT32_TOPK");
        std::shared_ptr<MemoryManager> memory =
            std::make_shared<SystemMemoryManager>();
        CPUBackend backend(memory);
        if (stages != "heads") {
            profileNpu(&backend, buckets, manifest_profile.head_counts,
                       raw_output, maximum_key_len, warmup, repetitions);
        }
        if (stages != "npu") {
            profileHeads(&backend, retentions, buckets[buckets.size() / 2U],
                         raw_output, maximum_key_len, warmup, repetitions);
        }
        std::cout << "PROFILE_COMPLETE output=" << raw_output
                  << " stages=" << stages
                  << " query_len=" << gQueryLen
                  << " max_key_len=" << maximum_key_len
                  << " warmup=" << warmup
                  << " repetitions=" << repetitions << '\n';
        return 0;
    } catch (const std::exception &exception) {
        std::cerr << "profile_hmx_pipeline_latency: " << exception.what()
                  << '\n';
        return 1;
    }
}
