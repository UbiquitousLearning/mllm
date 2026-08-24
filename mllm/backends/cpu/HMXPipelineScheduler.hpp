#ifndef MLLM_CPU_HMX_PIPELINE_SCHEDULER_HPP
#define MLLM_CPU_HMX_PIPELINE_SCHEDULER_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace mllm::hmx_pipeline {

struct ProfileHeadCost {
    double p50_us = 0.0;
    double p95_us = 0.0;
    float retention = 0.0F;
};

struct ProfileNpuCost {
    double p50_us = 0.0;
    double p95_us = 0.0;
};

class LatencyProfile final {
public:
    static LatencyProfile load(const std::string &path) {
        std::ifstream input(path);
        if (!input) {
            throw std::invalid_argument(
                "failed to open HMX pipeline latency profile: " + path);
        }
        LatencyProfile profile;
        std::string line;
        std::size_t line_number = 0;
        while (std::getline(input, line)) {
            ++line_number;
            if (line.empty()) continue;
            if (line[0] == '#') {
                const auto fields = splitTabs(line.substr(1));
                if (fields.size() != 2 || fields[0].empty()) {
                    throw rowError(path, line_number, "invalid metadata");
                }
                if (!profile.metadata_.emplace(fields[0], fields[1]).second) {
                    throw rowError(path, line_number,
                                   "duplicate metadata key " + fields[0]);
                }
                continue;
            }
            const auto fields = splitTabs(line);
            if (fields.empty() || fields[0] == "stage") continue;
            try {
                if (fields[0] == "npu") {
                    if (fields.size() != 7) {
                        throw std::invalid_argument(
                            "npu row requires 7 fields");
                    }
                    NpuKey key{std::stoi(fields[1]), std::stoi(fields[2]),
                               std::stof(fields[3]), std::stof(fields[4])};
                    ProfileNpuCost cost{std::stod(fields[5]),
                                        std::stod(fields[6])};
                    validateNpu(key, cost);
                    if (!profile.npu_.emplace(key, cost).second) {
                        throw std::invalid_argument("duplicate npu row");
                    }
                } else if (fields[0] == "topk"
                           || fields[0] == "sparse") {
                    if (fields.size() != 7) {
                        throw std::invalid_argument(
                            fields[0] + " row requires 7 fields");
                    }
                    HeadKey key{std::stoi(fields[1]), std::stoi(fields[2]),
                                std::stoi(fields[3])};
                    ProfileHeadCost cost{std::stod(fields[5]),
                                         std::stod(fields[6]),
                                         std::stof(fields[4])};
                    validateHead(key, cost);
                    auto &table = fields[0] == "topk"
                        ? profile.topk_ : profile.sparse_;
                    if (!table.emplace(key, cost).second) {
                        throw std::invalid_argument(
                            "duplicate " + fields[0] + " row");
                    }
                } else {
                    throw std::invalid_argument(
                        "unknown stage " + fields[0]);
                }
            } catch (const std::exception &exception) {
                throw rowError(path, line_number, exception.what());
            }
        }
        profile.validateMetadata(path);
        if (profile.npu_.empty() || profile.topk_.empty()
            || profile.sparse_.empty()) {
            throw std::invalid_argument(
                "HMX pipeline latency profile is incomplete: " + path);
        }
        return profile;
    }

    const std::string &metadata(const std::string &key) const {
        const auto found = metadata_.find(key);
        if (found == metadata_.end()) {
            throw std::invalid_argument(
                "HMX pipeline profile is missing metadata " + key);
        }
        return found->second;
    }

    int queryLen() const { return std::stoi(metadata("query_len")); }
    int headDim() const { return std::stoi(metadata("head_dim")); }
    int maxKeyLen() const { return std::stoi(metadata("max_key_len")); }

    void validateRuntimeBinding(const std::string &model,
                                const std::string &device, int query_len,
                                int head_dim, int key_len,
                                const std::string &main_cpu,
                                const std::string &topk_cpus,
                                const std::string &sparse_cpus) const {
        if (metadata("model") != model) {
            throw std::invalid_argument(
                "HMX pipeline latency profile model mismatch");
        }
        if (metadata("device") != device) {
            throw std::invalid_argument(
                "HMX pipeline latency profile device mismatch");
        }
        if (queryLen() != query_len || headDim() != head_dim
            || key_len <= 0 || key_len > maxKeyLen()) {
            throw std::invalid_argument(
                "HMX pipeline latency profile shape mismatch");
        }
        if (metadata("topk_cpus") != topk_cpus) {
            throw std::invalid_argument(
                "HMX pipeline latency profile Top-k CPU mismatch");
        }
        if (metadata("sparse_cpus") != sparse_cpus) {
            throw std::invalid_argument(
                "HMX pipeline latency profile sparse CPU mismatch");
        }
        if (metadata("main_cpu") != main_cpu) {
            throw std::invalid_argument(
                "HMX pipeline latency profile main CPU mismatch");
        }
    }

    ProfileNpuCost npu(int key_len, int heads, float q_scale,
                       float k_scale) const {
        const auto exact = npu_.find({key_len, heads, q_scale, k_scale});
        if (exact != npu_.end()) return exact->second;
        for (const auto &entry : npu_) {
            if (entry.first.key_len == key_len
                && entry.first.heads == heads
                && nearlyEqual(entry.first.q_scale, q_scale)
                && nearlyEqual(entry.first.k_scale, k_scale)) {
                return entry.second;
            }
        }
        throw std::invalid_argument(
            "HMX pipeline profile has no NPU row for N="
            + std::to_string(key_len) + " H=" + std::to_string(heads));
    }

    ProfileHeadCost topk(int key_len, int layer, int head,
                         float retention) const {
        return headCost(topk_, "Top-k", key_len, layer, head, retention);
    }

    ProfileHeadCost sparse(int key_len, int layer, int head,
                           float retention) const {
        return headCost(sparse_, "sparse", key_len, layer, head, retention);
    }

private:
    struct NpuKey {
        int key_len = 0;
        int heads = 0;
        float q_scale = 0.0F;
        float k_scale = 0.0F;

        bool operator<(const NpuKey &other) const noexcept {
            return std::tie(key_len, heads, q_scale, k_scale)
                < std::tie(other.key_len, other.heads,
                           other.q_scale, other.k_scale);
        }
    };

    struct HeadKey {
        int key_len = 0;
        int layer = -1;
        int head = -1;

        bool operator<(const HeadKey &other) const noexcept {
            return std::tie(key_len, layer, head)
                < std::tie(other.key_len, other.layer, other.head);
        }
    };

    static std::vector<std::string> splitTabs(const std::string &line) {
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

    static std::invalid_argument rowError(const std::string &path,
                                          std::size_t line,
                                          const std::string &detail) {
        return std::invalid_argument(
            "invalid HMX pipeline profile " + path + ":"
            + std::to_string(line) + ": " + detail);
    }

    static bool nearlyEqual(float left, float right) noexcept {
        const float tolerance = 1.0e-5F
            * std::max({1.0F, std::abs(left), std::abs(right)});
        return std::abs(left - right) <= tolerance;
    }

    static void validateNpu(const NpuKey &key,
                            const ProfileNpuCost &cost) {
        if (key.key_len <= 0 || key.heads <= 0
            || !std::isfinite(key.q_scale) || key.q_scale <= 0.0F
            || !std::isfinite(key.k_scale) || key.k_scale <= 0.0F
            || !std::isfinite(cost.p50_us) || cost.p50_us <= 0.0
            || !std::isfinite(cost.p95_us)
            || cost.p95_us < cost.p50_us) {
            throw std::invalid_argument("invalid npu timing values");
        }
    }

    static void validateHead(const HeadKey &key,
                             const ProfileHeadCost &cost) {
        if (key.key_len <= 0 || key.layer < 0 || key.head < 0
            || !std::isfinite(cost.retention)
            || cost.retention <= 0.0F || cost.retention > 1.0F
            || !std::isfinite(cost.p50_us) || cost.p50_us <= 0.0
            || !std::isfinite(cost.p95_us)
            || cost.p95_us < cost.p50_us) {
            throw std::invalid_argument("invalid head timing values");
        }
    }

    void validateMetadata(const std::string &path) const {
        try {
            if (metadata("schema_version") != "1") {
                throw std::invalid_argument("unsupported schema_version");
            }
            if (metadata("model").empty()
                || metadata("device").empty()
                || metadata("main_cpu").empty()
                || metadata("topk_cpus").empty()
                || metadata("sparse_cpus").empty()
                || metadata("manifest_sha256").empty()
                || metadata("head_profile_sha256").empty()
                || metadata("binary_sha256").empty()
                || metadata("statistic") != "p50"
                || metadata("npu_key_cache_policy") != "incremental"
                || metadata("npu_execution_scope") != "attention"
                || std::stoi(metadata("minimum_samples")) <= 0
                || queryLen() <= 0 || headDim() <= 0 || maxKeyLen() <= 0) {
                throw std::invalid_argument("invalid required metadata");
            }
        } catch (const std::exception &exception) {
            throw std::invalid_argument(
                "invalid HMX pipeline profile metadata " + path + ": "
                + exception.what());
        }
    }

    static ProfileHeadCost headCost(
        const std::map<HeadKey, ProfileHeadCost> &table,
        const char *stage, int key_len, int layer, int head,
        float retention) {
        const auto found = table.find({key_len, layer, head});
        if (found == table.end()) {
            throw std::invalid_argument(
                std::string("HMX pipeline profile has no ") + stage
                + " row for N=" + std::to_string(key_len)
                + " L=" + std::to_string(layer)
                + " H=" + std::to_string(head));
        }
        if (!nearlyEqual(found->second.retention, retention)) {
            throw std::invalid_argument(
                std::string("HMX pipeline profile retention mismatch for ")
                + stage + " L=" + std::to_string(layer)
                + " H=" + std::to_string(head));
        }
        return found->second;
    }

    std::map<std::string, std::string> metadata_;
    std::map<NpuKey, ProfileNpuCost> npu_;
    std::map<HeadKey, ProfileHeadCost> topk_;
    std::map<HeadKey, ProfileHeadCost> sparse_;
};

struct HeadJob {
    std::size_t matrix = 0;
    double topk_us = 0.0;
    double sparse_us = 0.0;
};

struct FusedGroup {
    std::size_t id = 0;
    double npu_us = 0.0;
    std::vector<HeadJob> heads;
};

struct PlannedGroup {
    std::size_t id = 0;
    std::vector<std::size_t> head_order;
};

struct Schedule {
    std::vector<PlannedGroup> groups;
    double npu_finish_us = 0.0;
    double topk_finish_us = 0.0;
    double sparse_finish_us = 0.0;
};

class TwoLevelGreedyScheduler final {
public:
    static Schedule plan(const std::vector<FusedGroup> &groups) {
        validate(groups);
        Schedule result;
        std::set<std::size_t> scheduled;
        while (scheduled.size() < groups.size()) {
            const FusedGroup *selected = nullptr;
            HeadPlan selected_heads;
            double selected_npu = 0.0;
            for (const FusedGroup &group : groups) {
                if (scheduled.count(group.id) != 0) continue;
                const double candidate_npu = result.npu_finish_us
                    + group.npu_us;
                const HeadPlan candidate = planHeads(
                    candidate_npu, result.topk_finish_us,
                    result.sparse_finish_us, group.heads);
                if (selected == nullptr
                    || candidate.sparse_finish_us
                           < selected_heads.sparse_finish_us - kTieEpsilon
                    || (std::abs(candidate.sparse_finish_us
                                 - selected_heads.sparse_finish_us)
                            <= kTieEpsilon
                        && group.id < selected->id)) {
                    selected = &group;
                    selected_heads = candidate;
                    selected_npu = candidate_npu;
                }
            }
            if (selected == nullptr) {
                throw std::logic_error("failed to select HMX fused group");
            }
            scheduled.insert(selected->id);
            result.groups.push_back({selected->id, selected_heads.order});
            result.npu_finish_us = selected_npu;
            result.topk_finish_us = selected_heads.topk_finish_us;
            result.sparse_finish_us = selected_heads.sparse_finish_us;
        }
        return result;
    }

    static Schedule fifo(const std::vector<FusedGroup> &groups) {
        validate(groups);
        Schedule result;
        for (const FusedGroup &group : groups) {
            result.npu_finish_us += group.npu_us;
            PlannedGroup planned;
            planned.id = group.id;
            for (const HeadJob &head : group.heads) {
                planned.head_order.push_back(head.matrix);
                result.topk_finish_us = std::max(
                    result.npu_finish_us, result.topk_finish_us)
                    + head.topk_us;
                result.sparse_finish_us = std::max(
                    result.sparse_finish_us, result.topk_finish_us)
                    + head.sparse_us;
            }
            result.groups.push_back(std::move(planned));
        }
        return result;
    }

    static Schedule planNoWorseThanFifo(
        const std::vector<FusedGroup> &groups,
        bool *used_greedy = nullptr,
        double *candidate_greedy_us = nullptr) {
        const Schedule fifo_schedule = fifo(groups);
        const Schedule greedy_schedule = plan(groups);
        if (candidate_greedy_us != nullptr) {
            *candidate_greedy_us = greedy_schedule.sparse_finish_us;
        }
        const bool select_greedy = greedy_schedule.sparse_finish_us
            <= fifo_schedule.sparse_finish_us + kTieEpsilon;
        if (used_greedy != nullptr) *used_greedy = select_greedy;
        return select_greedy ? greedy_schedule : fifo_schedule;
    }

private:
    static constexpr double kTieEpsilon = 1.0e-9;

    struct HeadPlan {
        std::vector<std::size_t> order;
        double topk_finish_us = 0.0;
        double sparse_finish_us = 0.0;
    };

    static HeadPlan planHeads(double npu_finish_us, double topk_finish_us,
                              double sparse_finish_us,
                              const std::vector<HeadJob> &heads) {
        HeadPlan result;
        result.topk_finish_us = topk_finish_us;
        result.sparse_finish_us = sparse_finish_us;
        std::set<std::size_t> scheduled;
        while (scheduled.size() < heads.size()) {
            const HeadJob *selected = nullptr;
            double selected_topk = 0.0;
            double selected_sparse = 0.0;
            for (const HeadJob &head : heads) {
                if (scheduled.count(head.matrix) != 0) continue;
                const double candidate_topk = std::max(
                    npu_finish_us, result.topk_finish_us) + head.topk_us;
                const double candidate_sparse = std::max(
                    result.sparse_finish_us, candidate_topk)
                    + head.sparse_us;
                if (selected == nullptr
                    || candidate_sparse < selected_sparse - kTieEpsilon
                    || (std::abs(candidate_sparse - selected_sparse)
                            <= kTieEpsilon
                        && head.matrix < selected->matrix)) {
                    selected = &head;
                    selected_topk = candidate_topk;
                    selected_sparse = candidate_sparse;
                }
            }
            if (selected == nullptr) {
                throw std::logic_error("failed to select HMX head");
            }
            scheduled.insert(selected->matrix);
            result.order.push_back(selected->matrix);
            result.topk_finish_us = selected_topk;
            result.sparse_finish_us = selected_sparse;
        }
        return result;
    }

    static void validate(const std::vector<FusedGroup> &groups) {
        if (groups.empty()) {
            throw std::invalid_argument("HMX schedule requires fused groups");
        }
        std::set<std::size_t> group_ids;
        std::set<std::size_t> matrices;
        for (const FusedGroup &group : groups) {
            if (!group_ids.insert(group.id).second || group.heads.empty()
                || !std::isfinite(group.npu_us) || group.npu_us <= 0.0) {
                throw std::invalid_argument("invalid HMX fused group");
            }
            for (const HeadJob &head : group.heads) {
                if (!matrices.insert(head.matrix).second
                    || !std::isfinite(head.topk_us) || head.topk_us <= 0.0
                    || !std::isfinite(head.sparse_us)
                    || head.sparse_us <= 0.0) {
                    throw std::invalid_argument("invalid HMX head job");
                }
            }
        }
    }
};

// The long-lived heterogeneous-scale H12 operator publishes one head at a
// time even though it uses a single FastRPC launch.  There is therefore no
// outer fused-group launch decision on that path: scheduling reduces to a
// three-machine flow shop over NPU head production, cooperative Top-k, and
// sparse QK/softmax/PV.  Keep this separate from TwoLevelGreedyScheduler so
// the legacy grouped cost model cannot accidentally treat all twelve heads as
// becoming ready at the end of the H12 RPC.
struct StreamingHeadJob {
    std::size_t matrix = 0;
    double npu_us = 0.0;
    double topk_us = 0.0;
    double sparse_us = 0.0;
};

struct StreamingHeadSchedule {
    std::vector<std::size_t> order;
    double npu_finish_us = 0.0;
    double topk_finish_us = 0.0;
    double sparse_finish_us = 0.0;
};

class StreamingHeadGreedyScheduler final {
public:
    static StreamingHeadSchedule plan(
        const std::vector<StreamingHeadJob> &heads) {
        validate(heads);
        StreamingHeadSchedule result;
        std::set<std::size_t> scheduled;
        while (scheduled.size() < heads.size()) {
            const StreamingHeadJob *selected = nullptr;
            StreamingHeadSchedule selected_state;
            for (const StreamingHeadJob &head : heads) {
                if (scheduled.count(head.matrix) != 0) continue;
                StreamingHeadSchedule candidate = result;
                append(candidate, head);
                if (selected == nullptr
                    || candidate.sparse_finish_us
                           < selected_state.sparse_finish_us - kTieEpsilon
                    || (std::abs(candidate.sparse_finish_us
                                 - selected_state.sparse_finish_us)
                            <= kTieEpsilon
                        && head.matrix < selected->matrix)) {
                    selected = &head;
                    selected_state = std::move(candidate);
                }
            }
            if (selected == nullptr) {
                throw std::logic_error(
                    "failed to select streaming HMX head");
            }
            scheduled.insert(selected->matrix);
            result = std::move(selected_state);
        }
        return result;
    }

    static StreamingHeadSchedule fifo(
        const std::vector<StreamingHeadJob> &heads) {
        validate(heads);
        StreamingHeadSchedule result;
        for (const StreamingHeadJob &head : heads) append(result, head);
        return result;
    }

    static StreamingHeadSchedule planNoWorseThanFifo(
        const std::vector<StreamingHeadJob> &heads,
        bool *used_greedy = nullptr,
        double *candidate_greedy_us = nullptr) {
        const StreamingHeadSchedule fifo_schedule = fifo(heads);
        const StreamingHeadSchedule greedy_schedule = plan(heads);
        if (candidate_greedy_us != nullptr) {
            *candidate_greedy_us = greedy_schedule.sparse_finish_us;
        }
        const bool select_greedy = greedy_schedule.sparse_finish_us
            <= fifo_schedule.sparse_finish_us + kTieEpsilon;
        if (used_greedy != nullptr) *used_greedy = select_greedy;
        return select_greedy ? greedy_schedule : fifo_schedule;
    }

private:
    static constexpr double kTieEpsilon = 1.0e-9;

    static void append(StreamingHeadSchedule &schedule,
                       const StreamingHeadJob &head) {
        schedule.npu_finish_us += head.npu_us;
        schedule.topk_finish_us = std::max(
            schedule.npu_finish_us, schedule.topk_finish_us)
            + head.topk_us;
        schedule.sparse_finish_us = std::max(
            schedule.topk_finish_us, schedule.sparse_finish_us)
            + head.sparse_us;
        schedule.order.push_back(head.matrix);
    }

    static void validate(const std::vector<StreamingHeadJob> &heads) {
        if (heads.empty()) {
            throw std::invalid_argument(
                "streaming HMX schedule requires heads");
        }
        std::set<std::size_t> matrices;
        for (const StreamingHeadJob &head : heads) {
            if (!matrices.insert(head.matrix).second
                || !std::isfinite(head.npu_us) || head.npu_us <= 0.0
                || !std::isfinite(head.topk_us) || head.topk_us <= 0.0
                || !std::isfinite(head.sparse_us)
                || head.sparse_us <= 0.0) {
                throw std::invalid_argument(
                    "invalid streaming HMX head job");
            }
        }
    }
};

} // namespace mllm::hmx_pipeline

#endif // MLLM_CPU_HMX_PIPELINE_SCHEDULER_HPP
