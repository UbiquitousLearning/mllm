#ifndef MLLM_CPU_ATTENTION_PROFILER_HPP
#define MLLM_CPU_ATTENTION_PROFILER_HPP

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>

namespace mllm {

enum class AttentionProfileStage : size_t {
    QK = 0,
    DENSE_SOFTMAX,
    DENSE_PV,
    SPARSE_TOTAL,
    SPARSE_PACK,
    SPARSE_SELECT_WORK,
    SPARSE_PV_WORK,
    PATTERN_QK_SOFTMAX_WORK,
    PATTERN_QK_WORK,
    PATTERN_SOFTMAX_WORK,
    PATTERN_NORMALIZE_PREP_WORK,
    ATTENTION_OUTER,
    HMX_SELECT,
    HMX_NPU_PRODUCE_WALL,
    HMX_TOPK_STAGE_WALL,
    HMX_SPARSE_STAGE_WALL,
    HMX_MUTEX_WAIT,
    HMX_PREPARE_Q,
    HMX_K_LAYOUT,
    HMX_K_CACHE_LOAD,
    HMX_K_PREPARE,
    HMX_FUSED_PREPARE_EXECUTE,
    HMX_K_CACHE_STORE,
    HMX_SESSION_INIT,
    HMX_SCOPE_BEGIN,
    HMX_SCOPE_END,
    HMX_BUFFER_ALLOC,
    HMX_Q_COPY,
    HMX_W_COPY,
    HMX_Q_LAYOUT,
    HMX_SCALE_PROFILE,
    HMX_MM,
    HMX_OUTPUT_LAYOUT,
    HMX_TOPK,
    HMX_BUFFER_FREE,
    HMX_SESSION_FINALIZE,
    COUNT,
};

struct AttentionProfileSnapshot {
    std::array<uint64_t, static_cast<size_t>(AttentionProfileStage::COUNT)> ns{};
    std::array<uint64_t, static_cast<size_t>(AttentionProfileStage::COUNT)> calls{};

    double milliseconds(AttentionProfileStage stage) const {
        return static_cast<double>(ns[static_cast<size_t>(stage)]) / 1.0e6;
    }

    uint64_t callCount(AttentionProfileStage stage) const {
        return calls[static_cast<size_t>(stage)];
    }
};

struct SparseSelectionSnapshot {
    uint64_t eligible = 0;
    uint64_t retained = 0;
    uint64_t sampled_rows = 0;
    uint64_t fallback_rows = 0;
    uint64_t candidate_elements = 0;
};

class CPUSparseSelectionStats {
public:
    static void add(uint64_t eligible, uint64_t retained,
                    uint64_t sampled_rows = 0,
                    uint64_t fallback_rows = 0,
                    uint64_t candidate_elements = 0) {
        eligible_.fetch_add(eligible, std::memory_order_relaxed);
        retained_.fetch_add(retained, std::memory_order_relaxed);
        sampled_rows_.fetch_add(sampled_rows, std::memory_order_relaxed);
        fallback_rows_.fetch_add(fallback_rows, std::memory_order_relaxed);
        candidate_elements_.fetch_add(candidate_elements,
                                      std::memory_order_relaxed);
    }

    static void reset() {
        eligible_.store(0, std::memory_order_relaxed);
        retained_.store(0, std::memory_order_relaxed);
        sampled_rows_.store(0, std::memory_order_relaxed);
        fallback_rows_.store(0, std::memory_order_relaxed);
        candidate_elements_.store(0, std::memory_order_relaxed);
    }

    static SparseSelectionSnapshot snapshot() {
        return {
            eligible_.load(std::memory_order_relaxed),
            retained_.load(std::memory_order_relaxed),
            sampled_rows_.load(std::memory_order_relaxed),
            fallback_rows_.load(std::memory_order_relaxed),
            candidate_elements_.load(std::memory_order_relaxed),
        };
    }

private:
    inline static std::atomic<uint64_t> eligible_{0};
    inline static std::atomic<uint64_t> retained_{0};
    inline static std::atomic<uint64_t> sampled_rows_{0};
    inline static std::atomic<uint64_t> fallback_rows_{0};
    inline static std::atomic<uint64_t> candidate_elements_{0};
};

class CPUAttentionProfiler {
public:
    static bool enabled() {
        static const bool value = [] {
            const char *env = std::getenv("MLLM_ATTENTION_PROFILE");
            return env != nullptr && env[0] != '\0'
                && !(env[0] == '0' && env[1] == '\0');
        }();
        return value;
    }

    // Per-row sparse kernel timers are useful for attribution, but a long
    // prefill invokes the clock several times for every query/head row. Coarse
    // stage profiling therefore stays production-like by default; opt in to
    // the detailed breakdown explicitly.
    static bool detailedEnabled() {
        const char *env = std::getenv("MLLM_ATTENTION_PROFILE_DETAILED");
        if (env == nullptr || env[0] == '\0') return false;
        return !(env[0] == '0' && env[1] == '\0');
    }

    static uint64_t nowNs() {
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
    }

    static void add(AttentionProfileStage stage, uint64_t ns,
                    uint64_t calls = 1) {
        const size_t index = static_cast<size_t>(stage);
        elapsed_ns_[index].fetch_add(ns, std::memory_order_relaxed);
        call_counts_[index].fetch_add(calls, std::memory_order_relaxed);
    }

    static void reset() {
        for (auto &value : elapsed_ns_) value.store(0, std::memory_order_relaxed);
        for (auto &value : call_counts_) value.store(0, std::memory_order_relaxed);
    }

    static AttentionProfileSnapshot snapshot() {
        AttentionProfileSnapshot result;
        for (size_t i = 0; i < result.ns.size(); ++i) {
            result.ns[i] = elapsed_ns_[i].load(std::memory_order_relaxed);
            result.calls[i] = call_counts_[i].load(std::memory_order_relaxed);
        }
        return result;
    }

private:
    inline static std::array<std::atomic<uint64_t>,
                             static_cast<size_t>(AttentionProfileStage::COUNT)>
        elapsed_ns_{};
    inline static std::array<std::atomic<uint64_t>,
                             static_cast<size_t>(AttentionProfileStage::COUNT)>
        call_counts_{};
};

class ScopedAttentionProfile {
public:
    ScopedAttentionProfile(AttentionProfileStage stage, bool active) :
        stage_(stage), active_(active),
        start_ns_(active ? CPUAttentionProfiler::nowNs() : 0) {}

    ~ScopedAttentionProfile() {
        if (active_) {
            CPUAttentionProfiler::add(
                stage_, CPUAttentionProfiler::nowNs() - start_ns_);
        }
    }

    ScopedAttentionProfile(const ScopedAttentionProfile &) = delete;
    ScopedAttentionProfile &operator=(const ScopedAttentionProfile &) = delete;

private:
    AttentionProfileStage stage_;
    bool active_;
    uint64_t start_ns_;
};

} // namespace mllm

#endif // MLLM_CPU_ATTENTION_PROFILER_HPP
