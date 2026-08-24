#ifndef MLLM_CPU_HMX_INT8_SELECTOR_HPP
#define MLLM_CPU_HMX_INT8_SELECTOR_HPP

#include "AttentionProfiler.hpp"
#include "AttentionPipelineExecutor.hpp"
#include "HMXPipelineScheduler.hpp"
#include "HMXPipelineLatencyRecorder.hpp"
#include "INT8TopK.hpp"
#include "PipelineTaskExecutor.hpp"
#include "third_party/ggml/Quantize.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <future>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#endif

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(__ANDROID__)
#include <sys/system_properties.h>
#endif

#if defined(__ANDROID__)
#include <dlfcn.h>
#endif

namespace mllm {

// Fixed-scale INT8 HMX Q*K^T followed by exact CPU INT8 Top-k. The selected
// operator SO supplies H/M/K/N and all quantization scales as metadata, making
// the Q/K quantizers and the DSP requantization one indivisible configuration.
class HMXInt8Selector final {
    // Score storage is completely overwritten by copyScoreBlock() before it
    // is consumed. std::vector::resize() nevertheless zero-initializes every
    // byte when the logical K length grows, which put large page-fault/clear
    // costs on the first prefill. Keep capacity and logical size separately
    // so reserve() can map the scratch arena during model setup without
    // touching it, and select() can resize it without clearing dead data.
    class UninitializedScoreBuffer final {
    public:
        std::size_t capacity() const noexcept { return capacity_; }
        std::size_t size() const noexcept { return size_; }
        std::int8_t *data() noexcept { return storage_.get(); }
        const std::int8_t *data() const noexcept { return storage_.get(); }
        std::int8_t &operator[](std::size_t index) noexcept {
            return storage_[index];
        }
        const std::int8_t &operator[](std::size_t index) const noexcept {
            return storage_[index];
        }

        void reserve(std::size_t required) {
            if (required <= capacity_) return;
            // Default-initialization of an int8_t array leaves its bytes
            // uninitialized. No old score survives across select() calls.
            storage_.reset(new std::int8_t[required]);
            capacity_ = required;
            size_ = 0;
        }

        void prepare(std::size_t required) {
            if (required <= capacity_) return;
            reserve(required);
            // Fault and map the shared pages during model setup. Otherwise
            // the first copyScoreBlock() merely inherits the page-fault cost
            // that vector::resize() used to expose as buffer allocation.
            std::memset(storage_.get(), 0, required * sizeof(std::int8_t));
        }

        void resize(std::size_t required) {
            reserve(required);
            size_ = required;
        }

    private:
        std::unique_ptr<std::int8_t[]> storage_;
        std::size_t capacity_ = 0;
        std::size_t size_ = 0;
    };

    static UninitializedScoreBuffer &sharedScoreBuffer() {
        // Transformer layers consume HMX scores synchronously and discard
        // them after Top-k. One process-wide arena avoids reserving/touching
        // the same H*M*N scratch independently for every sparse layer.
        static auto *buffer = new UninitializedScoreBuffer();
        return *buffer;
    }

    static std::mutex &sharedScoreMutex() {
        static auto *mutex = new std::mutex();
        return *mutex;
    }

public:
    using Half = std::uint16_t;

    struct Request {
        const float *const *query_f32_heads = nullptr;
        std::size_t query_f32_row_stride = 0;
        const Half *const *key_heads = nullptr;
        int batch = 0;
        int heads = 0;
        int query_len = 0;
        int key_len = 0;
        int head_dim = 0;
        const std::int32_t *row_offsets = nullptr;
        std::size_t row_offsets_head_stride = 0;
        int causal_prefix_tokens = -1;
        std::int32_t *topk_indices = nullptr;
        std::size_t topk_head_stride = 0;
        int thread_count = 1;
        int pipeline_group_heads = 0;
        int layer_id = -1;
        const float *head_retentions = nullptr;
        std::size_t head_retention_count = 0;
        std::string diagnostic_label;
        std::function<void(std::size_t, std::size_t)> topk_group_ready;
        std::function<void(std::size_t, std::size_t, int, int)>
            topk_group_rows_ready;
    };

    explicit HMXInt8Selector(std::size_t reserve_hint_tokens = 0)
        : reserve_hint_tokens_(reserve_hint_tokens),
          scores_(sharedScoreBuffer()) {}

    ~HMXInt8Selector() {
#if defined(__ANDROID__)
        std::lock_guard<std::mutex> lock(processOperatorMutex());
        for (auto &bucket : bucket_operators_) {
            if (bucket.context != nullptr) {
                if (!bucket.context_borrowed) {
                    releaseSharedContext(bucket.context_key, bucket.context);
                }
                bucket.context = nullptr;
                bucket.context_borrowed = false;
            }
        }
        if (context_ != nullptr) {
            releaseSharedContext(context_key_, context_);
            context_ = nullptr;
        }
        /* All destroy callbacks live in operator DSOs. Keep every DSO loaded
         * until every shared context has been released. */
        for (auto &bucket : bucket_operators_) {
            if (bucket.library != nullptr) (void)dlclose(bucket.library);
        }
        if (library_ != nullptr) (void)dlclose(library_);
#endif
    }

    HMXInt8Selector(const HMXInt8Selector &) = delete;
    HMXInt8Selector &operator=(const HMXInt8Selector &) = delete;

    static bool preloadLatencyProfile(std::string &error);

    bool reserve(int batch, int heads, int max_query_len, int max_key_len,
                 int head_dim, std::string *error = nullptr) {
        std::string detail;
        if (!ensureLoaded(detail)) return fail(detail, error);
        if (batch <= 0 || heads <= 0 || max_query_len <= 0
            || max_key_len <= 0 || head_dim <= 0) {
            return fail("INT8 HMX reserve requires positive dimensions",
                        error);
        }
        if (head_dim != info_.k || max_key_len > info_.n) {
            return fail(shapeError(batch * heads, max_query_len, max_key_len,
                                   head_dim), error);
        }
        if (!preloadLatencyProfile(detail)) return fail(detail, error);
        std::size_t matrix_count = 0;
        std::size_t score_rows = 0;
        std::size_t score_capacity = 0;
        if (!checkedProduct(
                static_cast<std::size_t>(batch),
                static_cast<std::size_t>(heads), matrix_count)
            || !checkedProduct(
                matrix_count, static_cast<std::size_t>(max_query_len),
                score_rows)
            || !checkedProduct(
                score_rows, static_cast<std::size_t>(max_key_len),
                score_capacity)) {
            return fail("INT8 HMX reserved scratch size overflows size_t",
                        error);
        }
        try {
            std::lock_guard<std::mutex> scratch_lock(sharedScoreMutex());
            scores_.prepare(score_capacity);
            score_requant_scales_.resize(score_rows);
            score_q_scales_.resize(matrix_count);
            score_k_scales_.resize(matrix_count);
            adaptive_requant_cache_.resize(matrix_count);
            topk_preselected_.resize(matrix_count);
        } catch (...) {
            return fail("failed to reserve INT8 HMX runtime scratch", error);
        }
#if defined(__ANDROID__)
        // Model setup already knows the selected bank and fixed shape. Acquire
        // the process-shared operator context here so rpcmem allocation,
        // FastRPC mapping, arena registration, and DSO callback pinning do not
        // appear in the first attention call.
        {
            std::lock_guard<std::mutex> lock(processOperatorMutex());
            if (context_ == nullptr) {
                const int status = acquireSharedContext(
                    context_key_, operator_path_, create_, destroy_,
                    &context_);
                if (status != 0 || context_ == nullptr) {
                    detail = callError("hmx_i8_create", status);
                    if (status == 0) detail += " (returned null context)";
                    return fail(detail, error);
                }
            }
        }
#endif
        return succeed(error);
    }

    bool select(const Request &request, std::string *error = nullptr) {
        // Scores are process-level scratch. The model's layer dependency is
        // already serial, while this guard preserves correctness for callers
        // that issue independent selectors concurrently.
        std::unique_lock<std::mutex> score_scratch_lock(sharedScoreMutex());
        std::string detail;
        if (!ensureLoaded(detail)) return fail(detail, error);
        ValidatedRequest validated;
        if (!validateRequest(request, validated, detail)) {
            return fail(detail, error);
        }
        if (bucket_operators_.empty()
            && request.pipeline_group_heads != 0
            && request.pipeline_group_heads != info_.heads) {
            return fail(
                "MLLM_HMX_PIPELINE_GROUP_HEADS must be zero or match the "
                "INT8 operator head group H=" + std::to_string(info_.heads),
                error);
        }
        if (!bucket_operators_.empty() && !raw_path_supported_) {
            return fail("INT8 HMX bucket manifest dispatch requires raw-path "
                        "operator symbols", error);
        }

        const bool profile_enabled = CPUAttentionProfiler::enabled();
        score_row_stride_ = request.key_len;
        const std::size_t score_head_elements
            = static_cast<std::size_t>(request.query_len)
            * static_cast<std::size_t>(score_row_stride_);
        std::size_t score_elements = 0;
        if (!checkedProduct(validated.matrix_count, score_head_elements,
                            score_elements)) {
            return fail("INT8 HMX score cache size overflows size_t", error);
        }
        std::size_t score_capacity_elements = score_elements;
        if (reserve_hint_tokens_
            > static_cast<std::size_t>(request.key_len)) {
            std::size_t reserved_head_elements = 0;
            if (!checkedProduct(
                    static_cast<std::size_t>(request.query_len),
                    reserve_hint_tokens_, reserved_head_elements)
                || !checkedProduct(
                    validated.matrix_count, reserved_head_elements,
                    score_capacity_elements)) {
                return fail(
                    "INT8 HMX reserved score cache size overflows size_t",
                    error);
            }
        }
        try {
            ScopedAttentionProfile buffer_profile(
                AttentionProfileStage::HMX_BUFFER_ALLOC,
                profile_enabled);
            if (scores_.capacity() < score_capacity_elements) {
                scores_.reserve(score_capacity_elements);
            }
            scores_.resize(score_elements);
            score_requant_scales_.assign(
                validated.matrix_count
                    * static_cast<std::size_t>(request.query_len),
                0.0F);
            score_q_scales_.assign(validated.matrix_count, 0.0F);
            score_k_scales_.assign(validated.matrix_count, 0.0F);
            if (adaptive_requant_cache_.size()
                != validated.matrix_count) {
                adaptive_requant_cache_.assign(
                    validated.matrix_count, 0.0F);
            }
            topk_preselected_.assign(validated.matrix_count, 0);
            // The fused bucket raw path reads and writes its rpcmem arena
            // directly.  These large host buffers belong only to the legacy
            // non-bucket quantized path; allocating them for every model
            // layer needlessly zeroed hundreds of MB before the timeline.
            if (bucket_operators_.empty()) {
                query_i8_.resize(static_cast<std::size_t>(info_.heads)
                                 * info_.m * info_.k);
                key_i8_.resize(static_cast<std::size_t>(info_.heads)
                               * info_.n * info_.k);
                group_scores_.resize(static_cast<std::size_t>(info_.heads)
                                     * info_.m * info_.n);
            }
        } catch (...) {
            return fail("failed to allocate INT8 HMX selector buffers", error);
        }

        if (request.pipeline_group_heads == 0) {
            std::future<void> deferred_cache_store;
            const bool produced = bucket_operators_.empty()
                ? produceAllGroups(request, validated, profile_enabled,
                                   ReadyGroupCallback(), detail)
                : produceBucketedGroups(request, validated, profile_enabled,
                                        ReadyGroupCallback(), detail,
                                        &deferred_cache_store);
            if (!produced) {
                if (deferred_cache_store.valid()) {
                    try {
                        deferred_cache_store.get();
                    } catch (...) {
                        // Preserve the producer error, but do not return while
                        // a worker may still read the shared operator arena.
                    }
                }
                return fail(detail, error);
            }
            bool topk_ok = true;
            {
                ScopedAttentionProfile topk_profile(
                    AttentionProfileStage::HMX_TOPK, profile_enabled);
                topk_ok = selectTopKRange(
                    request, validated, 0, validated.matrix_count);
            }
            if (deferred_cache_store.valid()) {
                try {
                    deferred_cache_store.get();
                } catch (const std::exception &exception) {
                    return fail(
                        std::string("asynchronous packed-K cache store failed: ")
                            + exception.what(),
                        error);
                } catch (...) {
                    return fail(
                        "asynchronous packed-K cache store failed", error);
                }
            }
            if (!topk_ok) return fail("INT8 exact Top-k failed", error);
            return succeed(error);
        }

        return selectPipelined(request, validated, profile_enabled, error);
    }

    void resetLayerPackedKeyCache() {
        layer_packed_key_cache_.clear();
        layer_key_scale_cache_.clear();
    }

    float qScale() const noexcept { return info_.q_scale; }
    float kScale() const noexcept { return info_.k_scale; }
    float outputScale() const noexcept { return info_.output_scale; }
    int operatorHeads() const noexcept { return info_.heads; }

private:
    static constexpr std::uint32_t kApiVersion = 5;
    static constexpr std::uint32_t kLegacyApiVersion = 1;
    static constexpr std::uint32_t kRequiredDirectFlags =
        (1u << 0) | (1u << 1) | (1u << 2) | (1u << 3) | (1u << 4) |
        (1u << 5) | (1u << 6) | (1u << 7) | (1u << 8) | (1u << 9) |
        (1u << 10) | (1u << 11);
    static constexpr std::uint32_t kLongRpcHeadReadyFlag = 1u << 16;
    static constexpr std::int32_t kLongRpcOutputReady = -2;
    static constexpr std::int32_t kLongRpcOutputError = -1;
    static constexpr std::size_t kLongRpcReadyStride = 32;

    struct OperatorInfo {
        std::uint32_t struct_size = sizeof(OperatorInfo);
        std::int32_t heads = 0;
        std::int32_t m = 0;
        std::int32_t k = 0;
        std::int32_t n = 0;
        float q_scale = 0.0F;
        float k_scale = 0.0F;
        float output_scale = 0.0F;
        float requant_scale = 0.0F;
        std::uint32_t flags = 0;
    };

    struct ValidatedRequest {
        std::size_t matrix_count = 0;
        std::size_t topk_head_stride = 0;
        int causal_prefix_tokens = 0;
        int topk_threads = 1;
    };

    struct ReadyGroup {
        std::size_t begin = 0;
        std::size_t count = 0;
    };

    // View of one in-flight all-head long RPC.  The DSP publishes each
    // ready word only after flushing the corresponding score matrix, so a
    // CPU consumer can select Top-k directly from rpcmem after an acquire
    // load without copying through scores_.  The view is valid only for the
    // duration of the callback.
    struct DirectScoreStream {
        volatile std::int32_t *ready = nullptr;
        std::size_t ready_stride = 0;
        const std::int8_t *scores = nullptr;
        std::size_t score_head_stride = 0;
        std::size_t score_row_stride = 0;
        const std::size_t *matrices = nullptr;
        std::size_t count = 0;
    };

    struct BucketExecutionGroup {
        std::size_t id = 0;
        std::size_t bucket_index = 0;
        std::size_t operator_index = 0;
        bool per_head_bucket_scales = false;
        std::vector<std::size_t> matrices;
    };

    using ReadyGroupCallback = std::function<void(ReadyGroup)>;
    using DirectScoreStreamCallback =
        std::function<bool(const DirectScoreStream &, std::string &)>;

    static PipelineTaskExecutor &topKTaskExecutor() {
        // Selectors are owned per attention layer. Keep the pipeline workers
        // process-wide so every layer reuses the same persistent stage pool.
        static auto *executor = new PipelineTaskExecutor(
            "MLLM_HMX_PIPELINE_TOPK_CPU",
            "MLLM_HMX_PIPELINE_TOPK_WORKERS", "Top-k worker", 1);
        return *executor;
    }

    static CooperativeTaskExecutor &cooperativeTopKTaskExecutor() {
        static auto *executor = new CooperativeTaskExecutor(
            "MLLM_HMX_PIPELINE_TOPK_CPU",
            "MLLM_HMX_PIPELINE_TOPK_WORKERS", "cooperative Top-k", 2);
        return *executor;
    }

    static PipelineTaskExecutor &sparseTaskExecutor() {
        static auto *executor = new PipelineTaskExecutor(
            "MLLM_HMX_PIPELINE_SPARSE_CPU",
            "MLLM_HMX_PIPELINE_SPARSE_WORKERS", "sparse worker", 1,
            "MLLM_HMX_PIPELINE_SPARSE_TAIL_CPU",
            "MLLM_HMX_PIPELINE_SPARSE_TAIL_WORKERS");
        return *executor;
    }

    static PipelineTaskExecutor &cpuPackTaskExecutor() {
        static auto *executor = new PipelineTaskExecutor(
            "MLLM_HMX_CPU_PACK_CPU", "MLLM_HMX_CPU_PACK_WORKERS",
            "HMX CPU Q/K scale/pack", 1, nullptr, nullptr,
            "MLLM_HMX_CPU_PACK_SPIN_US");
        return *executor;
    }

    static PipelineTaskExecutor &longRpcTaskExecutor() {
        // A FastRPC call blocks its issuing ARM thread for the complete H12
        // execution. Keep one process-wide dispatcher alive so the model
        // thread can consume per-head score-ready signals in the meantime.
        static auto *executor = new PipelineTaskExecutor(
            "MLLM_HMX_PIPELINE_RPC_CPU", nullptr,
            "long-lived HMX RPC dispatcher", 1);
        return *executor;
    }

    static void ensureCpuPackTaskExecutorReady() {
        static std::once_flag once;
        std::call_once(once, []() {
            cpuPackTaskExecutor().submit([]() {}).get();
        });
    }

    static void ensureLongRpcTaskExecutorReady() {
        static std::once_flag once;
        std::call_once(once, []() {
            longRpcTaskExecutor().submit([]() {}).get();
        });
    }

    struct MatrixScales {
        float q = 0.0F;
        float k = 0.0F;
    };

    struct LayerKeyScaleCache {
        const Half *source = nullptr;
        int tokens = 0;
        float scale = 0.0F;

        void reset(const Half *new_source) noexcept {
            source = new_source;
            tokens = 0;
            scale = 0.0F;
        }
    };

    struct CpuScalePackJob {
        enum Phase : int {
            PROFILING = 0,
            SCALES_READY = 1,
            PACK_READY = 2,
            CANCELLED = 3,
            DONE = 4,
        };

        std::atomic<int> phase{PROFILING};
        bool profile_ok = false;
        bool pack_ok = false;
        std::string profile_error;

        const Request *request = nullptr;
        std::size_t count = 0;
        const std::size_t *matrix_indices = nullptr;
        int query_begin = 0;
        int query_count = 0;
        int key_begin = 0;
        int padded_n = 0;
        const float *q_scales = nullptr;
        const float *k_scales = nullptr;
        std::int8_t *direct_query = nullptr;
        std::int8_t *direct_key = nullptr;
        std::int32_t *direct_sums = nullptr;
        volatile std::int32_t *ready = nullptr;
        std::size_t ready_stride = 1;
        int padded_m = 0;
        int k = 0;
        bool profile_enabled = false;
    };

    struct ScaleProfileStaging {
        void *context = nullptr;
        std::size_t matrix_count = 0;
        int query_begin = -1;
        int query_count = 0;
        int key_begin = -1;
        bool query_staged = false;
        bool key_staged = false;
        std::shared_ptr<CpuScalePackJob> cpu_job;
        std::future<void> cpu_future;

        void cancelCpuJob() noexcept {
            if (cpu_job == nullptr) return;
            int expected = CpuScalePackJob::SCALES_READY;
            (void)cpu_job->phase.compare_exchange_strong(
                expected, CpuScalePackJob::CANCELLED,
                std::memory_order_acq_rel, std::memory_order_acquire);
            if (cpu_future.valid()) {
                try {
                    cpu_future.get();
                } catch (...) {
                }
            }
            cpu_job.reset();
        }

        ~ScaleProfileStaging() { cancelCpuJob(); }
    };

    struct PackedKeyCache {
        const Half *source = nullptr;
        float scale = 0.0F;
        int tokens = 0;
        std::vector<std::int8_t> packed;
        std::vector<std::int32_t> sums;
    };

    struct LayerPackedKeyCache {
        const Half *source = nullptr;
        float scale = 0.0F;
        int tokens = 0;
        std::vector<std::int8_t> packed;
        std::vector<std::int32_t> sums;

        bool valid(const Half *candidate_source, float candidate_scale,
                   int key_len, int head_dim) const noexcept {
            return source == candidate_source && scale == candidate_scale
                && tokens >= 0 && tokens <= key_len && (tokens % 32) == 0
                && packed.size() >= static_cast<std::size_t>(tokens)
                    * static_cast<std::size_t>(head_dim)
                && sums.size() >= static_cast<std::size_t>(tokens);
        }

        void reset(const Half *new_source, float new_scale) {
            source = new_source;
            scale = new_scale;
            tokens = 0;
            packed.clear();
            sums.clear();
        }
    };

    struct LayerPackedKeyPlan {
        int key_begin = 0;
        bool reuse_packed = false;
    };

    using ApiVersionFn = std::uint32_t (*)();
    using GetInfoFn = int (*)(OperatorInfo *);
    using CreateFn = int (*)(void **);
    using DestroyFn = void (*)(void *);
    using MatmulFn = int (*)(void *, const std::int8_t *,
                             const std::int8_t *, std::int8_t *);
    using DataFn = std::int8_t *(*)(void *);
    using Int32DataFn = std::int32_t *(*)(void *);
    using ConstDataFn = const std::int8_t *(*)(void *);
    using ConstInt32DataFn = const std::int32_t *(*)(void *);
    using RawQueryDataFn = float *(*)(void *);
    using RawKeyDataFn = Half *(*)(void *);
    using ProfileRawFn = int (*)(void *, std::int32_t, std::int32_t,
                                 std::int32_t, std::int32_t, std::int32_t,
                                 float *, float *);
    using ProfileRawIncrementalFn = int (*)(
        void *, std::int32_t, std::int32_t, std::int32_t, std::int32_t,
        std::int32_t, const float *, float *, float *);
    struct DspTiming {
        std::uint32_t struct_size = 0;
        std::uint32_t version = 0;
        std::uint64_t total_ticks = 0;
        std::uint64_t resource_begin_ticks = 0;
        std::uint64_t pipeline_ticks = 0;
        std::uint64_t key_pack_work_ticks = 0;
        std::uint64_t query_pack_work_ticks = 0;
        std::uint64_t hmx_kernel_ticks = 0;
        std::uint64_t output_flush_ticks = 0;
        std::uint64_t resource_end_ticks = 0;
    };
    using LastDspTimingFn = int (*)(void *, DspTiming *);
    using ExecuteHnFn = int (*)(void *, std::int32_t, std::int32_t);
    using CpuPackReadyDataFn = volatile std::int32_t *(*)(void *);
    using ExecuteCpuPackedPerHeadHnFn = int (*)(
        void *, std::int32_t, std::int32_t, const float *);
    using PrepareRawKeyFn = int (*)(void *, std::int32_t, std::int32_t,
                                    std::int32_t);
    using PrepareRawKeyScaledFn = int (*)(void *, std::int32_t,
                                          std::int32_t, std::int32_t, float);
    using ExecuteRawHnFn = int (*)(void *, std::int32_t, std::int32_t,
                                   std::int32_t);
    using PrepareExecuteRawHnFn = int (*)(
        void *, std::int32_t, std::int32_t, std::int32_t, std::int32_t);
    using PrepareExecuteRawPerHeadHnFn = int (*)(
        void *, std::int32_t, std::int32_t, std::int32_t, std::int32_t,
        const float *, const float *, float);
    using PrepareExecuteRawPerHeadRequantHnFn = int (*)(
        void *, std::int32_t, std::int32_t, std::int32_t, std::int32_t,
        const float *, const float *, const float *);
    using ExecuteRawScaledHnFn = int (*)(void *, std::int32_t, std::int32_t,
                                         std::int32_t, float);
    using ExecuteRawDynamicHnFn = int (*)(void *, std::int32_t, std::int32_t,
                                          std::int32_t, float, float);
    using ExecuteRawI32TopKHnFn = int (*)(
        void *, std::int32_t, std::int32_t, std::int32_t, std::int32_t,
        float, const std::int32_t *, std::int32_t);
    using BeginFn = int (*)(void *, std::int32_t);
    using EndFn = int (*)(void *);
    using StatusStringFn = const char *(*)(int);

    struct AttentionExecutionScope {
        void *context = nullptr;
        EndFn end = nullptr;
        bool active = false;

        ~AttentionExecutionScope() {
            if (active && end != nullptr) (void)end(context);
        }

        void arm(void *new_context, EndFn new_end) noexcept {
            context = new_context;
            end = new_end;
            active = true;
        }

        int close() noexcept {
            if (!active || end == nullptr) return 0;
            active = false;
            return end(context);
        }
    };

    struct BucketOperator {
        std::string path;
        std::string context_key;
        void *library = nullptr;
        OperatorInfo info{};
        CreateFn create = nullptr;
        DestroyFn destroy = nullptr;
        MatmulFn matmul = nullptr;
        DataFn query_data = nullptr;
        DataFn key_data = nullptr;
        Int32DataFn key_sums_data = nullptr;
        ConstDataFn scores_data = nullptr;
        ConstInt32DataFn topk_indices_data = nullptr;
        RawQueryDataFn raw_query_data = nullptr;
        RawKeyDataFn raw_key_data = nullptr;
        ProfileRawFn profile_raw = nullptr;
        ExecuteHnFn execute_hn = nullptr;
        CpuPackReadyDataFn cpu_pack_ready_data = nullptr;
        ExecuteCpuPackedPerHeadHnFn execute_cpu_packed_per_head_hn = nullptr;
        PrepareRawKeyFn prepare_raw_key = nullptr;
        PrepareRawKeyScaledFn prepare_raw_key_scaled = nullptr;
        ExecuteRawHnFn execute_raw_hn = nullptr;
        PrepareExecuteRawHnFn prepare_execute_raw_hn = nullptr;
        PrepareExecuteRawPerHeadHnFn prepare_execute_raw_per_head_hn = nullptr;
        PrepareExecuteRawPerHeadRequantHnFn
            prepare_execute_raw_per_head_requant_hn = nullptr;
        ExecuteRawScaledHnFn execute_raw_scaled_hn = nullptr;
        ExecuteRawDynamicHnFn execute_raw_dynamic_hn = nullptr;
        ExecuteRawI32TopKHnFn execute_raw_i32_topk_hn = nullptr;
        BeginFn begin = nullptr;
        EndFn end = nullptr;
        StatusStringFn status_string = nullptr;
        void *context = nullptr;
        bool context_borrowed = false;
    };

    struct SharedContextEntry {
        void *context = nullptr;
        DestroyFn destroy = nullptr;
        void *callback_library = nullptr;
        std::size_t users = 0;
        std::vector<const void *> raw_key_sources;
        int raw_key_tokens = 0;
        int raw_key_padded_n = 0;
    };

    static std::map<std::string, SharedContextEntry> &sharedContexts() {
        static auto *contexts =
            new std::map<std::string, SharedContextEntry>();
        return *contexts;
    }

    static int acquireSharedContext(const std::string &key,
                                    const std::string &owner_path,
                                    CreateFn create, DestroyFn destroy,
                                    void **context) {
        if (context == nullptr || create == nullptr || destroy == nullptr) {
            return -1;
        }
        auto &contexts = sharedContexts();
        auto found = contexts.find(key);
        if (found != contexts.end()) {
            ++found->second.users;
            *context = found->second.context;
            return 0;
        }
        /* The destroy callback belongs to the operator DSO that first creates
         * this process-wide context. Pin that DSO independently of selector
         * lifetimes so the final user can always call the callback safely. */
        void *callback_library = nullptr;
#if defined(__ANDROID__)
        callback_library = dlopen(
            owner_path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (callback_library == nullptr) return -1;
#else
        (void)owner_path;
#endif
        void *created = nullptr;
        const int status = create(&created);
        if (status != 0 || created == nullptr) {
#if defined(__ANDROID__)
            (void)dlclose(callback_library);
#endif
            return status == 0 ? -1 : status;
        }
        contexts.emplace(
            key, SharedContextEntry{created, destroy, callback_library, 1});
        *context = created;
        return 0;
    }

    static void releaseSharedContext(const std::string &path, void *context) {
        auto &contexts = sharedContexts();
        auto found = contexts.find(path);
        if (found == contexts.end() || found->second.context != context) return;
        if (--found->second.users == 0) {
            found->second.destroy(found->second.context);
#if defined(__ANDROID__)
            if (found->second.callback_library != nullptr) {
                (void)dlclose(found->second.callback_library);
            }
#endif
            contexts.erase(found);
        }
    }

    struct RawKeyPlan {
        int key_begin = 0;
        bool reuse_packed = false;
    };

    static RawKeyPlan rawKeyPlan(
        const std::string &context_key, const Request &request,
        std::size_t begin, std::size_t count,
        const std::size_t *matrix_indices, int padded_n) {
        RawKeyPlan plan;
        auto found = sharedContexts().find(context_key);
        if (found == sharedContexts().end()) return plan;
        const SharedContextEntry &entry = found->second;
        if (entry.raw_key_padded_n != padded_n ||
            request.key_len < entry.raw_key_tokens ||
            entry.raw_key_sources.size() != count) {
            return plan;
        }
        for (std::size_t local = 0; local < count; ++local) {
            const std::size_t matrix = matrix_indices == nullptr
                ? begin + local : matrix_indices[local];
            if (entry.raw_key_sources[local] != request.key_heads[matrix]) {
                return plan;
            }
        }
        plan.key_begin = entry.raw_key_tokens;
        plan.reuse_packed = request.key_len == entry.raw_key_tokens;
        return plan;
    }

    static void commitRawKey(
        const std::string &context_key, const Request &request,
        std::size_t begin, std::size_t count,
        const std::size_t *matrix_indices, int padded_n) {
        auto found = sharedContexts().find(context_key);
        if (found == sharedContexts().end()) return;
        SharedContextEntry &entry = found->second;
        entry.raw_key_sources.resize(count);
        for (std::size_t local = 0; local < count; ++local) {
            const std::size_t matrix = matrix_indices == nullptr
                ? begin + local : matrix_indices[local];
            entry.raw_key_sources[local] = request.key_heads[matrix];
        }
        entry.raw_key_tokens = request.key_len;
        entry.raw_key_padded_n = padded_n;
    }

    static void invalidateRawKey(const std::string &context_key) {
        auto found = sharedContexts().find(context_key);
        if (found == sharedContexts().end()) return;
        found->second.raw_key_sources.clear();
        found->second.raw_key_tokens = 0;
        found->second.raw_key_padded_n = 0;
    }

    LayerPackedKeyPlan loadLayerPackedKeyGroup(
        const Request &request, std::size_t count,
        const std::size_t *matrix_indices, float k_scale, int actual_n,
        std::int8_t *direct_key, std::int32_t *direct_sums,
        bool profile_enabled, const float *per_head_k_scales = nullptr) {
        LayerPackedKeyPlan plan;
        if (!layerPackedKeyCacheEnabled() || matrix_indices == nullptr
            || direct_key == nullptr || direct_sums == nullptr) {
            return plan;
        }
        if (layer_packed_key_cache_.size()
            < static_cast<std::size_t>(request.batch * request.heads)) {
            layer_packed_key_cache_.resize(
                static_cast<std::size_t>(request.batch * request.heads));
        }
        const int causal_prefix = request.causal_prefix_tokens < 0
            ? request.key_len - request.query_len
            : request.causal_prefix_tokens;
        // A zero-prefix attention call starts a new KV sequence.  Its storage
        // may reuse the same address as a previous prompt, so pointer/length
        // equality alone cannot prove that cached packed keys are current.
        if (causal_prefix <= 0) return plan;

        int common_tokens = -1;
        for (std::size_t local = 0; local < count; ++local) {
            const std::size_t matrix = matrix_indices[local];
            LayerPackedKeyCache &cache = layer_packed_key_cache_[matrix];
            const float matrix_k_scale = per_head_k_scales == nullptr
                ? k_scale : per_head_k_scales[local];
            if (!cache.valid(request.key_heads[matrix], matrix_k_scale,
                             request.key_len, request.head_dim)) {
                common_tokens = 0;
                break;
            }
            if (common_tokens < 0) {
                common_tokens = cache.tokens;
            } else if (common_tokens != cache.tokens) {
                common_tokens = 0;
                break;
            }
        }
        plan.key_begin = std::max(0, common_tokens);
        if (plan.key_begin == 0) return plan;

        const std::size_t direct_head_stride
            = static_cast<std::size_t>(actual_n) * request.head_dim;
        const std::size_t packed_prefix
            = static_cast<std::size_t>(plan.key_begin) * request.head_dim;
        {
            ScopedAttentionProfile profile(
                AttentionProfileStage::HMX_K_CACHE_LOAD, profile_enabled);
            for (std::size_t local = 0; local < count; ++local) {
                const LayerPackedKeyCache &cache =
                    layer_packed_key_cache_[matrix_indices[local]];
                std::memcpy(direct_key + local * direct_head_stride,
                            cache.packed.data(), packed_prefix);
                std::memcpy(direct_sums
                                + local * static_cast<std::size_t>(actual_n),
                            cache.sums.data(),
                            static_cast<std::size_t>(plan.key_begin)
                                * sizeof(std::int32_t));
            }
        }
        plan.reuse_packed = plan.key_begin == request.key_len;
        return plan;
    }

    void commitLayerPackedKeyGroup(
        const Request &request, std::size_t count,
        const std::size_t *matrix_indices, float k_scale, int actual_n,
        int key_begin, const std::int8_t *direct_key,
        const std::int32_t *direct_sums, bool profile_enabled,
        const float *per_head_k_scales = nullptr) {
        if (!layerPackedKeyCacheEnabled() || matrix_indices == nullptr
            || direct_key == nullptr || direct_sums == nullptr) {
            return;
        }
        if (layer_packed_key_cache_.size()
            < static_cast<std::size_t>(request.batch * request.heads)) {
            layer_packed_key_cache_.resize(
                static_cast<std::size_t>(request.batch * request.heads));
        }
        const std::size_t direct_head_stride
            = static_cast<std::size_t>(actual_n) * request.head_dim;
        const std::size_t packed_begin
            = static_cast<std::size_t>(key_begin) * request.head_dim;
        {
            ScopedAttentionProfile profile(
                AttentionProfileStage::HMX_K_CACHE_STORE, profile_enabled);
            for (std::size_t local = 0; local < count; ++local) {
                const std::size_t matrix = matrix_indices[local];
                const float matrix_k_scale = per_head_k_scales == nullptr
                    ? k_scale : per_head_k_scales[local];
                LayerPackedKeyCache &cache = layer_packed_key_cache_[matrix];
                if (key_begin == 0
                    || cache.source != request.key_heads[matrix]
                    || cache.scale != matrix_k_scale) {
                    cache.reset(request.key_heads[matrix], matrix_k_scale);
                }
                cache.packed.resize(direct_head_stride);
                cache.sums.resize(static_cast<std::size_t>(actual_n));
                std::memcpy(cache.packed.data() + packed_begin,
                            direct_key + local * direct_head_stride
                                + packed_begin,
                            direct_head_stride - packed_begin);
                std::memcpy(cache.sums.data() + key_begin,
                            direct_sums
                                + local * static_cast<std::size_t>(actual_n)
                                + key_begin,
                            static_cast<std::size_t>(actual_n - key_begin)
                                * sizeof(std::int32_t));
                cache.source = request.key_heads[matrix];
                cache.scale = matrix_k_scale;
                cache.tokens = request.key_len;
            }
        }
    }

    static std::mutex &processOperatorMutex() {
        static std::mutex *mutex = new std::mutex();
        return *mutex;
    }

    static bool layerPackedKeyCacheEnabled() noexcept {
        const char *value = std::getenv("MLLM_HMX_INT8_LAYER_K_CACHE");
        return value == nullptr || value[0] == '\0'
            || std::strcmp(value, "0") != 0;
    }

    static bool asyncLayerPackedKeyStoreEnabled() noexcept {
        const char *value = std::getenv(
            "MLLM_HMX_INT8_ASYNC_K_CACHE_STORE");
        return value == nullptr || value[0] == '\0'
            || std::strcmp(value, "0") != 0;
    }

    static bool attentionExecutionScopeEnabled() noexcept {
        const char *value = std::getenv(
            "MLLM_HMX_INT8_ATTENTION_EXECUTION_SCOPE");
        return value == nullptr || value[0] == '\0'
            || std::strcmp(value, "0") != 0;
    }

    bool ensureLoaded(std::string &error) {
        if (ready_) return true;
        if (load_attempted_) {
            error = load_error_;
            return false;
        }
        load_attempted_ = true;
#if !defined(__ANDROID__)
        load_error_ = "INT8 HMX selector requires Android";
        error = load_error_;
        return false;
#else
        const char *path = std::getenv("MLLM_HMX_INT8_OPERATOR_LIBRARY");
        if (path == nullptr || path[0] == '\0') {
            path = std::getenv("MLLM_HMX_OPERATOR_LIBRARY");
        }
        if (path == nullptr || path[0] == '\0') {
            path = "libhmx_qk_i8_operator.so";
        }
        operator_path_ = path;
        context_key_ = operator_path_;
        dlerror();
        library_ = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (library_ == nullptr) {
            const char *message = dlerror();
            load_error_ = std::string("dlopen(") + path + ") failed";
            if (message != nullptr) load_error_ += ": " + std::string(message);
            error = load_error_;
            return false;
        }
        if (!loadSymbol("hmx_qk_i8_operator_api_version", api_version_, error)
            || !loadSymbol("hmx_qk_i8_operator_get_info", get_info_, error)
            || !loadSymbol("hmx_i8_create", create_, error)
            || !loadSymbol("hmx_i8_destroy", destroy_, error)
            || !loadSymbol("hmx_qk_i8_operator_matmul", matmul_, error)
            || !loadSymbol("hmx_qk_i8_operator_query_data", query_data_, error)
            || !loadSymbol("hmx_qk_i8_operator_key_data", key_data_, error)
            || !loadSymbol("hmx_qk_i8_operator_scores_data", scores_data_, error)
            || !loadSymbol("hmx_qk_i8_operator_execute_hn", execute_hn_, error)
            || !loadSymbol("hmx_i8_status_string", status_string_, error)) {
            load_error_ = error;
            (void)dlclose(library_);
            library_ = nullptr;
            clearSymbols();
            return false;
        }
        // Legacy packed ABI variants in this repo expose no raw-score path.
        // Detect raw symbols explicitly and only enable them when complete.
        const bool raw_query_loaded = loadOptionalSymbolFrom(
            library_, "hmx_qk_i8_operator_raw_query_data", raw_query_data_);
        const bool raw_key_loaded = loadOptionalSymbolFrom(
            library_, "hmx_qk_i8_operator_raw_key_data", raw_key_data_);
        const bool raw_profile_loaded = loadOptionalSymbolFrom(
            library_, "hmx_qk_i8_operator_profile_raw_scales_hn",
            profile_raw_);
        (void)loadOptionalSymbolFrom(
            library_,
            "hmx_qk_i8_operator_profile_raw_scales_incremental_hn",
            profile_raw_incremental_);
        (void)loadOptionalSymbolFrom(
            library_, "hmx_qk_i8_operator_last_dsp_timing",
            last_dsp_timing_);
        const bool raw_prepare_loaded = loadOptionalSymbolFrom(
            library_, "hmx_qk_i8_operator_prepare_raw_key_hn",
            prepare_raw_key_);
        const bool raw_execute_loaded = loadOptionalSymbolFrom(
            library_, "hmx_qk_i8_operator_execute_raw_hn",
            execute_raw_hn_);
        const bool raw_begin_loaded = loadOptionalSymbolFrom(
            library_, "hmx_qk_i8_operator_begin", begin_);
        const bool raw_end_loaded = loadOptionalSymbolFrom(
            library_, "hmx_qk_i8_operator_end", end_);
        const bool raw_sums_loaded = loadOptionalSymbolFrom(
            library_, "hmx_qk_i8_operator_key_sums_data", key_sums_data_);
        raw_path_supported_ = raw_query_loaded && raw_key_loaded
            && raw_profile_loaded && raw_prepare_loaded && raw_execute_loaded
            && raw_begin_loaded && raw_end_loaded && raw_sums_loaded;
        if (!raw_path_supported_) {
            raw_query_data_ = nullptr;
            raw_key_data_ = nullptr;
            profile_raw_ = nullptr;
            prepare_raw_key_ = nullptr;
            execute_raw_hn_ = nullptr;
            begin_ = nullptr;
            end_ = nullptr;
            key_sums_data_ = nullptr;
        }
        const std::uint32_t selected_api_version = api_version_();
        if (selected_api_version == 0) {
            if (const char *fallback = std::getenv(
                    "MLLM_HMX_INT8_ALLOW_API0")) {
                if (fallback[0] != '\0' && std::strcmp(fallback, "0") != 0) {
                    // Some operator builds currently return 0 due runtime
                    // symbol-thunk differences; allow a targeted override.
                } else {
                    load_error_ = "unsupported INT8 QK operator API version";
                    error = load_error_;
                    return false;
                }
            } else {
                load_error_ = "unsupported INT8 QK operator API version";
                error = load_error_;
                return false;
            }
        }
        info_ = {};
        info_.struct_size = sizeof(info_);
        const int status = get_info_(&info_);
        const float expected_requant =
            info_.q_scale * info_.k_scale / info_.output_scale;
        const float tolerance = 1.0e-5F
            * std::max(1.0F, std::abs(expected_requant));
        if (status != 0 || info_.heads <= 0 || info_.m <= 0 || info_.k <= 0
            || info_.n <= 0 || info_.q_scale <= 0.0F
            || info_.k_scale <= 0.0F || info_.output_scale <= 0.0F
            || !std::isfinite(info_.q_scale)
            || !std::isfinite(info_.k_scale)
            || !std::isfinite(info_.output_scale)
            || !std::isfinite(info_.requant_scale)) {
            load_error_ = status == 0
                ? "INT8 QK operator returned invalid shape/scale metadata"
                : callError("hmx_qk_i8_operator_get_info", status);
            error = load_error_;
            return false;
        }
        const char *manifest = std::getenv(
            "MLLM_HMX_INT8_OPERATOR_MANIFEST");
        if (manifest != nullptr && manifest[0] != '\0'
            && !loadBucketCatalog(manifest, error)) {
            load_error_ = error;
            return false;
        }
        for (const BucketOperator &bucket : bucket_operators_) {
            if (bucket.info.heads == info_.heads
                && bucket.info.m == info_.m && bucket.info.k == info_.k
                && bucket.info.n == info_.n) {
                context_key_ = bucket.context_key;
                break;
            }
        }
        ready_ = true;
        return true;
#endif
    }

    bool validateRequest(const Request &request, ValidatedRequest &validated,
                         std::string &error) const {
        if (request.query_f32_heads == nullptr || request.key_heads == nullptr
            || request.row_offsets == nullptr
            || request.topk_indices == nullptr || request.batch <= 0
            || request.heads <= 0 || request.query_len <= 0
            || request.key_len <= 0 || request.head_dim <= 0
            || request.thread_count <= 0) {
            error = "INT8 HMX selector received an incomplete request";
            return false;
        }
        if (request.head_dim != info_.k || request.key_len > info_.n) {
            error = shapeError(request.batch * request.heads,
                               request.query_len, request.key_len,
                               request.head_dim);
            return false;
        }
        if (!checkedProduct(static_cast<std::size_t>(request.batch),
                            static_cast<std::size_t>(request.heads),
                            validated.matrix_count)) {
            error = "INT8 HMX batch * heads overflows size_t";
            return false;
        }
        const std::size_t query_stride = request.query_f32_row_stride == 0
            ? static_cast<std::size_t>(request.head_dim)
            : request.query_f32_row_stride;
        if (query_stride < static_cast<std::size_t>(request.head_dim)) {
            error = "INT8 HMX FP32 Q row stride is smaller than head_dim";
            return false;
        }
        validated.causal_prefix_tokens = request.causal_prefix_tokens < 0
            ? request.key_len - request.query_len
            : request.causal_prefix_tokens;
        if (validated.causal_prefix_tokens < 0) {
            error = "INT8 HMX causal prefix is negative";
            return false;
        }
        validated.topk_threads = topKThreadCount(request.thread_count);
        if (validated.topk_threads <= 0) {
            error = "MLLM_HMX_TOPK_THREADS must be a positive integer";
            return false;
        }
        const std::int32_t *last_offsets = rowOffsetsForMatrix(
            request, validated.matrix_count - 1);
        if (last_offsets[0] != 0) {
            error = "INT8 HMX row-offset tables must start at zero";
            return false;
        }
        for (std::size_t matrix = 0; matrix < validated.matrix_count;
             ++matrix) {
            const std::int32_t *offsets = rowOffsetsForMatrix(request, matrix);
            if (offsets[0] != 0) {
                error = "INT8 HMX row-offset tables must start at zero";
                return false;
            }
            for (int row = 0; row < request.query_len; ++row) {
                const int valid = std::min(
                    request.key_len,
                    validated.causal_prefix_tokens + row + 1);
                const int keep = offsets[row + 1] - offsets[row];
                if (offsets[row] < 0 || keep <= 0 || keep > valid) {
                    error = "INT8 HMX row offsets request an invalid Top-k";
                    return false;
                }
            }
        }
        validated.topk_head_stride = request.topk_head_stride == 0
            ? static_cast<std::size_t>(last_offsets[request.query_len])
            : request.topk_head_stride;
        for (std::size_t matrix = 0; matrix < validated.matrix_count;
             ++matrix) {
            const std::int32_t *offsets = rowOffsetsForMatrix(request, matrix);
            if (static_cast<std::size_t>(offsets[request.query_len])
                > validated.topk_head_stride) {
                error = "INT8 HMX top-k head stride is too small";
                return false;
            }
        }
        return true;
    }

    bool produceAllGroupsRaw(const Request &request,
                            const ValidatedRequest &validated,
                            bool profile_enabled,
                            const ReadyGroupCallback &on_group_ready,
                            std::string &error) {
        std::unique_lock<std::mutex> operator_lock(processOperatorMutex());
        if (context_ == nullptr) {
            ScopedAttentionProfile init_profile(
                AttentionProfileStage::HMX_SESSION_INIT, profile_enabled);
            const int status = acquireSharedContext(
                context_key_, operator_path_, create_, destroy_, &context_);
            if (status != 0 || context_ == nullptr) {
                error = callError("hmx_i8_create", status);
                if (status == 0) error += " (returned null context)";
                return false;
            }
        }
        void *context = context_;
        bool ok = true;
        std::int8_t *direct_query = query_data_(context);
        std::int8_t *direct_key = key_data_(context);
        std::int32_t *direct_key_sums = key_sums_data_(context);
        const std::int8_t *direct_scores = scores_data_(context);
        float *raw_query = raw_query_data_(context);
        Half *raw_key = raw_key_data_(context);
        if (direct_query == nullptr || direct_key == nullptr
            || direct_key_sums == nullptr || direct_scores == nullptr
            || raw_query == nullptr || raw_key == nullptr) {
            error = "INT8 direct operator returned a null arena pointer";
            return false;
        }
        const int actual_n = (request.key_len + 31) & ~31;
        const std::size_t group_capacity
            = static_cast<std::size_t>(info_.heads);
        for (std::size_t begin = 0; begin < validated.matrix_count
             && ok; begin += group_capacity) {
            const std::size_t count = std::min(
                group_capacity, validated.matrix_count - begin);
            const RawKeyPlan key_plan = rawKeyPlan(
                context_key_, request, begin, count, nullptr, actual_n);
            if (!key_plan.reuse_packed) {
                ok = copyRawKeyGroup(request, begin, count, profile_enabled,
                                     nullptr, raw_key, actual_n,
                                     key_plan.key_begin);
            }
            if (ok && !key_plan.reuse_packed) {
                const int prepare_status = prepare_raw_key_(
                    context, static_cast<std::int32_t>(count), actual_n,
                    key_plan.key_begin);
                if (prepare_status != 0) {
                    error = callError(
                        "hmx_qk_i8_operator_prepare_raw_key_hn",
                        prepare_status);
                    ok = false;
                    invalidateRawKey(context_key_);
                } else {
                    commitRawKey(context_key_, request, begin, count,
                                 nullptr, actual_n);
                }
            }
            int scope_status = ok ? begin_(context, actual_n) : 0;
            const bool scope_active = ok && scope_status == 0;
            if (ok && scope_status != 0) {
                error = callError("hmx_qk_i8_operator_begin", scope_status);
                ok = false;
            }
            for (int query_begin = 0;
                 query_begin < request.query_len && ok;
                 query_begin += info_.m) {
                const int query_count = std::min(
                    info_.m, request.query_len - query_begin);
                ok = copyRawQueryGroup(request, begin, count, query_begin,
                                       query_count, profile_enabled, nullptr,
                                       raw_query);
                if (!ok) break;
                int status = 0;
                {
                    ScopedAttentionProfile mm_profile(
                        AttentionProfileStage::HMX_MM, profile_enabled);
                    status = execute_raw_hn_(
                        context, static_cast<std::int32_t>(count),
                        query_count, actual_n);
                }
                if (status != 0) {
                    error = callError("hmx_qk_i8_operator_matmul", status);
                    ok = false;
                    break;
                }
                ScopedAttentionProfile output_profile(
                    AttentionProfileStage::HMX_OUTPUT_LAYOUT,
                    profile_enabled);
                copyScoreBlock(begin, count, query_begin, query_count,
                               request.query_len, nullptr, direct_scores,
                               actual_n, request.key_len);
            }
            if (scope_active && ok) {
                const int end_status = end_(context);
                if (end_status != 0) {
                    error = callError("hmx_qk_i8_operator_end", end_status);
                    ok = false;
                }
            } else if (scope_active) {
                (void)end_(context);
            }
            if (ok && on_group_ready) {
                on_group_ready({begin, count});
            }
        }
        return ok;
    }

    bool produceAllGroupsPacked(const Request &request,
                               const ValidatedRequest &validated,
                               bool profile_enabled,
                               const ReadyGroupCallback &on_group_ready,
                               std::string &error) {
        std::unique_lock<std::mutex> operator_lock(processOperatorMutex());
        if (context_ == nullptr) {
            ScopedAttentionProfile init_profile(
                AttentionProfileStage::HMX_SESSION_INIT, profile_enabled);
            const int status = acquireSharedContext(
                context_key_, operator_path_, create_, destroy_, &context_);
            if (status != 0 || context_ == nullptr) {
                error = callError("hmx_i8_create", status);
                if (status == 0) error += " (returned null context)";
                return false;
            }
        }
        void *context = context_;
        std::int8_t *direct_query = query_data_(context);
        std::int8_t *direct_key = key_data_(context);
        std::int32_t *direct_key_sums = key_sums_data_(context);
        const std::int8_t *direct_scores = scores_data_(context);
        if (direct_query == nullptr || direct_key == nullptr
            || direct_key_sums == nullptr || direct_scores == nullptr) {
            error = "INT8 direct operator returned a null arena pointer";
            return false;
        }

        const int actual_n = (request.key_len + 31) & ~31;
        const std::size_t group_capacity
            = static_cast<std::size_t>(info_.heads);
        bool ok = true;
        for (std::size_t begin = 0; begin < validated.matrix_count
             && ok; begin += group_capacity) {
            const std::size_t count = std::min(
                group_capacity, validated.matrix_count - begin);
            if (!quantizeKeyGroup(request, begin, count, profile_enabled,
                                 info_.k_scale, nullptr, direct_key,
                                 actual_n, direct_key_sums)) {
                error = "INT8 direct quantize K failed";
                return false;
            }
            for (int query_begin = 0;
                 query_begin < request.query_len && ok;
                 query_begin += info_.m) {
                const int query_count = std::min(
                    info_.m, request.query_len - query_begin);
                if (!quantizeQueryGroup(request, begin, count, query_begin,
                                       query_count, profile_enabled,
                                       info_.q_scale, nullptr,
                                       direct_query)) {
                    error = "INT8 direct quantize Q failed";
                    return false;
                }
                int status = 0;
                {
                    ScopedAttentionProfile mm_profile(
                        AttentionProfileStage::HMX_MM, profile_enabled);
                    status = matmul_(
                        context,
                        direct_query,
                        direct_key,
                        const_cast<std::int8_t *>(direct_scores));
                }
                if (status != 0) {
                    error = callError("hmx_qk_i8_operator_matmul", status);
                    ok = false;
                    break;
                }
                copyScoreBlock(begin, count, query_begin, query_count,
                              request.query_len, nullptr, direct_scores,
                              actual_n, request.key_len);
            }
            if (ok && on_group_ready) {
                on_group_ready({begin, count});
            }
        }
        return ok;
    }

    bool produceAllGroups(const Request &request, const ValidatedRequest &validated,
                         bool profile_enabled,
                         const ReadyGroupCallback &on_group_ready,
                         std::string &error) {
        if (raw_path_supported_) {
            return produceAllGroupsRaw(request, validated, profile_enabled,
                                       on_group_ready, error);
        }
        return produceAllGroupsPacked(request, validated, profile_enabled,
                                     on_group_ready, error);
    }

    bool measureMatrixScales(const Request &request,
                             const ValidatedRequest &validated,
                             bool profile_enabled,
                             std::vector<MatrixScales> &scales,
                             std::unique_lock<std::mutex> &operator_lock,
                             ScaleProfileStaging &staging,
                             std::string &error) {
        scales.assign(validated.matrix_count, {});
        const bool incremental_key_scale = incrementalKeyScaleEnabled();
        const bool cpu_scale_profile = cpuScaleProfileEnabled();
        if (cpu_scale_profile) {
            const bool combine_scale_and_pack =
                perHeadBucketFusionGroupHeads(validated.matrix_count)
                == validated.matrix_count;
            if (!cpuPackedPipelineEnabled()) {
                error = "CPU Q/K scale profiling requires "
                    "MLLM_HMX_INT8_CPU_PACK=1";
                return false;
            }
            if (!perHeadBucketFusionEnabled()
                || dynamicQKScaleEnabled() || dynamicOutputScaleEnabled()
                || dspInt32TopKEnabled() || outlierFallbackEnabled()) {
                error = "CPU Q/K scale profiling requires static per-head "
                    "buckets, CPU Top-k, and no outlier fallback";
                return false;
            }
            try {
                ensureCpuPackTaskExecutorReady();
            } catch (const std::exception &exception) {
                error = std::string(
                    "failed to start CPU Q/K scale/pack worker: ")
                    + exception.what();
                return false;
            }
            if (layer_key_scale_cache_.size()
                != validated.matrix_count) {
                layer_key_scale_cache_.assign(
                    validated.matrix_count, LayerKeyScaleCache{});
            }
            staging.cpu_job = std::make_shared<CpuScalePackJob>();
            const std::shared_ptr<CpuScalePackJob> cpu_job =
                staging.cpu_job;
            staging.cpu_future = cpuPackTaskExecutor().submit(
                [&, cpu_job]() {
                    try {
                        {
                            ScopedAttentionProfile scale_profile(
                                AttentionProfileStage::HMX_SCALE_PROFILE,
                                profile_enabled);
                            cpu_job->profile_ok = true;
                            for (std::size_t matrix = 0;
                                 matrix < validated.matrix_count; ++matrix) {
                                const std::size_t query_stride =
                                    request.query_f32_row_stride == 0
                                    ? static_cast<std::size_t>(
                                        request.head_dim)
                                    : request.query_f32_row_stride;
                                float q_scale = 0.0F;
                                if (!cpuMaxAbsScaleF32(
                                        request.query_f32_heads[matrix],
                                        request.query_len, query_stride,
                                        request.head_dim, q_scale)) {
                                    cpu_job->profile_error =
                                        "non-finite F32 Q value during CPU "
                                        "scale profiling";
                                    cpu_job->profile_ok = false;
                                    cpu_job->phase.store(
                                        CpuScalePackJob::DONE,
                                        std::memory_order_release);
                                    return;
                                }
                                scales[matrix].q = q_scale;

                                const Half *source = request.key_heads[matrix];
                                LayerKeyScaleCache &cache =
                                    layer_key_scale_cache_[matrix];
                                int key_begin = 0;
                                float previous_scale = 0.0F;
                                if (incremental_key_scale) {
                                    if (cache.source != source
                                        || cache.tokens < 0
                                        || cache.tokens > request.key_len) {
                                        cache.reset(source);
                                    }
                                    key_begin = cache.tokens;
                                    previous_scale = cache.scale;
                                }
                                float measured_scale = 0.0F;
                                if (key_begin < request.key_len
                                    && !cpuMaxAbsScaleF16(
                                        source
                                            + static_cast<std::size_t>(
                                                key_begin)
                                                * request.head_dim,
                                        request.key_len - key_begin,
                                        static_cast<std::size_t>(
                                            request.head_dim),
                                        request.head_dim, measured_scale)) {
                                    cpu_job->profile_error =
                                        "non-finite F16 K value during CPU "
                                        "scale profiling";
                                    cpu_job->profile_ok = false;
                                    cpu_job->phase.store(
                                        CpuScalePackJob::DONE,
                                        std::memory_order_release);
                                    return;
                                }
                                float k_scale = std::max(previous_scale,
                                                         measured_scale);
                                if (!(k_scale > 0.0F)) {
                                    k_scale =
                                        std::numeric_limits<float>::min();
                                }
                                scales[matrix].k = k_scale;
                                if (incremental_key_scale) {
                                    cache.source = source;
                                    cache.tokens = request.key_len;
                                    cache.scale = k_scale;
                                }
                            }
                        }
                        cpu_job->phase.store(
                            CpuScalePackJob::SCALES_READY,
                            std::memory_order_release);
                        int phase = CpuScalePackJob::SCALES_READY;
                        while (phase == CpuScalePackJob::SCALES_READY) {
                            cpuStageSpinPause();
                            phase = cpu_job->phase.load(
                                std::memory_order_acquire);
                        }
                        if (phase == CpuScalePackJob::CANCELLED) return;
                        if (phase != CpuScalePackJob::PACK_READY
                            || cpu_job->request == nullptr) {
                            cpu_job->pack_ok = false;
                            cpu_job->phase.store(
                                CpuScalePackJob::DONE,
                                std::memory_order_release);
                            return;
                        }
                        cpu_job->pack_ok = cpuPackPerHeadGroup(
                            *cpu_job->request, cpu_job->count,
                            cpu_job->matrix_indices, cpu_job->query_begin,
                            cpu_job->query_count, cpu_job->key_begin,
                            cpu_job->padded_n, cpu_job->q_scales,
                            cpu_job->k_scales, cpu_job->direct_query,
                            cpu_job->direct_key, cpu_job->direct_sums,
                            cpu_job->ready, cpu_job->ready_stride,
                            cpu_job->padded_m, cpu_job->k,
                            cpu_job->profile_enabled);
                        cpu_job->phase.store(
                            CpuScalePackJob::DONE, std::memory_order_release);
                    } catch (...) {
                        cpu_job->phase.store(
                            CpuScalePackJob::DONE, std::memory_order_release);
                        throw;
                    }
                });
            int cpu_phase = CpuScalePackJob::PROFILING;
            while (cpu_phase == CpuScalePackJob::PROFILING) {
                cpuStageSpinPause();
                cpu_phase = cpu_job->phase.load(std::memory_order_acquire);
            }
            if (cpu_phase != CpuScalePackJob::SCALES_READY
                || !cpu_job->profile_ok) {
                if (staging.cpu_future.valid()) {
                    try {
                        staging.cpu_future.get();
                    } catch (const std::exception &exception) {
                        error = std::string(
                            "CPU Q/K scale profiling failed: ")
                            + exception.what();
                    }
                }
                if (error.empty()) {
                    error = cpu_job->profile_error.empty()
                        ? "CPU Q/K scale profiling failed"
                        : cpu_job->profile_error;
                }
                staging.cpu_job.reset();
                return false;
            }

            if (!combine_scale_and_pack) {
                // A per-head NPU pipeline cannot hand one all-head packing job
                // back to this scale worker. Finish the profiling phase now;
                // produceBucketedGroups will submit one persistent-pool pack
                // job for each actual fused NPU group below. This preserves the
                // same CPU scale values while allowing HMX(group i), Top-k(i-1)
                // and sparse(i-2) to overlap.
                cpu_job->phase.store(
                    CpuScalePackJob::CANCELLED, std::memory_order_release);
                try {
                    staging.cpu_future.get();
                } catch (const std::exception &exception) {
                    error = std::string("CPU Q/K scale profiling failed: ")
                        + exception.what();
                    staging.cpu_job.reset();
                    return false;
                } catch (...) {
                    error = "CPU Q/K scale profiling failed";
                    staging.cpu_job.reset();
                    return false;
                }
                staging.cpu_job.reset();
            }

            // Bucket selection happens immediately after this function. Keep
            // the operator lock/context lifetime identical to the HVX path,
            // but deliberately leave ScaleProfileStaging empty: CPU profiling
            // reads the original tensors and does not stage raw Q/K in rpcmem.
            operator_lock.lock();
            if (context_ == nullptr) {
                ScopedAttentionProfile init_profile(
                    AttentionProfileStage::HMX_SESSION_INIT,
                    profile_enabled);
                const int status = acquireSharedContext(
                    context_key_, operator_path_, create_, destroy_,
                    &context_);
                if (status != 0 || context_ == nullptr) {
                    error = callError("hmx_i8_create", status);
                    staging.cancelCpuJob();
                    return false;
                }
            }
            return true;
        }
        const bool dsp_incremental_key_scale = incremental_key_scale
            && profile_raw_incremental_ != nullptr;
        std::vector<int> key_begins(validated.matrix_count, 0);
        std::vector<float> previous_k_scales(validated.matrix_count, 0.0F);
        if (incremental_key_scale) {
            if (layer_key_scale_cache_.size() != validated.matrix_count) {
                layer_key_scale_cache_.assign(
                    validated.matrix_count, LayerKeyScaleCache{});
            }
            for (std::size_t matrix = 0;
                 matrix < validated.matrix_count; ++matrix) {
                const Half *source = request.key_heads[matrix];
                LayerKeyScaleCache &cache = layer_key_scale_cache_[matrix];
                if (cache.source != source || cache.tokens < 0
                    || cache.tokens > request.key_len) {
                    cache.reset(source);
                }
                key_begins[matrix] = cache.tokens;
                previous_k_scales[matrix] = cache.scale;
                if (!dsp_incremental_key_scale) {
                    ScopedAttentionProfile scale_profile(
                        AttentionProfileStage::HMX_SCALE_PROFILE,
                        profile_enabled);
                    float maximum = cache.scale * 127.0F;
                    for (int token = cache.tokens; token < request.key_len;
                         ++token) {
                        const Half *row = source
                            + static_cast<std::size_t>(token)
                                * static_cast<std::size_t>(request.head_dim);
                        for (int dimension = 0;
                             dimension < request.head_dim; ++dimension) {
                            const std::uint16_t bits =
                                static_cast<std::uint16_t>(row[dimension])
                                & UINT16_C(0x7fff);
                            if (bits >= UINT16_C(0x7c00)) {
                                error = "non-finite F16 K value during "
                                        "incremental scale profiling";
                                return false;
                            }
                            maximum = std::max(maximum, halfToFloat(bits));
                        }
                    }
                    cache.scale = maximum > 0.0F
                        ? maximum * (1.0F / 127.0F)
                        : std::numeric_limits<float>::min();
                    cache.tokens = request.key_len;
                    scales[matrix].k = cache.scale;
                }
            }
        }
        operator_lock.lock();
        if (context_ == nullptr) {
            ScopedAttentionProfile init_profile(
                AttentionProfileStage::HMX_SESSION_INIT, profile_enabled);
            const int status = acquireSharedContext(
                context_key_, operator_path_, create_, destroy_, &context_);
            if (status != 0 || context_ == nullptr) {
                error = callError("hmx_i8_create", status);
                return false;
            }
        }
        float *raw_query = raw_query_data_(context_);
        Half *raw_key = raw_key_data_(context_);
        if (raw_query == nullptr || raw_key == nullptr) {
            error = "INT8 scale profiler returned a null raw arena pointer";
            return false;
        }
        const int actual_n = (request.key_len + 31) & ~31;
        std::vector<float> q_chunk(static_cast<std::size_t>(info_.heads));
        std::vector<float> k_group(static_cast<std::size_t>(info_.heads));
        const std::size_t capacity = static_cast<std::size_t>(info_.heads);
        for (std::size_t begin = 0; begin < validated.matrix_count;
             begin += capacity) {
            const std::size_t count = std::min(
                capacity, validated.matrix_count - begin);
            int group_key_begin = 0;
            if (dsp_incremental_key_scale) {
                group_key_begin = request.key_len;
                for (std::size_t local = 0; local < count; ++local) {
                    group_key_begin = std::min(
                        group_key_begin, key_begins[begin + local]);
                    k_group[local] = previous_k_scales[begin + local];
                }
            }
            if ((!incremental_key_scale || dsp_incremental_key_scale)
                && !copyRawKeyGroup(
                    request, begin, count, profile_enabled, nullptr, raw_key,
                    actual_n, group_key_begin)) {
                error = "failed to stage raw K for HVX scale profiling";
                return false;
            }
            bool first_chunk = true;
            for (int query_begin = 0; query_begin < request.query_len;
                 query_begin += info_.m) {
                const int query_count = std::min(
                    info_.m, request.query_len - query_begin);
                if (!copyRawQueryGroup(request, begin, count, query_begin,
                                       query_count, profile_enabled, nullptr,
                                       raw_query)) {
                    error = "failed to copy raw Q for HVX scale profiling";
                    return false;
                }
                int status = 0;
                {
                    ScopedAttentionProfile scale_profile(
                        AttentionProfileStage::HMX_SCALE_PROFILE,
                        profile_enabled);
                    if (dsp_incremental_key_scale && first_chunk) {
                        status = profile_raw_incremental_(
                            context_, static_cast<std::int32_t>(count),
                            query_count, actual_n, group_key_begin, 1,
                            k_group.data(), q_chunk.data(), k_group.data());
                    } else {
                        status = profile_raw_(
                            context_, static_cast<std::int32_t>(count),
                            query_count, actual_n, 1,
                            !incremental_key_scale && first_chunk ? 1 : 0,
                            q_chunk.data(), k_group.data());
                    }
                }
                if (status != 0) {
                    error = callError(
                        "hmx_qk_i8_operator_profile_raw_scales_hn", status);
                    return false;
                }
                for (std::size_t local = 0; local < count; ++local) {
                    MatrixScales &destination = scales[begin + local];
                    destination.q = std::max(destination.q, q_chunk[local]);
                    if ((!incremental_key_scale
                         || dsp_incremental_key_scale) && first_chunk) {
                        destination.k = k_group[local];
                        if (dsp_incremental_key_scale) {
                            LayerKeyScaleCache &cache =
                                layer_key_scale_cache_[begin + local];
                            cache.scale = k_group[local];
                            cache.tokens = request.key_len;
                        }
                    }
                }
                first_chunk = false;
            }
            if (begin == 0 && count == validated.matrix_count
                && request.query_len <= info_.m) {
                staging.context = context_;
                staging.matrix_count = count;
                staging.query_begin = 0;
                staging.query_count = request.query_len;
                staging.key_begin = group_key_begin;
                staging.query_staged = true;
                staging.key_staged = !incremental_key_scale
                    || dsp_incremental_key_scale;
            }
        }
        return true;
    }

    std::size_t closestBucket(const MatrixScales &scale) const noexcept {
        std::size_t best = 0;
        float best_error = std::numeric_limits<float>::infinity();
        for (std::size_t index = 0; index < bucket_operators_.size();
             ++index) {
            const float q_error = scale.q
                - bucket_operators_[index].info.q_scale;
            const float k_error = scale.k
                - bucket_operators_[index].info.k_scale;
            const float mse = q_error * q_error + k_error * k_error;
            if (mse < best_error) {
                best = index;
                best_error = mse;
            }
        }
        return best;
    }

    std::size_t bucketOperatorForCount(std::size_t bucket_index,
                                       std::size_t remaining) const noexcept {
        const OperatorInfo &target = bucket_operators_[bucket_index].info;
        const std::size_t group_limit = bucketGroupHeadLimit();
        const std::size_t requested = group_limit == 0
            ? remaining : std::min(remaining, group_limit);
        std::size_t best = bucket_operators_.size();
        int best_heads = std::numeric_limits<int>::max();
        std::size_t fallback = bucket_operators_.size();
        int fallback_heads = 0;
        for (std::size_t index = 0; index < bucket_operators_.size();
             ++index) {
            const OperatorInfo &candidate = bucket_operators_[index].info;
            if (!nearlyEqual(candidate.q_scale, target.q_scale)
                || !nearlyEqual(candidate.k_scale, target.k_scale)
                || !nearlyEqual(candidate.output_scale,
                                target.output_scale)
                || candidate.heads <= 0) {
                continue;
            }
            if (candidate.heads > fallback_heads) {
                fallback = index;
                fallback_heads = candidate.heads;
            }
            if (static_cast<std::size_t>(candidate.heads) >= requested
                && candidate.heads < best_heads) {
                best = index;
                best_heads = candidate.heads;
            }
        }
        return best == bucket_operators_.size() ? fallback : best;
    }

    static std::size_t bucketGroupHeadLimit() noexcept {
        const char *value = std::getenv(
            "MLLM_HMX_INT8_BUCKET_MAX_GROUP_HEADS");
        if (value == nullptr || value[0] == '\0') return 0;
        char *end = nullptr;
        const unsigned long parsed = std::strtoul(value, &end, 10);
        if (end == value || *end != '\0' || parsed == 0
            || parsed > 64UL) {
            return 0;
        }
        return static_cast<std::size_t>(parsed);
    }

    std::size_t perHeadBucketOperatorForCount(
        std::size_t count, float output_scale,
        bool cpu_packed_pipeline,
        bool require_per_head_requant) const noexcept {
        std::size_t best = bucket_operators_.size();
        int best_heads = std::numeric_limits<int>::max();
        for (std::size_t index = 0; index < bucket_operators_.size();
             ++index) {
            const BucketOperator &candidate = bucket_operators_[index];
            const bool compatible_abi = cpu_packed_pipeline
                ? candidate.cpu_pack_ready_data != nullptr
                    && candidate.execute_cpu_packed_per_head_hn != nullptr
                : require_per_head_requant
                ? candidate.prepare_execute_raw_per_head_requant_hn != nullptr
                : candidate.prepare_execute_raw_per_head_requant_hn != nullptr
                    || (candidate.prepare_execute_raw_per_head_hn != nullptr
                        && nearlyEqual(candidate.info.output_scale,
                                       output_scale));
            if (!compatible_abi
                || candidate.info.heads < static_cast<int>(count)
                || candidate.info.heads >= best_heads) {
                continue;
            }
            best = index;
            best_heads = candidate.info.heads;
        }
        return best;
    }

    static bool bucketDiagnosticsEnabled() noexcept {
        const char *value = std::getenv(
            "MLLM_HMX_INT8_BUCKET_DIAGNOSTICS");
        return value != nullptr && value[0] != '\0'
            && std::strcmp(value, "0") != 0;
    }

    static bool dynamicOutputScaleEnabled() noexcept {
        const char *value = std::getenv(
            "MLLM_HMX_INT8_DYNAMIC_OUTPUT_SCALE");
        return value != nullptr && value[0] != '\0'
            && std::strcmp(value, "0") != 0;
    }

    static bool incrementalKeyScaleEnabled() noexcept {
        const char *value = std::getenv(
            "MLLM_HMX_INT8_INCREMENTAL_K_SCALE");
        return value == nullptr || value[0] == '\0'
            || std::strcmp(value, "0") != 0;
    }

    static bool perHeadBucketFusionEnabled() noexcept {
        const char *value = std::getenv(
            "MLLM_HMX_INT8_FUSE_PER_HEAD_BUCKETS");
        return value != nullptr && value[0] != '\0'
            && std::strcmp(value, "0") != 0;
    }

    static bool cpuPackedPipelineEnabled() noexcept {
        const char *value = std::getenv("MLLM_HMX_INT8_CPU_PACK");
        return value != nullptr && value[0] != '\0'
            && std::strcmp(value, "0") != 0;
    }

    static bool cpuScaleProfileEnabled() noexcept {
        const char *value = std::getenv(
            "MLLM_HMX_INT8_CPU_SCALE_PROFILE");
        return value != nullptr && value[0] != '\0'
            && std::strcmp(value, "0") != 0;
    }

    static bool cpuPackOverlapEnabled() noexcept {
        const char *value = std::getenv("MLLM_HMX_CPU_PACK_OVERLAP");
        return value == nullptr || value[0] == '\0'
            || std::strcmp(value, "0") != 0;
    }

    static bool longRpcPipelineEnabled() noexcept {
        const char *value = std::getenv("MLLM_HMX_PIPELINE_LONG_RPC");
        return value != nullptr && value[0] != '\0'
            && std::strcmp(value, "0") != 0;
    }

    static bool cpuPackIntraHeadEnabled() noexcept {
        const char *value = std::getenv(
            "MLLM_HMX_INT8_CPU_PACK_INTRA_HEAD");
        return value != nullptr && value[0] != '\0'
            && std::strcmp(value, "0") != 0;
    }

    static bool dspTimingEnabled() noexcept {
        const char *value = std::getenv("MLLM_HMX_INT8_DSP_TIMING");
        return value != nullptr && value[0] != '\0'
            && std::strcmp(value, "0") != 0;
    }

    static std::size_t perHeadBucketFusionGroupHeads(
        std::size_t matrix_count) noexcept {
        const char *value = std::getenv(
            "MLLM_HMX_INT8_PER_HEAD_BUCKET_GROUP_HEADS");
        if (value == nullptr || value[0] == '\0') return matrix_count;
        char *end = nullptr;
        const unsigned long parsed = std::strtoul(value, &end, 10);
        if (end == value || *end != '\0' || parsed == 0
            || parsed > matrix_count) {
            return 0;
        }
        return static_cast<std::size_t>(parsed);
    }

    static std::size_t pipelineReadyGroupHeads() noexcept {
        const char *value = std::getenv(
            "MLLM_HMX_PIPELINE_READY_GROUP_HEADS");
        if (value == nullptr || value[0] == '\0') return 1;
        char *end = nullptr;
        const unsigned long parsed = std::strtoul(value, &end, 10);
        if (end == value || *end != '\0' || parsed == 0
            || parsed > 64UL) {
            return 1;
        }
        return static_cast<std::size_t>(parsed);
    }

    static bool dynamicQKScaleEnabled() noexcept {
        const char *value = std::getenv("MLLM_HMX_INT8_DYNAMIC_QK_SCALE");
        return value != nullptr && value[0] != '\0'
            && std::strcmp(value, "0") != 0;
    }

    static bool dspInt32TopKEnabled() noexcept {
        const char *value = std::getenv("MLLM_HMX_INT8_DSP_INT32_TOPK");
        return value != nullptr && value[0] != '\0'
            && std::strcmp(value, "0") != 0;
    }

    static bool outlierFallbackEnabled() noexcept {
        const char *value = std::getenv(
            "MLLM_HMX_INT8_OUTLIER_FALLBACK");
        return value == nullptr || value[0] == '\0'
            || std::strcmp(value, "0") != 0;
    }

    static bool greedyPipelineRequested(std::string &error) {
        const char *value = std::getenv("MLLM_HMX_PIPELINE_SCHEDULE");
        if (value == nullptr || value[0] == '\0'
            || std::strcmp(value, "fifo") == 0) {
            return false;
        }
        if (std::strcmp(value, "greedy") == 0) return true;
        error = "MLLM_HMX_PIPELINE_SCHEDULE must be fifo or greedy";
        return false;
    }

    static std::mutex &latencyProfileMutex() {
        static auto *mutex = new std::mutex();
        return *mutex;
    }

    static std::map<std::string,
                    std::shared_ptr<const hmx_pipeline::LatencyProfile>>
        &latencyProfiles() {
        static auto *profiles = new std::map<
            std::string, std::shared_ptr<const hmx_pipeline::LatencyProfile>>();
        return *profiles;
    }

    static std::shared_ptr<const hmx_pipeline::LatencyProfile>
    loadLatencyProfile(std::string &error) {
        const char *path = std::getenv(
            "MLLM_HMX_PIPELINE_LATENCY_PROFILE");
        if (path == nullptr || path[0] == '\0') {
            error = "greedy HMX pipeline requires "
                    "MLLM_HMX_PIPELINE_LATENCY_PROFILE";
            return nullptr;
        }
        std::lock_guard<std::mutex> lock(latencyProfileMutex());
        auto &profiles = latencyProfiles();
        const auto found = profiles.find(path);
        if (found != profiles.end()) return found->second;
        try {
            auto profile = std::make_shared<hmx_pipeline::LatencyProfile>(
                hmx_pipeline::LatencyProfile::load(path));
            profiles.emplace(path, profile);
            return profile;
        } catch (const std::exception &exception) {
            error = exception.what();
            return nullptr;
        }
    }

    bool validateGreedyConfiguration(
        const Request &request, const hmx_pipeline::LatencyProfile &profile,
        std::string &error) const {
        const char *model = std::getenv("MLLM_HMX_INT8_OPERATOR_MODEL");
        const std::string device = runtimeDeviceName();
        const char *main_cpu = std::getenv("MLLM_HMX_PIPELINE_MAIN_CPU");
        const char *topk_cpus = std::getenv("MLLM_HMX_PIPELINE_TOPK_CPU");
        const char *sparse_cpus = std::getenv(
            "MLLM_HMX_PIPELINE_SPARSE_CPU");
        const char *topk_mode = std::getenv(
            "MLLM_HMX_PIPELINE_TOPK_MODE");
        const bool supported_topk_mode = topk_mode != nullptr
            && (std::strcmp(topk_mode, "cooperative") == 0
                || std::strcmp(topk_mode, "independent") == 0);
        if (!supported_topk_mode) {
            error = "greedy HMX pipeline requires an explicit cooperative "
                    "or independent Top-k mode";
        } else if (!layerPackedKeyCacheEnabled()
                   || (!attentionExecutionScopeEnabled()
                       && !(directThreeStageExecutorEnabled()
                            && longRpcPipelineEnabled()))) {
            error = "greedy HMX pipeline profile requires incremental "
                    "layer K cache and attention-level execution scope";
        } else if (request.pipeline_group_heads == 0
                   || !request.topk_group_ready) {
            error = "greedy HMX scheduling requires the three-stage "
                    "pipeline";
        } else if (dynamicQKScaleEnabled() || dynamicOutputScaleEnabled()
                   || dspInt32TopKEnabled() || topKOversample() != 1.0F
                   || outlierFallbackEnabled()) {
            error = "greedy HMX pipeline requires static INT8 buckets, CPU "
                    "exact Top-k, oversample=1, and no fallback";
        } else if (request.layer_id < 0
                   || request.head_retentions == nullptr
                   || request.head_retention_count
                       != static_cast<std::size_t>(request.heads)) {
            error = "greedy HMX pipeline requires layer id and per-head "
                    "retentions";
        } else if (model == nullptr || model[0] == '\0'
                   || device.empty()
                   || main_cpu == nullptr || main_cpu[0] == '\0'
                   || topk_cpus == nullptr || topk_cpus[0] == '\0'
                   || sparse_cpus == nullptr || sparse_cpus[0] == '\0') {
            error = "greedy HMX pipeline requires explicit model and CPU "
                    "resource bindings";
        } else {
            try {
                profile.validateRuntimeBinding(
                    model, device, request.query_len, request.head_dim,
                    request.key_len, main_cpu, topk_cpus, sparse_cpus);
            } catch (const std::exception &exception) {
                error = exception.what();
            }
        }
        return error.empty();
    }

    static std::string runtimeDeviceName() {
#if defined(__ANDROID__)
        char value[PROP_VALUE_MAX] = {};
        const int length = __system_property_get("ro.product.device", value);
        return length > 0 ? std::string(value, static_cast<std::size_t>(length))
                          : std::string();
#else
        const char *value = std::getenv("MLLM_HMX_PIPELINE_DEVICE");
        return value == nullptr ? std::string() : std::string(value);
#endif
    }

    bool bucketScaleCovered(const MatrixScales &scale) const noexcept {
        float maximum_q = 0.0F;
        float maximum_k = 0.0F;
        for (const BucketOperator &bucket : bucket_operators_) {
            maximum_q = std::max(maximum_q, bucket.info.q_scale);
            maximum_k = std::max(maximum_k, bucket.info.k_scale);
        }
        const float tolerance = 1.0F + 1.0e-5F;
        return scale.q <= maximum_q * tolerance
            && scale.k <= maximum_k * tolerance;
    }

    bool selectExactFloatTopKMatrix(const Request &request,
                                    const ValidatedRequest &validated,
                                    std::size_t matrix,
                                    bool profile_enabled) {
        ScopedAttentionProfile profile(
            AttentionProfileStage::HMX_TOPK, profile_enabled);
        const std::size_t query_stride = request.query_f32_row_stride == 0
            ? static_cast<std::size_t>(request.head_dim)
            : request.query_f32_row_stride;
        const float *query_base = request.query_f32_heads[matrix];
        const Half *key_base = request.key_heads[matrix];
        const std::int32_t *offsets = rowOffsetsForMatrix(request, matrix);
        std::vector<std::pair<float, std::int32_t>> ranked;
        ranked.resize(static_cast<std::size_t>(request.key_len));
        const auto score_order = [](const auto &left, const auto &right) {
            if (left.first != right.first) return left.first > right.first;
            return left.second < right.second;
        };
        const auto index_order = [](const auto &left, const auto &right) {
            return left.second < right.second;
        };
        for (int row = 0; row < request.query_len; ++row) {
            const int valid = std::min(
                request.key_len,
                validated.causal_prefix_tokens + row + 1);
            const int keep = offsets[row + 1] - offsets[row];
            const float *query = query_base
                + static_cast<std::size_t>(row) * query_stride;
            for (int key = 0; key < valid; ++key) {
                const Half *key_vector = key_base
                    + static_cast<std::size_t>(key) * request.head_dim;
                float score = 0.0F;
                for (int dimension = 0; dimension < request.head_dim;
                     ++dimension) {
                    score += query[dimension]
                        * halfToFloat(key_vector[dimension]);
                }
                ranked[static_cast<std::size_t>(key)] = {score, key};
            }
            if (keep < valid) {
                std::nth_element(
                    ranked.begin(), ranked.begin() + keep,
                    ranked.begin() + valid, score_order);
            }
            std::sort(ranked.begin(), ranked.begin() + keep, index_order);
            std::int32_t *destination = request.topk_indices
                + matrix * validated.topk_head_stride + offsets[row];
            for (int selected = 0; selected < keep; ++selected) {
                destination[selected]
                    = ranked[static_cast<std::size_t>(selected)].second;
            }
        }
        topk_preselected_[matrix] = 1;
        return true;
    }

    static std::string bucketCallError(const BucketOperator &bucket,
                                       const char *operation, int status) {
        std::string result(operation);
        result += " failed with status " + std::to_string(status);
        if (bucket.status_string != nullptr) {
            const char *description = bucket.status_string(status);
            if (description != nullptr && description[0] != '\0') {
                result += " (";
                result += description;
                result += ")";
            }
        }
        result += "; bucket=" + bucket.path;
        return result;
    }

    bool produceBucketedGroups(const Request &request,
                               const ValidatedRequest &validated,
                               bool profile_enabled,
                               const ReadyGroupCallback &on_group_ready,
                               std::string &error,
                               std::future<void> *deferred_cache_store = nullptr,
                               const DirectScoreStreamCallback
                                   &on_direct_stream = {}) {
        std::vector<MatrixScales> scales;
        std::unique_lock<std::mutex> operator_lock(
            processOperatorMutex(), std::defer_lock);
        ScaleProfileStaging scale_staging;
        if (!measureMatrixScales(request, validated, profile_enabled,
                                 scales, operator_lock, scale_staging,
                                 error)) {
            return false;
        }
        const bool dynamic_qk_scale = dynamicQKScaleEnabled();
        const bool dsp_int32_topk = dspInt32TopKEnabled();
        if (dsp_int32_topk && !dynamic_qk_scale) {
            error = "DSP INT32 Top-k requires "
                "MLLM_HMX_INT8_DYNAMIC_QK_SCALE=1";
            return false;
        }
        if (dynamic_qk_scale) {
            for (std::size_t matrix = 0;
                 matrix < validated.matrix_count; ++matrix) {
                score_q_scales_[matrix] = scales[matrix].q;
                score_k_scales_[matrix] = scales[matrix].k;
            }
        }
        std::map<std::size_t, std::vector<std::size_t>> bucket_matrices;
        std::vector<std::size_t> fallback_matrices;
        std::vector<std::size_t> selected_buckets(
            validated.matrix_count, bucket_operators_.size());
        for (std::size_t matrix = 0; matrix < validated.matrix_count;
             ++matrix) {
            if (outlierFallbackEnabled()
                && !bucketScaleCovered(scales[matrix])) {
                fallback_matrices.push_back(matrix);
            } else {
                const std::size_t selected = closestBucket(scales[matrix]);
                selected_buckets[matrix] = selected;
                bucket_matrices[selected].push_back(matrix);
            }
        }
        if (bucketDiagnosticsEnabled()) {
            for (const auto &entry : bucket_matrices) {
                const BucketOperator &bucket = bucket_operators_[entry.first];
                std::cerr << "[HMX_INT8_BUCKET] op="
                          << (request.diagnostic_label.empty()
                                  ? "unknown" : request.diagnostic_label)
                          << " layer=" << request.layer_id
                          << " key_len=" << request.key_len
                          << " q=" << bucket.info.q_scale
                          << " k=" << bucket.info.k_scale
                          << " output=" << bucket.info.output_scale
                          << " requant=" << bucket.info.requant_scale
                          << " matrices=" << entry.second.size() << ':';
                for (const std::size_t matrix : entry.second) {
                    std::cerr << ' ' << matrix << "(q=" << scales[matrix].q
                              << ",k=" << scales[matrix].k << ')';
                }
                std::cerr << std::endl;
            }
            if (!fallback_matrices.empty()) {
                std::cerr << "[HMX_INT8_BUCKET_FALLBACK] op="
                          << (request.diagnostic_label.empty()
                                  ? "unknown" : request.diagnostic_label)
                          << " layer=" << request.layer_id
                          << " key_len=" << request.key_len
                          << " matrices="
                          << fallback_matrices.size() << ':';
                for (const std::size_t matrix : fallback_matrices) {
                    std::cerr << ' ' << matrix << "(q=" << scales[matrix].q
                              << ",k=" << scales[matrix].k << ')';
                }
                std::cerr << std::endl;
            }
        }

        std::vector<BucketExecutionGroup> execution_groups;
        const bool per_head_bucket_fusion = perHeadBucketFusionEnabled();
        if (per_head_bucket_fusion) {
            if (dynamic_qk_scale || dynamicOutputScaleEnabled()
                || dsp_int32_topk || !fallback_matrices.empty()
                || bucket_matrices.empty()) {
                error = "per-head bucket fusion requires static covered Q/K "
                    "buckets, static per-head requant scales, CPU Top-k, and "
                    "no fallback";
                return false;
            }
            const std::size_t first_bucket = selected_buckets.front();
            const float output_scale =
                bucket_operators_[first_bucket].info.output_scale;
            const bool cpu_packed_pipeline = cpuPackedPipelineEnabled();
            const std::size_t group_capacity =
                perHeadBucketFusionGroupHeads(validated.matrix_count);
            if (group_capacity == 0) {
                error = "MLLM_HMX_INT8_PER_HEAD_BUCKET_GROUP_HEADS must be "
                    "between 1 and the runtime matrix count";
                return false;
            }
            for (std::size_t offset = 0; offset < validated.matrix_count;
                 offset += group_capacity) {
                const std::size_t count = std::min(
                    group_capacity, validated.matrix_count - offset);
                const std::size_t operator_index =
                    perHeadBucketOperatorForCount(
                        count, output_scale, cpu_packed_pipeline,
                        !cpu_packed_pipeline);
                if (operator_index == bucket_operators_.size()) {
                    error = "per-head bucket fusion requires a compatible H="
                        + std::to_string(count)
                        + " operator with heterogeneous-scale ABI";
                    return false;
                }
                BucketExecutionGroup group;
                group.id = execution_groups.size();
                group.bucket_index = selected_buckets[offset];
                group.operator_index = operator_index;
                group.per_head_bucket_scales = true;
                group.matrices.reserve(count);
                for (std::size_t matrix = offset;
                     matrix < offset + count; ++matrix) {
                    group.matrices.push_back(matrix);
                }
                execution_groups.push_back(std::move(group));
            }
            if (bucketDiagnosticsEnabled()) {
                std::cerr << "[HMX_PER_HEAD_BUCKET_FUSION] op="
                          << (request.diagnostic_label.empty()
                                  ? "unknown" : request.diagnostic_label)
                          << " heads=" << validated.matrix_count
                          << " group_heads=" << group_capacity
                          << " rpc_groups=" << execution_groups.size()
                          << std::endl;
            }
        } else {
            for (const auto &entry : bucket_matrices) {
                std::size_t offset = 0;
                while (offset < entry.second.size()) {
                    const std::size_t operator_index = bucketOperatorForCount(
                        entry.first, entry.second.size() - offset);
                    if (operator_index == bucket_operators_.size()) {
                        error = "INT8 HMX bucket has no graph shape for "
                            + std::to_string(entry.second.size() - offset)
                            + " remaining heads";
                        return false;
                    }
                    const BucketOperator &bucket =
                        bucket_operators_[operator_index];
                    const std::size_t count = dsp_int32_topk
                            || dynamicOutputScaleEnabled() || dynamic_qk_scale
                        ? 1U
                        : std::min(
                            static_cast<std::size_t>(bucket.info.heads),
                            entry.second.size() - offset);
                    BucketExecutionGroup group;
                    group.id = execution_groups.size();
                    group.bucket_index = entry.first;
                    group.operator_index = operator_index;
                    group.matrices.assign(
                        entry.second.begin() + offset,
                        entry.second.begin() + offset + count);
                    execution_groups.push_back(std::move(group));
                    offset += count;
                }
            }
        }

        std::string schedule_error;
        const bool greedy_schedule = greedyPipelineRequested(schedule_error);
        if (!schedule_error.empty()) {
            error = schedule_error;
            return false;
        }
        if (greedy_schedule && per_head_bucket_fusion) {
            if (!directThreeStageExecutorEnabled()
                || !longRpcPipelineEnabled()
                || execution_groups.size() != 1U) {
                error = "heterogeneous-scale greedy scheduling requires one "
                    "direct-three-stage H long-RPC group";
                return false;
            }
            const uint64_t profile_begin = CPUAttentionProfiler::nowNs();
            const auto latency_profile = loadLatencyProfile(error);
            if (!latency_profile
                || !validateGreedyConfiguration(
                    request, *latency_profile, error)) {
                return false;
            }
            const uint64_t profile_lookup_us =
                (CPUAttentionProfiler::nowNs() - profile_begin) / 1000U;
            const uint64_t schedule_begin = CPUAttentionProfiler::nowNs();
            try {
                BucketExecutionGroup &group = execution_groups.front();
                const BucketOperator &bucket =
                    bucket_operators_[group.operator_index];
                if (static_cast<std::size_t>(bucket.info.heads)
                        != group.matrices.size()
                    || group.matrices.size() != validated.matrix_count) {
                    throw std::invalid_argument(
                        "streaming greedy requires one exact all-head H "
                        "operator");
                }
                const double npu_group_us = latency_profile->npu(
                    request.key_len, bucket.info.heads,
                    bucket.info.q_scale, bucket.info.k_scale).p50_us;
                const double npu_head_us = npu_group_us
                    / static_cast<double>(group.matrices.size());
                std::vector<hmx_pipeline::StreamingHeadJob> costs;
                costs.reserve(group.matrices.size());
                for (const std::size_t matrix : group.matrices) {
                    const int head = static_cast<int>(
                        matrix % static_cast<std::size_t>(request.heads));
                    const float retention = request.head_retentions[head];
                    costs.push_back({
                        matrix, npu_head_us,
                        latency_profile->topk(
                            request.key_len, request.layer_id, head,
                            retention).p50_us,
                        latency_profile->sparse(
                            request.key_len, request.layer_id, head,
                            retention).p50_us,
                    });
                }
                const auto fifo_prediction =
                    hmx_pipeline::StreamingHeadGreedyScheduler::fifo(costs);
                bool used_greedy = false;
                double candidate_greedy_us = 0.0;
                const auto planned = hmx_pipeline::
                    StreamingHeadGreedyScheduler::planNoWorseThanFifo(
                        costs, &used_greedy, &candidate_greedy_us);
                group.matrices = planned.order;
                const uint64_t schedule_us =
                    (CPUAttentionProfiler::nowNs() - schedule_begin) / 1000U;
                if (pipelinePlanDiagnosticsEnabled()) {
                    std::cerr << "[HMX_STREAMING_HEAD_PLAN] layer="
                              << request.layer_id << " key_len="
                              << request.key_len << " schedule_us="
                              << schedule_us << " profile_lookup_us="
                              << profile_lookup_us << " fifo_us="
                              << fifo_prediction.sparse_finish_us
                              << " greedy_us=" << planned.sparse_finish_us
                              << " candidate_greedy_us="
                              << candidate_greedy_us << " selected="
                              << (used_greedy ? "greedy" : "fifo")
                              << " heads=";
                    for (std::size_t index = 0;
                         index < group.matrices.size(); ++index) {
                        if (index != 0) std::cerr << ',';
                        std::cerr << group.matrices[index];
                    }
                    std::cerr << std::endl;
                }
            } catch (const std::exception &exception) {
                error = exception.what();
                return false;
            }
        } else if (greedy_schedule) {
            const uint64_t profile_begin = CPUAttentionProfiler::nowNs();
            const auto latency_profile = loadLatencyProfile(error);
            if (!latency_profile
                || !validateGreedyConfiguration(
                    request, *latency_profile, error)) {
                return false;
            }
            const uint64_t profile_lookup_us =
                (CPUAttentionProfiler::nowNs() - profile_begin) / 1000U;
            const uint64_t schedule_begin = CPUAttentionProfiler::nowNs();
            try {
                std::vector<hmx_pipeline::FusedGroup> costs;
                costs.reserve(execution_groups.size());
                for (const BucketExecutionGroup &group : execution_groups) {
                    const BucketOperator &bucket =
                        bucket_operators_[group.operator_index];
                    if (static_cast<std::size_t>(bucket.info.heads)
                        != group.matrices.size()) {
                        throw std::invalid_argument(
                            "greedy HMX pipeline requires an exact fused H "
                            "operator for every bucket group");
                    }
                    hmx_pipeline::FusedGroup cost;
                    cost.id = group.id;
                    cost.npu_us = latency_profile->npu(
                        request.key_len, bucket.info.heads,
                        bucket.info.q_scale, bucket.info.k_scale).p50_us;
                    for (const std::size_t matrix : group.matrices) {
                        const int head = static_cast<int>(
                            matrix % static_cast<std::size_t>(request.heads));
                        const float retention = request.head_retentions[head];
                        cost.heads.push_back({
                            matrix,
                            latency_profile->topk(
                                request.key_len, request.layer_id, head,
                                retention).p50_us,
                            latency_profile->sparse(
                                request.key_len, request.layer_id, head,
                                retention).p50_us,
                        });
                    }
                    costs.push_back(std::move(cost));
                }
                const auto fifo_prediction =
                    hmx_pipeline::TwoLevelGreedyScheduler::fifo(costs);
                bool used_greedy = false;
                double candidate_greedy_us = 0.0;
                const auto planned = hmx_pipeline::TwoLevelGreedyScheduler::
                    planNoWorseThanFifo(
                        costs, &used_greedy, &candidate_greedy_us);
                std::vector<BucketExecutionGroup> reordered;
                reordered.reserve(execution_groups.size());
                for (const auto &planned_group : planned.groups) {
                    const auto found = std::find_if(
                        execution_groups.begin(), execution_groups.end(),
                        [&](const BucketExecutionGroup &candidate) {
                            return candidate.id == planned_group.id;
                        });
                    if (found == execution_groups.end()) {
                        throw std::logic_error(
                            "HMX planner returned an unknown group");
                    }
                    BucketExecutionGroup group = *found;
                    group.matrices = planned_group.head_order;
                    reordered.push_back(std::move(group));
                }
                execution_groups = std::move(reordered);
                const uint64_t schedule_us =
                    (CPUAttentionProfiler::nowNs() - schedule_begin) / 1000U;
                if (pipelinePlanDiagnosticsEnabled()) {
                    std::cerr << "[HMX_PIPELINE_PLAN] layer="
                              << request.layer_id << " key_len="
                              << request.key_len << " schedule_us="
                              << schedule_us << " profile_lookup_us="
                              << profile_lookup_us << " fifo_us="
                              << fifo_prediction.sparse_finish_us
                              << " greedy_us=" << planned.sparse_finish_us
                              << " candidate_greedy_us="
                              << candidate_greedy_us << " selected="
                              << (used_greedy ? "greedy" : "fifo")
                              << " groups=";
                    for (const auto &group : execution_groups) {
                        std::cerr << group.id << '[';
                        for (std::size_t index = 0;
                             index < group.matrices.size(); ++index) {
                            if (index != 0) std::cerr << ',';
                            std::cerr << group.matrices[index];
                        }
                        std::cerr << "]";
                    }
                    std::cerr << std::endl;
                }
            } catch (const std::exception &exception) {
                error = exception.what();
                return false;
            }
        }

        if (pipelineTimelineDiagnosticsEnabled()) {
            std::cerr << "[HMX_PIPELINE_GROUPS] layer="
                      << request.layer_id << " key_len=" << request.key_len
                      << " schedule=" << (greedy_schedule ? "greedy" : "fifo")
                      << " fused_h=";
            for (std::size_t index = 0; index < execution_groups.size();
                 ++index) {
                if (index != 0) std::cerr << ',';
                std::cerr << execution_groups[index].matrices.size();
            }
            std::cerr << " groups=";
            for (const auto &group : execution_groups) {
                const BucketOperator &bucket =
                    bucket_operators_[group.operator_index];
                std::cerr << group.id << "(b=" << group.bucket_index
                          << ",h=" << bucket.info.heads << ",heads=";
                for (std::size_t index = 0;
                     index < group.matrices.size(); ++index) {
                    if (index != 0) std::cerr << ',';
                    std::cerr << group.matrices[index];
                }
                std::cerr << ")";
            }
            std::cerr << std::endl;
        }

        const bool dynamic_output_scale = dynamicOutputScaleEnabled();
        const bool attention_execution_scope =
            attentionExecutionScopeEnabled();
        AttentionExecutionScope shared_scope;
        BucketOperator *shared_scope_bucket = nullptr;
        struct PendingNpuLatency {
            int heads = 0;
            float q_scale = 0.0F;
            float k_scale = 0.0F;
            uint64_t elapsed_ns = 0;
        };
        std::vector<PendingNpuLatency> pending_npu_latencies;
        for (const auto &execution_group : execution_groups) {
            const std::vector<std::size_t> &matrices =
                execution_group.matrices;
            bool ok = true;
            std::size_t offset = 0;
            while (offset < matrices.size() && ok) {
                const std::size_t operator_index =
                    execution_group.per_head_bucket_scales
                    ? execution_group.operator_index
                    : bucketOperatorForCount(
                        execution_group.bucket_index,
                        matrices.size() - offset);
                if (operator_index == bucket_operators_.size()) {
                    error = "INT8 HMX bucket has no graph shape for "
                        + std::to_string(matrices.size() - offset)
                        + " remaining heads";
                    return false;
                }
                BucketOperator &bucket
                    = bucket_operators_[operator_index];
                if (bucket.context == nullptr) {
                    if (context_ != nullptr
                        && bucket.context_key == context_key_) {
                        bucket.context = context_;
                        bucket.context_borrowed = true;
                    } else {
                        ScopedAttentionProfile init_profile(
                            AttentionProfileStage::HMX_SESSION_INIT,
                            profile_enabled);
                        const int status = acquireSharedContext(
                            bucket.context_key, bucket.path, bucket.create,
                            bucket.destroy, &bucket.context);
                        if (status != 0 || bucket.context == nullptr) {
                            error = bucketCallError(
                                bucket, "hmx_i8_create", status);
                            if (status == 0) {
                                error += " (returned null context)";
                            }
                            return false;
                        }
                    }
                }
                void *context = bucket.context;
                std::int8_t *direct_query = bucket.query_data(context);
                std::int8_t *direct_key = bucket.key_data(context);
                std::int32_t *direct_key_sums =
                    bucket.key_sums_data(context);
                const std::int8_t *direct_scores
                    = bucket.scores_data(context);
                const std::int32_t *direct_topk =
                    bucket.topk_indices_data == nullptr
                    ? nullptr : bucket.topk_indices_data(context);
                float *raw_query = bucket.raw_query_data(context);
                Half *raw_key = bucket.raw_key_data(context);
                if (direct_query == nullptr || direct_key == nullptr
                    || direct_key_sums == nullptr || direct_scores == nullptr
                    || raw_query == nullptr || raw_key == nullptr) {
                    error = "INT8 bucket direct operator returned a null "
                            "arena pointer; bucket=" + bucket.path;
                    return false;
                }
                if (dsp_int32_topk
                    && (direct_topk == nullptr
                        || bucket.execute_raw_i32_topk_hn == nullptr)) {
                    error = "DSP INT32 Top-k requires the operator Top-k ABI; "
                        "bucket=" + bucket.path;
                    return false;
                }
                const int actual_n = (request.key_len + 31) & ~31;
                while (offset < matrices.size()
                       && (execution_group.per_head_bucket_scales
                           || bucketOperatorForCount(
                                  execution_group.bucket_index,
                                  matrices.size() - offset)
                                  == operator_index) && ok) {
                    const std::size_t count
                        = dsp_int32_topk || dynamic_output_scale
                              || dynamic_qk_scale
                        ? 1
                        : std::min(static_cast<std::size_t>(bucket.info.heads),
                                   matrices.size() - offset);
                    const std::size_t *group = matrices.data() + offset;
                    std::vector<float> per_head_q_scales;
                    std::vector<float> per_head_k_scales;
                    std::vector<float> per_head_requant_scales;
                    if (execution_group.per_head_bucket_scales) {
                        per_head_q_scales.reserve(count);
                        per_head_k_scales.reserve(count);
                        per_head_requant_scales.reserve(count);
                        for (std::size_t local = 0; local < count; ++local) {
                            const std::size_t selected =
                                selected_buckets[group[local]];
                            per_head_q_scales.push_back(
                                bucket_operators_[selected].info.q_scale);
                            per_head_k_scales.push_back(
                                bucket_operators_[selected].info.k_scale);
                            per_head_requant_scales.push_back(
                                bucket_operators_[selected]
                                    .info.requant_scale);
                        }
                    }
                    const uint64_t group_begin_ns =
                        hmx_pipeline::LatencyRecorder::enabled()
                        ? CPUAttentionProfiler::nowNs() : 0;
                    const bool use_layer_key_cache = !dynamic_qk_scale
                        && layerPackedKeyCacheEnabled();
                    const LayerPackedKeyPlan layer_key_plan =
                        use_layer_key_cache
                        ? loadLayerPackedKeyGroup(
                            request, count, group, bucket.info.k_scale,
                            actual_n, direct_key, direct_key_sums,
                            profile_enabled,
                            execution_group.per_head_bucket_scales
                                ? per_head_k_scales.data() : nullptr)
                        : LayerPackedKeyPlan{};
                    const RawKeyPlan shared_key_plan =
                        dynamic_qk_scale || use_layer_key_cache
                        ? RawKeyPlan{}
                        : rawKeyPlan(bucket.context_key, request, 0, count,
                                     group, actual_n);
                    const int key_begin = use_layer_key_cache
                        ? layer_key_plan.key_begin
                        : shared_key_plan.key_begin;
                    const bool reuse_packed = use_layer_key_cache
                        ? layer_key_plan.reuse_packed
                        : shared_key_plan.reuse_packed;
                    const bool cpu_packed_pipeline =
                        cpuPackedPipelineEnabled()
                        && execution_group.per_head_bucket_scales;
                    const bool cpu_pack_overlap =
                        cpu_packed_pipeline && cpuPackOverlapEnabled();
                    const bool long_rpc_requested =
                        longRpcPipelineEnabled()
                        && (static_cast<bool>(on_group_ready)
                            || static_cast<bool>(on_direct_stream));
                    const bool long_rpc_pipeline = long_rpc_requested
                        && cpu_packed_pipeline
                        && execution_groups.size() == 1U
                        && offset == 0U
                        && count == matrices.size()
                        && count
                            == static_cast<std::size_t>(bucket.info.heads)
                        && count == validated.matrix_count
                        && request.query_len <= bucket.info.m
                        && !attention_execution_scope;
                    if (cpu_packed_pipeline
                        && (dynamic_qk_scale || dynamic_output_scale
                            || dsp_int32_topk
                            || bucket.cpu_pack_ready_data == nullptr
                            || bucket.execute_cpu_packed_per_head_hn
                                == nullptr)) {
                        error = "CPU-packed HMX pipeline requires static "
                                "per-head buckets and the CPU-packed operator "
                                "ABI; bucket=" + bucket.path;
                        ok = false;
                        break;
                    }
                    if (long_rpc_requested && !long_rpc_pipeline) {
                        error = "MLLM_HMX_PIPELINE_LONG_RPC=1 requires one "
                                "CPU-packed all-head H operator group, one "
                                "query chunk, and execution scope disabled";
                        ok = false;
                        break;
                    }
                    if (long_rpc_pipeline
                        && (bucket.info.flags & kLongRpcHeadReadyFlag) == 0) {
                        error = "selected HMX operator does not support "
                                "in-flight long-RPC head-ready publication; "
                                "bucket=" + bucket.path;
                        ok = false;
                        break;
                    }
                    if (long_rpc_pipeline) {
                        try {
                            ensureLongRpcTaskExecutorReady();
                        } catch (const std::exception &exception) {
                            error = std::string(
                                "failed to start long-lived HMX RPC worker: ")
                                + exception.what();
                            ok = false;
                            break;
                        }
                    }
                    const bool fuse_prepare_execute =
                        cpu_packed_pipeline
                        || (execution_group.per_head_bucket_scales
                            ? bucket.prepare_execute_raw_per_head_requant_hn
                                    != nullptr
                                || bucket.prepare_execute_raw_per_head_hn
                                    != nullptr
                            : !reuse_packed && !dynamic_qk_scale
                                && !dynamic_output_scale && !dsp_int32_topk
                                && bucket.prepare_execute_raw_hn != nullptr);
                    bool fused_prepare_pending = fuse_prepare_execute;
                    bool long_rpc_group_ready_emitted = false;
                    uint64_t direct_rpc_elapsed_for_profile_ns = 0;
                    const bool fused_rpc_owns_scope =
                        !attention_execution_scope && fuse_prepare_execute;
                    bool scale_staged_group = context == scale_staging.context
                        && count == scale_staging.matrix_count;
                    for (std::size_t local = 0;
                         local < count && scale_staged_group; ++local) {
                        scale_staged_group = group[local] == local;
                    }
                    const bool raw_key_already_staged = scale_staged_group
                        && scale_staging.key_staged
                        && scale_staging.key_begin == key_begin;
                    if (!cpu_packed_pipeline && !reuse_packed
                        && !raw_key_already_staged) {
                        ok = copyRawKeyGroup(
                            request, 0, count, profile_enabled, group,
                            raw_key, actual_n, key_begin);
                    }
                    if (ok && !cpu_packed_pipeline && !reuse_packed
                        && !fuse_prepare_execute) {
                        if (dynamic_qk_scale
                            && bucket.prepare_raw_key_scaled == nullptr) {
                            error = "dynamic INT8 K scaling requires "
                                "hmx_qk_i8_operator_prepare_raw_key_scaled_hn; "
                                "bucket=" + bucket.path;
                            ok = false;
                            break;
                        }
                        int prepare_status = 0;
                        {
                            ScopedAttentionProfile prepare_profile(
                                AttentionProfileStage::HMX_K_PREPARE,
                                profile_enabled);
                            prepare_status = dynamic_qk_scale
                                ? bucket.prepare_raw_key_scaled(
                                    context,
                                    static_cast<std::int32_t>(count),
                                    actual_n, 0, scales[group[0]].k)
                                : bucket.prepare_raw_key(
                                    context,
                                    static_cast<std::int32_t>(count),
                                    actual_n, key_begin);
                        }
                        if (prepare_status != 0) {
                            error = bucketCallError(
                                bucket,
                                "hmx_qk_i8_operator_prepare_raw_key_hn",
                                prepare_status);
                            ok = false;
                            invalidateRawKey(bucket.context_key);
                        } else if (use_layer_key_cache) {
                            commitLayerPackedKeyGroup(
                                request, count, group, bucket.info.k_scale,
                                actual_n, key_begin, direct_key,
                                direct_key_sums, profile_enabled,
                                execution_group.per_head_bucket_scales
                                    ? per_head_k_scales.data() : nullptr);
                        } else if (!dynamic_qk_scale) {
                            commitRawKey(bucket.context_key, request, 0,
                                         count, group, actual_n);
                        }
                    }
                    int scope_status = 0;
                    bool local_scope_active = false;
                    if (ok && attention_execution_scope) {
                        if (!shared_scope.active) {
                            ScopedAttentionProfile begin_profile(
                                AttentionProfileStage::HMX_SCOPE_BEGIN,
                                profile_enabled);
                            scope_status = bucket.begin(context, actual_n);
                            if (scope_status == 0) {
                                shared_scope.arm(context, bucket.end);
                                shared_scope_bucket = &bucket;
                            }
                        }
                    } else if (ok && !fused_rpc_owns_scope) {
                        ScopedAttentionProfile begin_profile(
                            AttentionProfileStage::HMX_SCOPE_BEGIN,
                            profile_enabled);
                        scope_status = bucket.begin(context, actual_n);
                        local_scope_active = scope_status == 0;
                    }
                    if (ok && scope_status != 0) {
                        error = bucketCallError(
                            bucket, "hmx_qk_i8_operator_begin", scope_status);
                        ok = false;
                    }
                    for (int query_begin = 0;
                         query_begin < request.query_len && ok;
                         query_begin += bucket.info.m) {
                        const int query_count = std::min(
                            bucket.info.m,
                            request.query_len - query_begin);
                        if (cpu_packed_pipeline) {
                            try {
                                ensureCpuPackTaskExecutorReady();
                            } catch (const std::exception &exception) {
                                error = std::string(
                                    "failed to start CPU Q/K pack worker: ")
                                    + exception.what();
                                ok = false;
                                break;
                            }
                            volatile std::int32_t *ready =
                                bucket.cpu_pack_ready_data(context);
                            if (ready == nullptr) {
                                error = "CPU-packed HMX operator returned a "
                                        "null ready array; bucket="
                                    + bucket.path;
                                ok = false;
                                break;
                            }
                            const std::size_t ready_stride =
                                (bucket.info.flags
                                 & kLongRpcHeadReadyFlag) != 0
                                ? kLongRpcReadyStride : 1U;
                            for (std::size_t local = 0; local < count;
                                 ++local) {
                                __atomic_store_n(
                                    ready + local * ready_stride, 0,
                                    __ATOMIC_RELEASE);
                            }
                            std::atomic<bool> pack_ok{false};
                            const int cpu_key_begin = query_begin == 0
                                ? key_begin : actual_n;
                            const bool combined_cpu_stage =
                                scale_staging.cpu_job != nullptr;
                            std::future<void> pack_future;
                            if (combined_cpu_stage) {
                                const std::shared_ptr<CpuScalePackJob> job =
                                    scale_staging.cpu_job;
                                if (job->phase.load(std::memory_order_acquire)
                                        != CpuScalePackJob::SCALES_READY
                                    || count != validated.matrix_count) {
                                    error = "combined CPU scale+pack stage "
                                        "did not receive its one all-head "
                                        "packing group";
                                    ok = false;
                                    break;
                                }
                                job->request = &request;
                                job->count = count;
                                job->matrix_indices = group;
                                job->query_begin = query_begin;
                                job->query_count = query_count;
                                job->key_begin = cpu_key_begin;
                                job->padded_n = actual_n;
                                job->q_scales = per_head_q_scales.data();
                                job->k_scales = per_head_k_scales.data();
                                job->direct_query = direct_query;
                                job->direct_key = direct_key;
                                job->direct_sums = direct_key_sums;
                                job->ready = ready;
                                job->ready_stride = ready_stride;
                                job->padded_m = bucket.info.m;
                                job->k = bucket.info.k;
                                job->profile_enabled = profile_enabled;
                                job->phase.store(
                                    CpuScalePackJob::PACK_READY,
                                    std::memory_order_release);
                            } else {
                                pack_future = cpuPackTaskExecutor().submit(
                                    [&, cpu_key_begin]() noexcept {
                                        pack_ok.store(
                                            cpuPackPerHeadGroup(
                                                request, count, group,
                                                query_begin, query_count,
                                                cpu_key_begin, actual_n,
                                                per_head_q_scales.data(),
                                                per_head_k_scales.data(),
                                                direct_query, direct_key,
                                                direct_key_sums, ready,
                                                ready_stride,
                                                bucket.info.m, bucket.info.k,
                                                profile_enabled),
                                            std::memory_order_release);
                                    });
                            }
                            const auto wait_for_cpu_pack = [&]() {
                                try {
                                    if (combined_cpu_stage) {
                                        scale_staging.cpu_future.get();
                                        pack_ok.store(
                                            scale_staging.cpu_job->pack_ok,
                                            std::memory_order_release);
                                        scale_staging.cpu_job.reset();
                                    } else {
                                        pack_future.get();
                                    }
                                } catch (const std::exception &exception) {
                                    error = std::string(
                                        "CPU Q/K pack worker failed: ")
                                        + exception.what();
                                    return false;
                                }
                                if (!pack_ok.load(std::memory_order_acquire)) {
                                    error = "CPU Q/K pack worker returned "
                                            "failure";
                                    return false;
                                }
                                return true;
                            };
                            if (!cpu_pack_overlap
                                && !wait_for_cpu_pack()) {
                                ok = false;
                                break;
                            }
                            int cpu_status = 0;
                            std::string long_rpc_error;
                            {
                                ScopedAttentionProfile fused_profile(
                                    AttentionProfileStage::
                                        HMX_FUSED_PREPARE_EXECUTE,
                                    profile_enabled && !on_direct_stream);
                                if (!long_rpc_pipeline) {
                                    cpu_status =
                                        bucket.execute_cpu_packed_per_head_hn(
                                            context,
                                            static_cast<std::int32_t>(count),
                                            actual_n,
                                            per_head_requant_scales.data());
                                } else {
                                    std::atomic<std::uint64_t>
                                        direct_rpc_elapsed_ns{0};
                                    std::future<void> rpc_future =
                                        longRpcTaskExecutor().submit([&]() {
                                            const std::uint64_t rpc_begin_ns =
                                                (profile_enabled
                                                 || hmx_pipeline::
                                                     LatencyRecorder::enabled())
                                                && on_direct_stream
                                                ? CPUAttentionProfiler::nowNs()
                                                : 0;
                                            cpu_status = bucket
                                                .execute_cpu_packed_per_head_hn(
                                                    context,
                                                    static_cast<std::int32_t>(
                                                    count),
                                                    actual_n,
                                                    per_head_requant_scales
                                                        .data());
                                            if (rpc_begin_ns != 0) {
                                                direct_rpc_elapsed_ns.store(
                                                    CPUAttentionProfiler::
                                                        nowNs()
                                                        - rpc_begin_ns,
                                                    std::memory_order_release);
                                            }
                                        });
                                    const std::size_t source_stride =
                                        static_cast<std::size_t>(
                                            bucket.info.m)
                                        * static_cast<std::size_t>(actual_n);
                                    if (on_direct_stream) {
                                        bool stream_ok = false;
                                        try {
                                            DirectScoreStream stream;
                                            stream.ready = ready;
                                            stream.ready_stride = ready_stride;
                                            stream.scores = direct_scores;
                                            stream.score_head_stride =
                                                source_stride;
                                            stream.score_row_stride =
                                                static_cast<std::size_t>(
                                                    actual_n);
                                            stream.matrices = group;
                                            stream.count = count;
                                            stream_ok = on_direct_stream(
                                                stream, long_rpc_error);
                                        } catch (const std::exception
                                                     &exception) {
                                            long_rpc_error = std::string(
                                                "direct long-RPC score "
                                                "consumer failed: ")
                                                + exception.what();
                                        } catch (...) {
                                            long_rpc_error =
                                                "direct long-RPC score "
                                                "consumer failed";
                                        }
                                        try {
                                            rpc_future.get();
                                        } catch (const std::exception
                                                     &exception) {
                                            if (long_rpc_error.empty()) {
                                                long_rpc_error = std::string(
                                                    "long-lived HMX RPC "
                                                    "worker failed: ")
                                                    + exception.what();
                                            }
                                        } catch (...) {
                                            if (long_rpc_error.empty()) {
                                                long_rpc_error =
                                                    "long-lived HMX RPC "
                                                    "worker failed";
                                            }
                                        }
                                        if (profile_enabled) {
                                            CPUAttentionProfiler::add(
                                                AttentionProfileStage::
                                                    HMX_FUSED_PREPARE_EXECUTE,
                                                direct_rpc_elapsed_ns.load(
                                                    std::memory_order_acquire));
                                        }
                                        direct_rpc_elapsed_for_profile_ns =
                                            direct_rpc_elapsed_ns.load(
                                                std::memory_order_acquire);
                                        if (!stream_ok
                                            && long_rpc_error.empty()) {
                                            long_rpc_error =
                                                "direct long-RPC score "
                                                "consumer returned failure";
                                        }
                                        long_rpc_group_ready_emitted =
                                            stream_ok
                                            && long_rpc_error.empty();
                                    } else {
                                    bool rpc_joined = false;
                                    std::size_t published_heads = 0;
                                    const std::size_t requested_ready_heads =
                                        pipelineReadyGroupHeads();
                                    const auto deadline =
                                        std::chrono::steady_clock::now()
                                        + std::chrono::seconds(6);
                                    while (published_heads < count) {
                                        std::size_t ready_count = 1;
                                        while (ready_count
                                                   < requested_ready_heads
                                               && published_heads
                                                       + ready_count
                                                   < count
                                               && group[published_heads
                                                        + ready_count]
                                                   == group[published_heads]
                                                       + ready_count) {
                                            ++ready_count;
                                        }
                                        bool block_ready = true;
                                        for (std::size_t local = 0;
                                             local < ready_count; ++local) {
                                            const std::int32_t state =
                                                __atomic_load_n(
                                                    ready
                                                        + (published_heads
                                                           + local)
                                                            * ready_stride,
                                                    __ATOMIC_ACQUIRE);
                                            if (state
                                                == kLongRpcOutputError) {
                                                long_rpc_error =
                                                    "long-lived HMX RPC "
                                                    "reported a DSP failure "
                                                    "before head "
                                                    + std::to_string(
                                                        published_heads
                                                        + local);
                                                block_ready = false;
                                                break;
                                            }
                                            if (state
                                                != kLongRpcOutputReady) {
                                                block_ready = false;
                                            }
                                        }
                                        if (block_ready) {
                                            {
                                                ScopedAttentionProfile
                                                    output_profile(
                                                        AttentionProfileStage::
                                                            HMX_OUTPUT_LAYOUT,
                                                        profile_enabled);
                                                copyScoreBlock(
                                                    0, ready_count,
                                                    query_begin,
                                                    query_count,
                                                    request.query_len,
                                                    group + published_heads,
                                                    direct_scores
                                                        + published_heads
                                                            * source_stride,
                                                    actual_n,
                                                    request.key_len);
                                            }
                                            on_group_ready(
                                                {group[published_heads],
                                                 ready_count});
                                            published_heads += ready_count;
                                            continue;
                                        }
                                        if (!long_rpc_error.empty()) break;
                                        if (rpc_joined) {
                                            long_rpc_error =
                                                "long-lived HMX RPC returned "
                                                "without publishing head "
                                                + std::to_string(
                                                    published_heads);
                                            break;
                                        }
                                        if (rpc_future.wait_for(
                                                std::chrono::microseconds(0))
                                                == std::future_status::ready) {
                                            try {
                                                rpc_future.get();
                                                rpc_joined = true;
                                            } catch (const std::exception
                                                         &exception) {
                                                long_rpc_error = std::string(
                                                    "long-lived HMX RPC "
                                                    "worker failed: ")
                                                    + exception.what();
                                                rpc_joined = true;
                                            } catch (...) {
                                                long_rpc_error =
                                                    "long-lived HMX RPC "
                                                    "worker failed";
                                                rpc_joined = true;
                                            }
                                            if (cpu_status != 0
                                                || !long_rpc_error.empty()) {
                                                break;
                                            }
                                            /* A successful RPC return is a
                                             * full memory boundary. Every
                                             * per-head flag must already have
                                             * been published before it. */
                                            continue;
                                        }
                                        if (std::chrono::steady_clock::now()
                                                >= deadline) {
                                            long_rpc_error =
                                                "timed out waiting for "
                                                "in-flight HMX head "
                                                + std::to_string(
                                                    published_heads);
                                            break;
                                        }
                                        std::this_thread::sleep_for(
                                            std::chrono::microseconds(10));
                                    }
                                    if (!rpc_joined) {
                                        try {
                                            rpc_future.get();
                                        } catch (const std::exception
                                                     &exception) {
                                            if (long_rpc_error.empty()) {
                                                long_rpc_error = std::string(
                                                    "long-lived HMX RPC "
                                                    "worker failed: ")
                                                    + exception.what();
                                            }
                                        } catch (...) {
                                            if (long_rpc_error.empty()) {
                                                long_rpc_error =
                                                    "long-lived HMX RPC "
                                                    "worker failed";
                                            }
                                        }
                                    }
                                    long_rpc_group_ready_emitted =
                                        published_heads == count;
                                    }
                                }
                            }
                            if (cpu_pack_overlap
                                && !wait_for_cpu_pack()) {
                                ok = false;
                                break;
                            }
                            if (!long_rpc_error.empty()) {
                                error = long_rpc_error;
                                ok = false;
                                break;
                            }
                            if (cpu_status != 0) {
                                error = bucketCallError(
                                    bucket,
                                    "hmx_qk_i8_operator_execute_cpu_packed_"
                                    "per_head_hn",
                                    cpu_status);
                                ok = false;
                                break;
                            }
                            if (dspTimingEnabled()
                                && last_dsp_timing_ != nullptr) {
                                DspTiming timing;
                                if (last_dsp_timing_(context, &timing) == 0
                                    && timing.version == 1
                                    && timing.struct_size == sizeof(timing)) {
                                    std::cerr
                                        << "[HMX_DSP_TIMING] layer="
                                        << request.layer_id
                                        << " key_len=" << request.key_len
                                        << " heads=" << count
                                        << " cpu_pack_overlap="
                                        << (cpu_pack_overlap ? 1 : 0)
                                        << " total_ticks="
                                        << timing.total_ticks
                                        << " resource_begin_ticks="
                                        << timing.resource_begin_ticks
                                        << " pipeline_ticks="
                                        << timing.pipeline_ticks
                                        << " key_pack_work_ticks="
                                        << timing.key_pack_work_ticks
                                        << " query_pack_work_ticks="
                                        << timing.query_pack_work_ticks
                                        << " hmx_kernel_ticks="
                                        << timing.hmx_kernel_ticks
                                        << " output_flush_ticks="
                                        << timing.output_flush_ticks
                                        << " resource_end_ticks="
                                        << timing.resource_end_ticks
                                        << std::endl;
                                }
                            }
                            if (query_begin == 0 && use_layer_key_cache) {
                                bool identity_full_group =
                                    execution_groups.size() == 1
                                    && count == validated.matrix_count
                                    && request.query_len <= bucket.info.m;
                                for (std::size_t local = 0;
                                     local < count && identity_full_group;
                                     ++local) {
                                    identity_full_group = group[local] == local;
                                }
                                if (deferred_cache_store != nullptr
                                    && !deferred_cache_store->valid()
                                    && identity_full_group
                                    && asyncLayerPackedKeyStoreEnabled()) {
                                    std::vector<const Half *> key_sources(
                                        request.key_heads,
                                        request.key_heads
                                            + validated.matrix_count);
                                    std::vector<std::size_t> matrix_indices(
                                        group, group + count);
                                    std::vector<float> cached_k_scales(
                                        per_head_k_scales.begin(),
                                        per_head_k_scales.end());
                                    const int cache_batch = request.batch;
                                    const int cache_heads = request.heads;
                                    const int cache_key_len = request.key_len;
                                    const int cache_head_dim = request.head_dim;
                                    const float cache_k_scale =
                                        bucket.info.k_scale;
                                    *deferred_cache_store =
                                        cpuPackTaskExecutor().submit(
                                            [this,
                                             key_sources = std::move(key_sources),
                                             matrix_indices =
                                                 std::move(matrix_indices),
                                             cached_k_scales =
                                                 std::move(cached_k_scales),
                                             cache_batch, cache_heads,
                                             cache_key_len, cache_head_dim,
                                             count, cache_k_scale, actual_n,
                                             key_begin, direct_key,
                                             direct_key_sums,
                                             profile_enabled]() mutable {
                                                Request cache_request;
                                                cache_request.batch =
                                                    cache_batch;
                                                cache_request.heads =
                                                    cache_heads;
                                                cache_request.key_len =
                                                    cache_key_len;
                                                cache_request.head_dim =
                                                    cache_head_dim;
                                                cache_request.key_heads =
                                                    key_sources.data();
                                                commitLayerPackedKeyGroup(
                                                    cache_request, count,
                                                    matrix_indices.data(),
                                                    cache_k_scale, actual_n,
                                                    key_begin, direct_key,
                                                    direct_key_sums,
                                                    profile_enabled,
                                                    cached_k_scales.data());
                                            });
                                } else {
                                    commitLayerPackedKeyGroup(
                                        request, count, group,
                                        bucket.info.k_scale, actual_n,
                                        key_begin, direct_key,
                                        direct_key_sums, profile_enabled,
                                        per_head_k_scales.data());
                                }
                            }
                            if (!long_rpc_pipeline) {
                                ScopedAttentionProfile output_profile(
                                    AttentionProfileStage::HMX_OUTPUT_LAYOUT,
                                    profile_enabled);
                                copyScoreBlock(
                                    0, count, query_begin, query_count,
                                    request.query_len, group, direct_scores,
                                    actual_n, request.key_len);
                            }
                            fused_prepare_pending = false;
                            continue;
                        }
                        const bool raw_query_already_staged =
                            scale_staged_group
                            && scale_staging.query_staged
                            && scale_staging.query_begin == query_begin
                            && scale_staging.query_count == query_count;
                        if (!raw_query_already_staged) {
                            ok = copyRawQueryGroup(
                                request, 0, count, query_begin, query_count,
                                profile_enabled, group, raw_query);
                        }
                        if (!ok) break;
                        if (dsp_int32_topk) {
                            const std::int32_t *offsets =
                                rowOffsetsForMatrix(request, group[0]);
                            const std::int32_t *chunk_offsets =
                                offsets + query_begin;
                            int status = 0;
                            {
                                ScopedAttentionProfile mm_profile(
                                    AttentionProfileStage::HMX_MM,
                                    profile_enabled);
                                status = bucket.execute_raw_i32_topk_hn(
                                    context, query_count, request.key_len,
                                    actual_n,
                                    validated.causal_prefix_tokens
                                        + query_begin,
                                    scales[group[0]].q, chunk_offsets,
                                    query_count + 1);
                            }
                            if (status != 0) {
                                error = bucketCallError(
                                    bucket,
                                    "hmx_qk_i8_operator_execute_raw_i32_topk_hn",
                                    status);
                                ok = false;
                                break;
                            }
                            const int index_count =
                                chunk_offsets[query_count] - chunk_offsets[0];
                            if (index_count < 0) {
                                error = "DSP INT32 Top-k received invalid row "
                                    "offsets";
                                ok = false;
                                break;
                            }
                            {
                                ScopedAttentionProfile output_profile(
                                    AttentionProfileStage::HMX_OUTPUT_LAYOUT,
                                    profile_enabled);
                                std::memcpy(
                                    request.topk_indices
                                        + group[0]
                                            * validated.topk_head_stride
                                        + chunk_offsets[0],
                                    direct_topk,
                                    static_cast<std::size_t>(index_count)
                                        * sizeof(std::int32_t));
                            }
                            if (bucketDiagnosticsEnabled()) {
                                std::cerr << "[HMX_INT32_TOPK] op="
                                          << (request.diagnostic_label.empty()
                                                  ? "unknown"
                                                  : request.diagnostic_label)
                                          << " layer=" << request.layer_id
                                          << " key_len=" << request.key_len
                                          << " matrix=" << group[0]
                                          << " query_begin=" << query_begin
                                          << " rows=" << query_count
                                          << " indices=" << index_count
                                          << " q_scale="
                                          << scales[group[0]].q
                                          << " k_scale="
                                          << scales[group[0]].k
                                          << std::endl;
                            }
                            continue;
                        }
                        int status = 0;
                        float effective_requant_scale
                            = dynamic_output_scale
                                  && adaptive_requant_cache_[group[0]] > 0.0F
                            ? adaptive_requant_cache_[group[0]]
                            : bucket.info.requant_scale;
                        int output_scale_attempts = 1;
                        int output_peak = 0;
                        bool output_scale_converged
                            = !dynamic_output_scale;
                        bool fused_call_failed = false;
                        if (dynamic_output_scale
                            && bucket.execute_raw_scaled_hn == nullptr) {
                            error = "dynamic INT8 output scaling requires "
                                "hmx_qk_i8_operator_execute_raw_scaled_hn; "
                                "bucket=" + bucket.path;
                            ok = false;
                            break;
                        }
                        if (dynamic_qk_scale
                            && bucket.execute_raw_dynamic_hn == nullptr) {
                            error = "dynamic INT8 Q scaling requires "
                                "hmx_qk_i8_operator_execute_raw_dynamic_hn; "
                                "bucket=" + bucket.path;
                            ok = false;
                            break;
                        }
                        for (int attempt = 0; attempt < 12; ++attempt) {
                            const bool first_fused_call =
                                fused_prepare_pending;
                            const bool use_fused_call =
                                execution_group.per_head_bucket_scales
                                || first_fused_call;
                            const int fused_key_begin = first_fused_call
                                ? key_begin : actual_n;
                            if (use_fused_call) {
                                ScopedAttentionProfile fused_profile(
                                    AttentionProfileStage::
                                        HMX_FUSED_PREPARE_EXECUTE,
                                    profile_enabled);
                                if (execution_group.per_head_bucket_scales) {
                                    if (bucket
                                            .prepare_execute_raw_per_head_requant_hn
                                        != nullptr) {
                                        status = bucket
                                            .prepare_execute_raw_per_head_requant_hn(
                                        context,
                                        static_cast<std::int32_t>(count),
                                        query_count, actual_n,
                                        fused_key_begin,
                                        per_head_q_scales.data(),
                                        per_head_k_scales.data(),
                                        per_head_requant_scales.data());
                                    } else {
                                        status = bucket
                                            .prepare_execute_raw_per_head_hn(
                                        context,
                                        static_cast<std::int32_t>(count),
                                        query_count, actual_n,
                                        fused_key_begin,
                                        per_head_q_scales.data(),
                                        per_head_k_scales.data(),
                                        bucket.info.output_scale);
                                    }
                                } else {
                                    status = bucket.prepare_execute_raw_hn(
                                        context,
                                        static_cast<std::int32_t>(count),
                                        query_count, actual_n,
                                        fused_key_begin);
                                }
                            } else {
                                ScopedAttentionProfile mm_profile(
                                    AttentionProfileStage::HMX_MM,
                                    profile_enabled);
                                status = dynamic_qk_scale
                                    ? bucket.execute_raw_dynamic_hn(
                                        context,
                                        static_cast<std::int32_t>(count),
                                        query_count, actual_n,
                                        scales[group[0]].q,
                                        effective_requant_scale)
                                    : dynamic_output_scale
                                    ? bucket.execute_raw_scaled_hn(
                                        context,
                                        static_cast<std::int32_t>(count),
                                        query_count, actual_n,
                                        effective_requant_scale)
                                    : bucket.execute_raw_hn(
                                        context,
                                        static_cast<std::int32_t>(count),
                                        query_count, actual_n);
                            }
                            if (use_fused_call && status == 0
                                && dspTimingEnabled()
                                && last_dsp_timing_ != nullptr) {
                                DspTiming timing;
                                if (last_dsp_timing_(context, &timing) == 0
                                    && timing.version == 1
                                    && timing.struct_size == sizeof(timing)) {
                                    std::cerr
                                        << "[HMX_DSP_TIMING] layer="
                                        << request.layer_id
                                        << " key_len=" << request.key_len
                                        << " heads=" << count
                                        << " total_ticks="
                                        << timing.total_ticks
                                        << " resource_begin_ticks="
                                        << timing.resource_begin_ticks
                                        << " pipeline_ticks="
                                        << timing.pipeline_ticks
                                        << " key_pack_work_ticks="
                                        << timing.key_pack_work_ticks
                                        << " query_pack_work_ticks="
                                        << timing.query_pack_work_ticks
                                        << " hmx_kernel_ticks="
                                        << timing.hmx_kernel_ticks
                                        << " output_flush_ticks="
                                        << timing.output_flush_ticks
                                        << " resource_end_ticks="
                                        << timing.resource_end_ticks
                                        << std::endl;
                                }
                            }
                            if (use_fused_call) {
                                fused_prepare_pending = false;
                                fused_call_failed = status != 0;
                                if (status == 0 && first_fused_call
                                    && use_layer_key_cache) {
                                    commitLayerPackedKeyGroup(
                                        request, count, group,
                                        bucket.info.k_scale, actual_n,
                                        key_begin, direct_key,
                                        direct_key_sums, profile_enabled,
                                        execution_group.per_head_bucket_scales
                                            ? per_head_k_scales.data()
                                            : nullptr);
                                } else if (status == 0 && first_fused_call) {
                                    commitRawKey(
                                        bucket.context_key, request, 0,
                                        count, group, actual_n);
                                } else {
                                    invalidateRawKey(bucket.context_key);
                                }
                            }
                            if (status != 0 || !dynamic_output_scale) break;
                            output_scale_attempts = attempt + 1;
                            output_peak = 0;
                            const std::size_t head_stride
                                = static_cast<std::size_t>(bucket.info.m)
                                    * actual_n;
                            for (std::size_t local = 0; local < count; ++local) {
                                const std::int8_t *head_scores = direct_scores
                                    + local * head_stride;
                                for (int row = 0; row < query_count; ++row) {
                                    const std::int8_t *score_row = head_scores
                                        + static_cast<std::size_t>(row)
                                            * actual_n;
                                    for (int column = 0;
                                         column < request.key_len; ++column) {
                                        const int value = score_row[column];
                                        output_peak = std::max(
                                            output_peak,
                                            value == -128 ? 128
                                                : std::abs(value));
                                    }
                                }
                            }
                            if (output_peak >= 127) {
                                effective_requant_scale *= 0.25F;
                                continue;
                            }
                            if (output_peak < 96) {
                                const float multiplier = output_peak == 0
                                    ? 4.0F
                                    : std::min(
                                        8.0F, 112.0F
                                            / static_cast<float>(output_peak));
                                effective_requant_scale *= multiplier;
                                continue;
                            }
                            output_scale_converged = true;
                            break;
                        }
                        if (status != 0) {
                            error = bucketCallError(
                                bucket,
                                fused_call_failed
                                    ? "hmx_qk_i8_operator_prepare_execute_raw_hn"
                                    : "hmx_qk_i8_operator_matmul",
                                status);
                            ok = false;
                            break;
                        }
                        if (dynamic_output_scale
                            && !output_scale_converged) {
                            error = "dynamic INT8 output scale did not "
                                "converge without saturation; bucket="
                                + bucket.path;
                            ok = false;
                            break;
                        }
                        if (dynamic_output_scale) {
                            adaptive_requant_cache_[group[0]]
                                = effective_requant_scale;
                            for (std::size_t local = 0; local < count; ++local) {
                                const std::size_t matrix = group[local];
                                for (int row = 0; row < query_count; ++row) {
                                    score_requant_scales_[
                                        matrix * request.query_len
                                        + query_begin + row]
                                        = effective_requant_scale;
                                }
                            }
                            if (bucketDiagnosticsEnabled()) {
                                std::cerr << "[HMX_INT8_OUTPUT_SCALE] op="
                                          << (request.diagnostic_label.empty()
                                                  ? "unknown"
                                                  : request.diagnostic_label)
                                          << " layer=" << request.layer_id
                                          << " key_len=" << request.key_len
                                          << " matrix=" << group[0]
                                          << " query_begin=" << query_begin
                                          << " requant="
                                          << effective_requant_scale
                                          << " output="
                                          << ((dynamic_qk_scale
                                                  ? scales[group[0]].q
                                                  : bucket.info.q_scale)
                                              * (dynamic_qk_scale
                                                  ? scales[group[0]].k
                                                  : bucket.info.k_scale)
                                              / effective_requant_scale)
                                          << " q_scale="
                                          << (dynamic_qk_scale
                                                  ? scales[group[0]].q
                                                  : bucket.info.q_scale)
                                          << " k_scale="
                                          << (dynamic_qk_scale
                                                  ? scales[group[0]].k
                                                  : bucket.info.k_scale)
                                          << " peak=" << output_peak
                                          << " attempts="
                                          << output_scale_attempts
                                          << std::endl;
                            }
                        }
                        ScopedAttentionProfile output_profile(
                            AttentionProfileStage::HMX_OUTPUT_LAYOUT,
                            profile_enabled);
                        copyScoreBlock(0, count, query_begin, query_count,
                                       request.query_len, group,
                                       direct_scores, actual_n,
                                       request.key_len);
                    }
                    if (local_scope_active && ok) {
                        int end_status = 0;
                        {
                            ScopedAttentionProfile end_profile(
                                AttentionProfileStage::HMX_SCOPE_END,
                                profile_enabled);
                            end_status = bucket.end(context);
                        }
                        if (end_status != 0) {
                            error = bucketCallError(
                                bucket, "hmx_qk_i8_operator_end", end_status);
                            ok = false;
                        }
                    } else if (local_scope_active) {
                        ScopedAttentionProfile end_profile(
                            AttentionProfileStage::HMX_SCOPE_END,
                            profile_enabled);
                        (void)bucket.end(context);
                    }
                    if (ok && dsp_int32_topk) {
                        topk_preselected_[group[0]] = 1;
                    }
                    if (ok && hmx_pipeline::LatencyRecorder::enabled()) {
                        const PendingNpuLatency latency{
                            static_cast<int>(count),
                            execution_group.per_head_bucket_scales
                                ? per_head_q_scales.front()
                                : bucket.info.q_scale,
                            execution_group.per_head_bucket_scales
                                ? per_head_k_scales.front()
                                : bucket.info.k_scale,
                            direct_rpc_elapsed_for_profile_ns != 0
                                ? direct_rpc_elapsed_for_profile_ns
                                : CPUAttentionProfiler::nowNs()
                                      - group_begin_ns};
                        if (attention_execution_scope) {
                            pending_npu_latencies.push_back(latency);
                        } else {
                            hmx_pipeline::LatencyRecorder::npu(
                                request.key_len, latency.heads,
                                latency.q_scale, latency.k_scale,
                                latency.elapsed_ns);
                        }
                    }
                    if (ok && on_group_ready
                        && !long_rpc_group_ready_emitted) {
                        const std::size_t requested_ready_heads =
                            pipelineReadyGroupHeads();
                        for (std::size_t local = 0; local < count;) {
                            std::size_t ready_count = 1;
                            while (ready_count < requested_ready_heads
                                   && local + ready_count < count
                                   && group[local + ready_count]
                                       == group[local] + ready_count) {
                                ++ready_count;
                            }
                            on_group_ready({group[local], ready_count});
                            local += ready_count;
                        }
                    }
                    offset += count;
                }
            }
            if (!ok) return false;
        }
        if (shared_scope.active) {
            int end_status = 0;
            const uint64_t scope_end_begin_ns =
                hmx_pipeline::LatencyRecorder::enabled()
                ? CPUAttentionProfiler::nowNs() : 0;
            {
                ScopedAttentionProfile end_profile(
                    AttentionProfileStage::HMX_SCOPE_END,
                    profile_enabled);
                end_status = shared_scope.close();
            }
            if (end_status != 0) {
                error = shared_scope_bucket == nullptr
                    ? "INT8 HMX attention execution scope end failed"
                    : bucketCallError(
                        *shared_scope_bucket,
                        "hmx_qk_i8_operator_end", end_status);
                return false;
            }
            if (!pending_npu_latencies.empty()) {
                pending_npu_latencies.back().elapsed_ns +=
                    CPUAttentionProfiler::nowNs() - scope_end_begin_ns;
            }
        }
        for (const PendingNpuLatency &latency : pending_npu_latencies) {
            hmx_pipeline::LatencyRecorder::npu(
                request.key_len, latency.heads, latency.q_scale,
                latency.k_scale, latency.elapsed_ns);
        }
        operator_lock.unlock();
        for (const std::size_t matrix : fallback_matrices) {
            if (!selectExactFloatTopKMatrix(
                    request, validated, matrix, profile_enabled)) {
                error = "CPU exact Top-k fallback failed for out-of-range "
                    "INT8 HMX bucket scale";
                return false;
            }
            if (on_group_ready) on_group_ready({matrix, 1});
        }
        return true;
    }

    bool copyRawQueryGroup(const Request &request, std::size_t begin,
                           std::size_t count, int query_begin,
                           int query_count, bool profile_enabled,
                           const std::size_t *matrix_indices,
                           float *output) {
        ScopedAttentionProfile profile(
            AttentionProfileStage::HMX_Q_LAYOUT, profile_enabled);
        if (output == nullptr) return false;
        const std::size_t source_stride = request.query_f32_row_stride == 0
            ? static_cast<std::size_t>(request.head_dim)
            : request.query_f32_row_stride;
        const std::size_t destination_head_stride
            = static_cast<std::size_t>(info_.m) * info_.k;
        for (std::size_t local = 0; local < count; ++local) {
            const std::size_t matrix = matrix_indices == nullptr
                ? begin + local : matrix_indices[local];
            float *destination = output + local * destination_head_stride;
            std::fill(destination, destination + destination_head_stride,
                      0.0F);
            const float *source = request.query_f32_heads[matrix]
                + static_cast<std::size_t>(query_begin) * source_stride;
            for (int row = 0; row < query_count; ++row) {
                std::memcpy(destination + static_cast<std::size_t>(row) * info_.k,
                            source + static_cast<std::size_t>(row) * source_stride,
                            static_cast<std::size_t>(info_.k) * sizeof(float));
            }
        }
        return true;
    }

    bool copyRawKeyGroup(const Request &request, std::size_t begin,
                         std::size_t count, bool profile_enabled,
                         const std::size_t *matrix_indices, Half *output,
                         int destination_n, int source_token_begin = 0) {
        ScopedAttentionProfile profile(
            AttentionProfileStage::HMX_K_LAYOUT, profile_enabled);
        if (output == nullptr || destination_n < request.key_len ||
            source_token_begin < 0 || source_token_begin > request.key_len) {
            return false;
        }
        const std::size_t destination_head_stride
            = static_cast<std::size_t>(destination_n) * info_.k;
        const std::size_t copy_begin
            = static_cast<std::size_t>(source_token_begin) * info_.k;
        const std::size_t copy_end
            = static_cast<std::size_t>(request.key_len) * info_.k;
        for (std::size_t local = 0; local < count; ++local) {
            const std::size_t matrix = matrix_indices == nullptr
                ? begin + local : matrix_indices[local];
            Half *destination = output + local * destination_head_stride;
            std::memcpy(destination + copy_begin,
                        request.key_heads[matrix] + copy_begin,
                        (copy_end - copy_begin) * sizeof(Half));
            std::fill(destination + copy_end,
                      destination + destination_head_stride, Half{0});
        }
        return true;
    }

    template<typename Function>
    static void forEachGroupMatrix(std::int64_t count, int thread_count,
                                   bool allow_openmp,
                                   Function &&function) {
#if defined(_OPENMP)
        if (allow_openmp && thread_count > 1) {
#pragma omp parallel for schedule(static) num_threads(thread_count)
            for (std::int64_t local = 0; local < count; ++local) {
                function(local);
            }
            return;
        }
#else
        (void)thread_count;
        (void)allow_openmp;
#endif
        for (std::int64_t local = 0; local < count; ++local) {
            function(local);
        }
    }

#if defined(__aarch64__)
    static void cpuStageSpinPause() noexcept {
        asm volatile("yield" ::: "memory");
    }

    static bool cpuMaxAbsScaleF32(
        const float *source, int rows, std::size_t stride, int columns,
        float &scale) noexcept {
        if (source == nullptr || rows <= 0 || columns <= 0
            || stride < static_cast<std::size_t>(columns)) {
            return false;
        }
        const uint32x4_t abs_mask = vdupq_n_u32(UINT32_C(0x7fffffff));
        uint32x4_t vector_maximum = vdupq_n_u32(0);
        std::uint32_t tail_maximum = 0;
        for (int row = 0; row < rows; ++row) {
            const float *values = source
                + static_cast<std::size_t>(row) * stride;
            int column = 0;
            for (; column + 4 <= columns; column += 4) {
                const uint32x4_t bits = vandq_u32(
                    vreinterpretq_u32_f32(vld1q_f32(values + column)),
                    abs_mask);
                vector_maximum = vmaxq_u32(vector_maximum, bits);
            }
            for (; column < columns; ++column) {
                std::uint32_t bits = 0;
                std::memcpy(&bits, values + column, sizeof(bits));
                bits &= UINT32_C(0x7fffffff);
                tail_maximum = std::max(tail_maximum, bits);
            }
        }
        const std::uint32_t maximum_bits = std::max(
            tail_maximum, vmaxvq_u32(vector_maximum));
        if (maximum_bits >= UINT32_C(0x7f800000)) return false;
        float maximum = 0.0F;
        std::memcpy(&maximum, &maximum_bits, sizeof(maximum));
        scale = maximum > 0.0F
            ? maximum * (1.0F / 127.0F)
            : std::numeric_limits<float>::min();
        return true;
    }

    static bool cpuMaxAbsScaleF16(
        const Half *source, int rows, std::size_t stride, int columns,
        float &scale) noexcept {
        if (source == nullptr || rows <= 0 || columns <= 0
            || stride < static_cast<std::size_t>(columns)) {
            return false;
        }
        const uint16x8_t abs_mask = vdupq_n_u16(UINT16_C(0x7fff));
        uint16x8_t vector_maximum = vdupq_n_u16(0);
        std::uint16_t tail_maximum = 0;
        for (int row = 0; row < rows; ++row) {
            const Half *values = source
                + static_cast<std::size_t>(row) * stride;
            int column = 0;
            for (; column + 8 <= columns; column += 8) {
                const uint16x8_t bits = vandq_u16(
                    vld1q_u16(values + column), abs_mask);
                vector_maximum = vmaxq_u16(vector_maximum, bits);
            }
            for (; column < columns; ++column) {
                const std::uint16_t bits = values[column]
                    & UINT16_C(0x7fff);
                tail_maximum = std::max(tail_maximum, bits);
            }
        }
        const std::uint16_t maximum_bits = std::max(
            tail_maximum, vmaxvq_u16(vector_maximum));
        if (maximum_bits >= UINT16_C(0x7c00)) return false;
        const float maximum = halfToFloat(maximum_bits);
        scale = maximum > 0.0F
            ? maximum * (1.0F / 127.0F)
            : std::numeric_limits<float>::min();
        return true;
    }

    static int8x16_t quantizeF32x16Neon(
        const float *source, float inverse_scale) noexcept {
        const float32x4_t scale = vdupq_n_f32(inverse_scale);
        const float32x4_t minimum = vdupq_n_f32(-127.0F);
        const float32x4_t maximum = vdupq_n_f32(127.0F);
        int32x4_t quantized[4];
        for (int index = 0; index < 4; ++index) {
            float32x4_t value = vmulq_f32(
                vld1q_f32(source + index * 4), scale);
            value = vmaxq_f32(minimum, vminq_f32(maximum, value));
            quantized[index] = vcvtnq_s32_f32(value);
        }
        const int16x8_t low = vcombine_s16(
            vqmovn_s32(quantized[0]), vqmovn_s32(quantized[1]));
        const int16x8_t high = vcombine_s16(
            vqmovn_s32(quantized[2]), vqmovn_s32(quantized[3]));
        return vcombine_s8(vqmovn_s16(low), vqmovn_s16(high));
    }

    static std::uint32_t quantizeF16x4Neon(
        const Half *source, float inverse_scale,
        std::int32_t &sum) noexcept {
        float32x4_t value = vcvt_f32_f16(
            vreinterpret_f16_u16(vld1_u16(source)));
        value = vmulq_n_f32(value, inverse_scale);
        value = vmaxq_f32(
            vdupq_n_f32(-127.0F),
            vminq_f32(vdupq_n_f32(127.0F), value));
        const int16x4_t narrowed = vqmovn_s32(vcvtnq_s32_f32(value));
        sum += vaddv_s16(narrowed);
        const int8x8_t packed = vqmovn_s16(
            vcombine_s16(narrowed, vdup_n_s16(0)));
        return vget_lane_u32(vreinterpret_u32_s8(packed), 0);
    }
#endif

#if !defined(__aarch64__)
    static void cpuStageSpinPause() noexcept {
        std::this_thread::yield();
    }

    static bool cpuMaxAbsScaleF32(
        const float *source, int rows, std::size_t stride, int columns,
        float &scale) noexcept {
        if (source == nullptr || rows <= 0 || columns <= 0
            || stride < static_cast<std::size_t>(columns)) {
            return false;
        }
        std::uint32_t maximum_bits = 0;
        for (int row = 0; row < rows; ++row) {
            const float *values = source
                + static_cast<std::size_t>(row) * stride;
            for (int column = 0; column < columns; ++column) {
                std::uint32_t bits = 0;
                std::memcpy(&bits, values + column, sizeof(bits));
                bits &= UINT32_C(0x7fffffff);
                if (bits >= UINT32_C(0x7f800000)) return false;
                maximum_bits = std::max(maximum_bits, bits);
            }
        }
        float maximum = 0.0F;
        std::memcpy(&maximum, &maximum_bits, sizeof(maximum));
        scale = maximum > 0.0F
            ? maximum * (1.0F / 127.0F)
            : std::numeric_limits<float>::min();
        return true;
    }

    static bool cpuMaxAbsScaleF16(
        const Half *source, int rows, std::size_t stride, int columns,
        float &scale) noexcept {
        if (source == nullptr || rows <= 0 || columns <= 0
            || stride < static_cast<std::size_t>(columns)) {
            return false;
        }
        std::uint16_t maximum_bits = 0;
        for (int row = 0; row < rows; ++row) {
            const Half *values = source
                + static_cast<std::size_t>(row) * stride;
            for (int column = 0; column < columns; ++column) {
                const std::uint16_t bits = values[column]
                    & UINT16_C(0x7fff);
                if (bits >= UINT16_C(0x7c00)) return false;
                maximum_bits = std::max(maximum_bits, bits);
            }
        }
        const float maximum = halfToFloat(maximum_bits);
        scale = maximum > 0.0F
            ? maximum * (1.0F / 127.0F)
            : std::numeric_limits<float>::min();
        return true;
    }
#endif

    static void cpuPackQueryHeadTile(
        const Request &request, std::size_t matrix, int query_begin,
        int query_count, float q_scale, std::int8_t *destination,
        int row_tile, int k) noexcept {
        const std::size_t source_stride = request.query_f32_row_stride == 0
            ? static_cast<std::size_t>(request.head_dim)
            : request.query_f32_row_stride;
        const float *source = request.query_f32_heads[matrix];
        const float inverse_scale = 1.0F / q_scale;
        const int k_tiles = k / 32;
        for (int inner_tile = 0; inner_tile < k_tiles; ++inner_tile) {
            std::uint8_t *tile =
                reinterpret_cast<std::uint8_t *>(destination)
                + (static_cast<std::size_t>(row_tile) * k_tiles
                   + inner_tile) * 2048u;
            for (int row = 0; row < 64; ++row) {
                const int source_row = row_tile * 64 + row;
                std::uint8_t *packed = tile
                    + static_cast<std::size_t>(row) * 32u;
                if (source_row >= query_count) {
                    std::memset(packed, 0x80, 32u);
                    continue;
                }
                const float *values = source
                    + static_cast<std::size_t>(query_begin + source_row)
                        * source_stride
                    + inner_tile * 32;
#if defined(__aarch64__)
                const uint8x16_t sign = vdupq_n_u8(0x80u);
                vst1q_u8(
                    packed,
                    veorq_u8(
                        vreinterpretq_u8_s8(
                            quantizeF32x16Neon(values, inverse_scale)),
                        sign));
                vst1q_u8(
                    packed + 16,
                    veorq_u8(
                        vreinterpretq_u8_s8(
                            quantizeF32x16Neon(
                                values + 16, inverse_scale)),
                        sign));
#else
                for (int inner = 0; inner < 32; ++inner) {
                    packed[inner] = static_cast<std::uint8_t>(
                        quantizeWithInverse(values[inner], inverse_scale))
                        ^ 0x80u;
                }
#endif
            }
        }
    }

    static void cpuPackQueryHead(
        const Request &request, std::size_t matrix, int query_begin,
        int query_count, float q_scale, std::int8_t *destination,
        int padded_m, int k) noexcept {
        for (int row_tile = 0; row_tile < padded_m / 64; ++row_tile) {
            cpuPackQueryHeadTile(
                request, matrix, query_begin, query_count, q_scale,
                destination, row_tile, k);
        }
    }

    static void cpuPackKeyTokenRange(
        const Request &request, std::size_t matrix, int token_begin,
        int token_end, float k_scale, std::int8_t *destination,
        std::int32_t *sums, int k) noexcept {
        const Half *source = request.key_heads[matrix];
        const float inverse_scale = 1.0F / k_scale;
        const int k_tiles = k / 32;
        for (int token = token_begin; token < token_end; ++token) {
            std::int32_t sum = 0;
            const int token_tile = token / 32;
            const int token_in_tile = token % 32;
            for (int inner_tile = 0; inner_tile < k_tiles; ++inner_tile) {
                std::int8_t *tile = destination
                    + (static_cast<std::size_t>(token_tile) * k_tiles
                       + inner_tile) * 1024u;
                for (int inner_group = 0; inner_group < 8; ++inner_group) {
                    std::int8_t *packed = tile
                        + static_cast<std::size_t>(inner_group) * 128u
                        + token_in_tile * 4;
                    if (token >= request.key_len) {
                        std::memset(packed, 0, 4u);
                        continue;
                    }
                    const Half *values = source
                        + static_cast<std::size_t>(token) * k
                        + inner_tile * 32 + inner_group * 4;
#if defined(__aarch64__)
                    const std::uint32_t word = quantizeF16x4Neon(
                        values, inverse_scale, sum);
                    std::memcpy(packed, &word, sizeof(word));
#else
                    for (int lane = 0; lane < 4; ++lane) {
                        packed[lane] = quantizeWithInverse(
                            halfToFloat(values[lane]), inverse_scale);
                        sum += packed[lane];
                    }
#endif
                }
            }
            sums[token] = sum;
        }
    }

    static void cpuPackKeyHead(
        const Request &request, std::size_t matrix, int key_begin,
        int padded_n, float k_scale, std::int8_t *destination,
        std::int32_t *sums, int k) noexcept {
        cpuPackKeyTokenRange(
            request, matrix, (key_begin / 32) * 32, padded_n, k_scale,
            destination, sums, k);
    }

    static void publishCpuPackProgress(
        volatile std::int32_t *ready, int query_tiles,
        int key_tiles) noexcept {
        constexpr std::int32_t stream_flag = 0x40000000;
        constexpr int query_shift = 16;
        constexpr int query_mask = 0xff;
        constexpr int key_mask = 0xffff;
        const std::int32_t value = stream_flag
            | ((query_tiles & query_mask) << query_shift)
            | (key_tiles & key_mask);
        std::atomic_thread_fence(std::memory_order_release);
        __atomic_store_n(ready, value, __ATOMIC_RELEASE);
    }

    static bool cpuPackPerHeadGroup(
        const Request &request, std::size_t count,
        const std::size_t *matrix_indices, int query_begin, int query_count,
        int key_begin, int padded_n, const float *q_scales,
        const float *k_scales, std::int8_t *direct_query,
        std::int8_t *direct_key, std::int32_t *direct_sums,
        volatile std::int32_t *ready, std::size_t ready_stride,
        int padded_m, int k,
        bool profile_enabled) noexcept {
        if (matrix_indices == nullptr || q_scales == nullptr
            || k_scales == nullptr || direct_query == nullptr
            || direct_key == nullptr || direct_sums == nullptr
            || ready == nullptr || ready_stride == 0) {
            return false;
        }
        const std::size_t query_stride =
            static_cast<std::size_t>(padded_m) * k;
        const std::size_t key_stride =
            static_cast<std::size_t>(padded_n) * k;
        const bool intra_head = cpuPackIntraHeadEnabled();
        for (std::size_t local = 0; local < count; ++local) {
            if (intra_head) {
                const int query_tiles = padded_m / 64;
                const int key_tiles = padded_n / 32;
                int ready_query_tiles = 0;
                int ready_key_tiles = (key_begin / 32);
                std::uint64_t query_ns = 0;
                std::uint64_t key_ns = 0;
                std::int8_t *query_destination =
                    direct_query + local * query_stride;
                std::int8_t *key_destination =
                    direct_key + local * key_stride;
                std::int32_t *sum_destination = direct_sums
                    + local * static_cast<std::size_t>(padded_n);

                if (query_tiles > 0) {
                    const std::uint64_t begin = profile_enabled
                        ? CPUAttentionProfiler::nowNs() : 0;
                    cpuPackQueryHeadTile(
                        request, matrix_indices[local], query_begin,
                        query_count, q_scales[local], query_destination,
                        0, k);
                    if (profile_enabled) {
                        query_ns += CPUAttentionProfiler::nowNs() - begin;
                    }
                    ready_query_tiles = 1;
                    publishCpuPackProgress(
                        ready + local * ready_stride,
                        ready_query_tiles, ready_key_tiles);
                }

                const int first_chunk_end = std::min(
                    key_tiles, ready_key_tiles + 8);
                if (ready_key_tiles < first_chunk_end) {
                    const std::uint64_t begin = profile_enabled
                        ? CPUAttentionProfiler::nowNs() : 0;
                    cpuPackKeyTokenRange(
                        request, matrix_indices[local],
                        ready_key_tiles * 32, first_chunk_end * 32,
                        k_scales[local], key_destination, sum_destination, k);
                    if (profile_enabled) {
                        key_ns += CPUAttentionProfiler::nowNs() - begin;
                    }
                    ready_key_tiles = first_chunk_end;
                    publishCpuPackProgress(
                        ready + local * ready_stride,
                        ready_query_tiles, ready_key_tiles);
                }

                for (int row_tile = 1; row_tile < query_tiles; ++row_tile) {
                    const std::uint64_t begin = profile_enabled
                        ? CPUAttentionProfiler::nowNs() : 0;
                    cpuPackQueryHeadTile(
                        request, matrix_indices[local], query_begin,
                        query_count, q_scales[local], query_destination,
                        row_tile, k);
                    if (profile_enabled) {
                        query_ns += CPUAttentionProfiler::nowNs() - begin;
                    }
                    ready_query_tiles = row_tile + 1;
                    publishCpuPackProgress(
                        ready + local * ready_stride,
                        ready_query_tiles, ready_key_tiles);
                }

                while (ready_key_tiles < key_tiles) {
                    const int chunk_end = std::min(
                        key_tiles, ready_key_tiles + 8);
                    const std::uint64_t begin = profile_enabled
                        ? CPUAttentionProfiler::nowNs() : 0;
                    cpuPackKeyTokenRange(
                        request, matrix_indices[local],
                        ready_key_tiles * 32, chunk_end * 32,
                        k_scales[local], key_destination, sum_destination, k);
                    if (profile_enabled) {
                        key_ns += CPUAttentionProfiler::nowNs() - begin;
                    }
                    ready_key_tiles = chunk_end;
                    publishCpuPackProgress(
                        ready + local * ready_stride,
                        ready_query_tiles, ready_key_tiles);
                }
                if (profile_enabled) {
                    CPUAttentionProfiler::add(
                        AttentionProfileStage::HMX_Q_LAYOUT, query_ns);
                    CPUAttentionProfiler::add(
                        AttentionProfileStage::HMX_K_LAYOUT, key_ns);
                }
                continue;
            }
            const std::uint64_t key_start = profile_enabled
                ? CPUAttentionProfiler::nowNs() : 0;
            cpuPackKeyHead(
                request, matrix_indices[local], key_begin, padded_n,
                k_scales[local], direct_key + local * key_stride,
                direct_sums + local * static_cast<std::size_t>(padded_n), k);
            if (profile_enabled) {
                CPUAttentionProfiler::add(
                    AttentionProfileStage::HMX_K_LAYOUT,
                    CPUAttentionProfiler::nowNs() - key_start);
            }
            const std::uint64_t query_start = profile_enabled
                ? CPUAttentionProfiler::nowNs() : 0;
            cpuPackQueryHead(
                request, matrix_indices[local], query_begin, query_count,
                q_scales[local], direct_query + local * query_stride,
                padded_m, k);
            if (profile_enabled) {
                CPUAttentionProfiler::add(
                    AttentionProfileStage::HMX_Q_LAYOUT,
                    CPUAttentionProfiler::nowNs() - query_start);
            }
            std::atomic_thread_fence(std::memory_order_release);
            __atomic_store_n(
                ready + local * ready_stride, 1, __ATOMIC_RELEASE);
        }
        return true;
    }

    bool quantizeQueryGroup(const Request &request, std::size_t begin,
                            std::size_t count, int query_begin,
                            int query_count, bool profile_enabled,
                            float q_scale,
                            const std::size_t *matrix_indices = nullptr,
                            std::int8_t *direct_output = nullptr) {
        ScopedAttentionProfile profile(
            AttentionProfileStage::HMX_Q_LAYOUT, profile_enabled);
        std::int8_t *output = direct_output == nullptr
            ? query_i8_.data() : direct_output;
        const std::size_t source_stride = request.query_f32_row_stride == 0
            ? static_cast<std::size_t>(request.head_dim)
            : request.query_f32_row_stride;
        const std::size_t destination_head_stride
            = static_cast<std::size_t>(info_.m) * info_.k;
        if (direct_output == nullptr) {
            std::fill(output, output + count * destination_head_stride,
                      std::int8_t{0});
        } else if (query_count < info_.m) {
            for (std::size_t local = 0; local < count; ++local) {
                std::fill(output + local * destination_head_stride,
                          output + (local + 1) * destination_head_stride,
                          std::int8_t{0});
            }
        }
        std::atomic<bool> finite{true};
        const std::int64_t local_count = static_cast<std::int64_t>(count);
        forEachGroupMatrix(
            local_count, request.thread_count,
            request.pipeline_group_heads == 0,
            [&](std::int64_t local) {
            const std::size_t matrix = matrix_indices == nullptr
                ? begin + static_cast<std::size_t>(local)
                : matrix_indices[local];
            const float *source = request.query_f32_heads[matrix];
            std::int8_t *destination = output
                + static_cast<std::size_t>(local) * destination_head_stride;
            const float inverse_scale = 1.0F / q_scale;
            if (direct_output != nullptr) {
                std::uint8_t *packed =
                    reinterpret_cast<std::uint8_t *>(destination);
                const int k_tiles = info_.k / 32;
                for (int row_tile = 0; row_tile < info_.m / 64; ++row_tile) {
                    for (int inner_tile = 0; inner_tile < k_tiles;
                         ++inner_tile) {
                        std::uint8_t *tile = packed
                            + (static_cast<std::size_t>(row_tile) * k_tiles
                               + inner_tile) * 2048u;
                        for (int row = 0; row < 64; ++row) {
                            for (int inner = 0; inner < 32; ++inner) {
                                const int source_row = row_tile * 64 + row;
                                const int source_inner = inner_tile * 32 + inner;
                                std::int8_t quantized = 0;
                                if (source_row < query_count) {
                                    const float value = source[
                                        static_cast<std::size_t>(
                                            query_begin + source_row)
                                            * source_stride + source_inner];
                                    if (!std::isfinite(value)) finite.store(false);
                                    quantized = quantizeWithInverse(
                                        value, inverse_scale);
                                }
                                tile[static_cast<std::size_t>(row) * 32
                                     + inner]
                                    = static_cast<std::uint8_t>(quantized)
                                      ^ 0x80u;
                            }
                        }
                    }
                }
                return;
            }
            for (int row = 0; row < query_count; ++row) {
                for (int inner = 0; inner < info_.k; ++inner) {
                    const float value = source[
                        static_cast<std::size_t>(query_begin + row)
                            * source_stride + inner];
                    if (!std::isfinite(value)) finite.store(false);
                    destination[static_cast<std::size_t>(row) * info_.k
                                + inner] = quantizeWithInverse(
                                    value, inverse_scale);
                }
            }
        });
        return finite.load();
    }

    bool quantizeKeyGroup(const Request &request, std::size_t begin,
                          std::size_t count, bool profile_enabled,
                          float k_scale,
                          const std::size_t *matrix_indices = nullptr,
                          std::int8_t *direct_output = nullptr,
                          int direct_n = 0,
                          std::int32_t *direct_sums = nullptr) {
        ScopedAttentionProfile profile(
            AttentionProfileStage::HMX_K_LAYOUT, profile_enabled);
        std::int8_t *output = direct_output == nullptr
            ? key_i8_.data() : direct_output;
        if (direct_output != nullptr && direct_sums == nullptr) return false;
        const int destination_n = direct_output == nullptr
            ? info_.n : direct_n;
        const std::size_t destination_head_stride
            = static_cast<std::size_t>(destination_n) * info_.k;
        if (direct_output == nullptr) {
            std::fill(output, output + count * destination_head_stride,
                      std::int8_t{0});
        }
        std::atomic<bool> finite{true};
        if (packed_key_cache_.size()
            < static_cast<std::size_t>(request.batch * request.heads)) {
            packed_key_cache_.resize(
                static_cast<std::size_t>(request.batch * request.heads));
        }
        const std::int64_t local_count = static_cast<std::int64_t>(count);
        forEachGroupMatrix(
            local_count, request.thread_count,
            request.pipeline_group_heads == 0,
            [&](std::int64_t local) {
            const std::size_t matrix = matrix_indices == nullptr
                ? begin + static_cast<std::size_t>(local)
                : matrix_indices[local];
            const Half *source = request.key_heads[matrix];
            std::int8_t *destination = output
                + static_cast<std::size_t>(local) * destination_head_stride;
            std::int32_t *sums = direct_sums == nullptr ? nullptr
                : direct_sums + static_cast<std::size_t>(local) * destination_n;
            const float inverse_scale = 1.0F / k_scale;
            if (direct_output == nullptr) {
                const std::size_t elements
                    = static_cast<std::size_t>(request.key_len) * info_.k;
                for (std::size_t element = 0; element < elements; ++element) {
                    const float value = halfToFloat(source[element]);
                    if (!std::isfinite(value)) finite.store(false);
                    destination[element] = quantizeWithInverse(
                        value, inverse_scale);
                }
                return;
            }
            const int k_tiles = info_.k / 32;
            PackedKeyCache &cache = packed_key_cache_[matrix];
            if (cache.source != source || cache.scale != k_scale
                || request.key_len < cache.tokens) {
                std::fill(cache.packed.begin(), cache.packed.end(),
                          std::int8_t{0});
                std::fill(cache.sums.begin(), cache.sums.end(),
                          std::int32_t{0});
                cache.tokens = 0;
                cache.source = source;
                cache.scale = k_scale;
            }
            const std::size_t packed_elements =
                static_cast<std::size_t>(destination_n) * info_.k;
            cache.packed.resize(packed_elements, std::int8_t{0});
            cache.sums.resize(destination_n, std::int32_t{0});
            const int first_token = cache.tokens;
            const int first_token_tile = first_token / 32;
            for (int token_tile = first_token_tile;
                 token_tile < destination_n / 32;
                 ++token_tile) {
                for (int inner_tile = 0; inner_tile < k_tiles; ++inner_tile) {
                    std::int8_t *tile = cache.packed.data()
                        + (static_cast<std::size_t>(token_tile) * k_tiles
                           + inner_tile) * 1024u;
                    for (int inner_group = 0; inner_group < 8;
                         ++inner_group) {
                        for (int token_in_tile = 0; token_in_tile < 32;
                             ++token_in_tile) {
                            const int token = token_tile * 32 + token_in_tile;
                            std::int8_t *packed = tile
                                + static_cast<std::size_t>(inner_group) * 128u
                                + token_in_tile * 4;
                            for (int lane = 0; lane < 4; ++lane) {
                                const int inner = inner_tile * 32
                                    + inner_group * 4 + lane;
                                if (token >= first_token
                                    && token < request.key_len) {
                                    const float value = halfToFloat(source[
                                        static_cast<std::size_t>(token)
                                            * info_.k + inner]);
                                    if (!std::isfinite(value)) finite.store(false);
                                    const std::int8_t quantized =
                                        quantizeWithInverse(
                                        value, inverse_scale);
                                    cache.sums[token] += quantized;
                                    packed[lane] = quantized;
                                }
                            }
                        }
                    }
                }
            }
            cache.tokens = request.key_len;
            std::memcpy(destination, cache.packed.data(),
                        static_cast<std::size_t>(destination_n) * info_.k);
            std::memcpy(sums, cache.sums.data(),
                        static_cast<std::size_t>(destination_n)
                            * sizeof(std::int32_t));
        });
        return finite.load();
    }

    void copyScoreBlock(std::size_t matrix_begin, std::size_t matrix_count,
                        int query_begin, int query_count,
                        int query_len,
                        const std::size_t *matrix_indices = nullptr,
                        const std::int8_t *direct_scores = nullptr,
                        int direct_n = 0,
                        int copy_columns = 0) noexcept {
        const std::size_t source_row_stride = direct_scores == nullptr
            ? static_cast<std::size_t>(info_.n)
            : static_cast<std::size_t>(direct_n);
        const std::size_t source_head_stride
            = static_cast<std::size_t>(info_.m) * source_row_stride;
        const std::size_t columns = copy_columns <= 0
            ? static_cast<std::size_t>(score_row_stride_)
            : std::min(static_cast<std::size_t>(copy_columns),
                       static_cast<std::size_t>(score_row_stride_));
        const std::size_t destination_head_stride
            = static_cast<std::size_t>(query_len) * score_row_stride_;
        for (std::size_t local = 0; local < matrix_count; ++local) {
            const std::int8_t *source = (direct_scores == nullptr
                    ? group_scores_.data() : direct_scores)
                + local * source_head_stride;
            std::int8_t *destination = scores_.data()
                + (matrix_indices == nullptr ? matrix_begin + local
                                             : matrix_indices[local])
                    * destination_head_stride
                + static_cast<std::size_t>(query_begin) * score_row_stride_;
            for (int row = 0; row < query_count; ++row) {
                std::memcpy(
                    destination
                        + static_cast<std::size_t>(row) * score_row_stride_,
                    source + static_cast<std::size_t>(row) * source_row_stride,
                    columns);
            }
        }
    }

    bool selectTopKDirectMatrix(
        const Request &request, const ValidatedRequest &validated,
        std::size_t matrix, const std::int8_t *source,
        std::size_t source_row_stride) noexcept {
        return selectTopKDirectMatrixPartition(
            request, validated, matrix, source, source_row_stride, 0U, 1U);
    }

    bool selectTopKDirectMatrixPartition(
        const Request &request, const ValidatedRequest &validated,
        std::size_t matrix, const std::int8_t *source,
        std::size_t source_row_stride, std::size_t worker,
        std::size_t worker_count) noexcept {
        if (matrix >= validated.matrix_count || source == nullptr
            || source_row_stride < static_cast<std::size_t>(request.key_len)
            || worker_count == 0 || worker >= worker_count
            || topKOversample() != 1.0F
            || topk_preselected_[matrix] != 0) {
            return false;
        }
        constexpr std::size_t rows_per_block = 4;
        const std::size_t total_rows = static_cast<std::size_t>(
            request.query_len);
        const std::size_t total_blocks =
            (total_rows + rows_per_block - 1U) / rows_per_block;
        const std::size_t block_begin = total_blocks * worker / worker_count;
        const std::size_t block_end = total_blocks * (worker + 1U)
            / worker_count;
        const std::size_t row_begin = std::min(
            total_rows, block_begin * rows_per_block);
        const std::size_t row_end = std::min(
            total_rows, block_end * rows_per_block);
        if (row_begin == row_end) return true;
        const std::int32_t *offsets = rowOffsetsForMatrix(request, matrix);
        std::int32_t *destination = request.topk_indices
            + matrix * validated.topk_head_stride;
        return INT8TopK::selectCausalRowRange(
            source, total_rows, row_begin, row_end - row_begin,
            source_row_stride,
            static_cast<std::size_t>(validated.causal_prefix_tokens + 1),
            static_cast<std::size_t>(request.key_len), offsets, destination);
    }

    bool selectTopKRange(const Request &request,
                         const ValidatedRequest &validated,
                         std::size_t matrix_begin,
                         std::size_t matrix_count,
                         bool parallel_rows = true) noexcept {
        const bool contains_preselected = std::any_of(
            topk_preselected_.begin() + matrix_begin,
            topk_preselected_.begin() + matrix_begin + matrix_count,
            [](std::uint8_t value) { return value != 0; });
        if (!contains_preselected && topKOversample() == 1.0F
            && (!parallel_rows || validated.topk_threads == 1)) {
            for (std::size_t local = 0; local < matrix_count; ++local) {
                const std::size_t matrix = matrix_begin + local;
                const std::int32_t *offsets = rowOffsetsForMatrix(
                    request, matrix);
                const std::int8_t *source = scores_.data()
                    + matrix * static_cast<std::size_t>(request.query_len)
                        * score_row_stride_;
                std::int32_t *destination = request.topk_indices
                    + matrix * validated.topk_head_stride;
                if (!INT8TopK::selectCausalRows(
                        source, static_cast<std::size_t>(request.query_len),
                        static_cast<std::size_t>(score_row_stride_),
                        static_cast<std::size_t>(
                            validated.causal_prefix_tokens + 1),
                        static_cast<std::size_t>(request.key_len), offsets,
                        destination)) {
                    return false;
                }
            }
            if (recallDiagnosticsEnabled()) {
                diagnoseExactRecall(
                    request, validated, matrix_begin, matrix_count);
            }
            return true;
        }
        const std::int64_t row_count = static_cast<std::int64_t>(matrix_count)
            * request.query_len;
        std::atomic<bool> ok{true};
        const auto select_row = [&](std::int64_t local_row) {
            const std::size_t matrix = matrix_begin
                + static_cast<std::size_t>(
                    local_row / request.query_len);
            if (topk_preselected_[matrix] != 0) return;
            const int row = static_cast<int>(local_row % request.query_len);
            const std::int32_t *offsets = rowOffsetsForMatrix(request, matrix);
            const int valid = std::min(
                request.key_len,
                validated.causal_prefix_tokens + row + 1);
            const int keep = offsets[row + 1] - offsets[row];
            const std::int8_t *source = scores_.data()
                + (matrix * request.query_len + row) * score_row_stride_;
            std::int32_t *destination = request.topk_indices
                + matrix * validated.topk_head_stride + offsets[row];
            const bool selected_ok = INT8TopK::selectRow(
                source, static_cast<std::size_t>(valid),
                static_cast<std::size_t>(keep), destination);
            if (!selected_ok) {
                ok.store(false);
            }
        };
        if (!parallel_rows || validated.topk_threads == 1) {
            for (std::int64_t local_row = 0; local_row < row_count;
                 ++local_row) {
                select_row(local_row);
            }
        } else {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(validated.topk_threads)
#endif
            for (std::int64_t local_row = 0; local_row < row_count;
                 ++local_row) {
                select_row(local_row);
            }
        }
        if (ok.load() && recallDiagnosticsEnabled()) {
            if (dspInt32TopKEnabled()) {
                diagnoseDSPInt32Recall(
                    request, validated, matrix_begin, matrix_count);
            } else {
                diagnoseExactRecall(
                    request, validated, matrix_begin, matrix_count);
            }
        }
        return ok.load();
    }

    bool selectTopKCooperativePartition(
        const Request &request, const ValidatedRequest &validated,
        std::size_t matrix, std::size_t worker,
        std::size_t worker_count) noexcept {
        if (worker_count == 0 || worker >= worker_count
            || matrix >= validated.matrix_count
            || topk_preselected_[matrix] != 0
            || topKOversample() != 1.0F) {
            return false;
        }
        constexpr std::size_t rows_per_block = 4;
        const std::size_t total_rows = static_cast<std::size_t>(
            request.query_len);
        const std::size_t total_blocks =
            (total_rows + rows_per_block - 1U) / rows_per_block;
        const std::size_t block_begin = total_blocks * worker / worker_count;
        const std::size_t block_end = total_blocks * (worker + 1U)
            / worker_count;
        const std::size_t row_begin = std::min(
            total_rows, block_begin * rows_per_block);
        const std::size_t row_end = std::min(
            total_rows, block_end * rows_per_block);
        if (row_begin == row_end) return true;
        const std::int32_t *offsets = rowOffsetsForMatrix(request, matrix);
        const std::int8_t *source = scores_.data()
            + matrix * total_rows * static_cast<std::size_t>(
                score_row_stride_);
        std::int32_t *destination = request.topk_indices
            + matrix * validated.topk_head_stride;
        return INT8TopK::selectCausalRowRange(
            source, total_rows, row_begin, row_end - row_begin,
            static_cast<std::size_t>(score_row_stride_),
            static_cast<std::size_t>(validated.causal_prefix_tokens + 1),
            static_cast<std::size_t>(request.key_len), offsets, destination);
    }

    bool selectTopKMatrixRowRange(
        const Request &request, const ValidatedRequest &validated,
        std::size_t matrix, int row_begin, int row_count) noexcept {
        if (matrix >= validated.matrix_count || row_begin < 0 || row_count < 0
            || row_begin > request.query_len
            || row_count > request.query_len - row_begin
            || topKOversample() != 1.0F) {
            return false;
        }
        if (row_count == 0 || topk_preselected_[matrix] != 0) return true;
        const std::int32_t *offsets = rowOffsetsForMatrix(request, matrix);
        const std::size_t total_rows = static_cast<std::size_t>(
            request.query_len);
        const std::int8_t *source = scores_.data()
            + matrix * total_rows
                * static_cast<std::size_t>(score_row_stride_);
        std::int32_t *destination = request.topk_indices
            + matrix * validated.topk_head_stride;
        return INT8TopK::selectCausalRowRange(
            source, total_rows, static_cast<std::size_t>(row_begin),
            static_cast<std::size_t>(row_count),
            static_cast<std::size_t>(score_row_stride_),
            static_cast<std::size_t>(validated.causal_prefix_tokens + 1),
            static_cast<std::size_t>(request.key_len), offsets, destination);
    }

    static bool recallDiagnosticsEnabled() noexcept {
        const char *value = std::getenv("MLLM_HMX_INT8_RECALL_DIAGNOSTICS");
        return value != nullptr && value[0] != '\0'
            && std::strcmp(value, "0") != 0;
    }

    void diagnoseDSPInt32Recall(const Request &request,
                                const ValidatedRequest &validated,
                                std::size_t matrix_begin,
                                std::size_t matrix_count) noexcept {
        try {
            const auto score_order = [](const auto &left,
                                        const auto &right) {
                if (left.first != right.first) return left.first > right.first;
                return left.second < right.second;
            };
            for (std::size_t local = 0; local < matrix_count; ++local) {
                const std::size_t matrix = matrix_begin + local;
                if (topk_preselected_[matrix] == 0) continue;
                const std::int32_t *offsets = rowOffsetsForMatrix(
                    request, matrix);
                const std::size_t query_stride =
                    request.query_f32_row_stride == 0
                    ? static_cast<std::size_t>(request.head_dim)
                    : request.query_f32_row_stride;
                const float *query_base = request.query_f32_heads[matrix];
                const Half *key_base = request.key_heads[matrix];
                std::vector<std::pair<float, std::int32_t>> float_ranked(
                    static_cast<std::size_t>(request.key_len));
                std::vector<std::pair<std::int32_t, std::int32_t>> i32_ranked(
                    static_cast<std::size_t>(request.key_len));
                std::vector<std::int32_t> expected;
                std::uint64_t selected_total = 0;
                std::uint64_t float_overlap_total = 0;
                std::uint64_t i32_overlap_total = 0;
                for (int row = 0; row < request.query_len; ++row) {
                    const int valid = std::min(
                        request.key_len,
                        validated.causal_prefix_tokens + row + 1);
                    const int keep = offsets[row + 1] - offsets[row];
                    const float *query = query_base
                        + static_cast<std::size_t>(row) * query_stride;
                    for (int key_index = 0; key_index < valid; ++key_index) {
                        const Half *key = key_base
                            + static_cast<std::size_t>(key_index)
                                * request.head_dim;
                        float float_score = 0.0F;
                        std::int32_t i32_score = 0;
                        for (int dimension = 0;
                             dimension < request.head_dim; ++dimension) {
                            const float key_value = halfToFloat(
                                key[dimension]);
                            float_score += query[dimension] * key_value;
                            i32_score += static_cast<std::int32_t>(
                                    quantize(query[dimension],
                                             score_q_scales_[matrix]))
                                * static_cast<std::int32_t>(
                                    quantize(key_value,
                                             score_k_scales_[matrix]));
                        }
                        float_ranked[static_cast<std::size_t>(key_index)] = {
                            float_score, key_index};
                        i32_ranked[static_cast<std::size_t>(key_index)] = {
                            i32_score, key_index};
                    }
                    const std::int32_t *selected = request.topk_indices
                        + matrix * validated.topk_head_stride + offsets[row];
                    auto count_overlap = [&](auto &ranked) {
                        if (keep < valid) {
                            std::nth_element(
                                ranked.begin(), ranked.begin() + keep,
                                ranked.begin() + valid, score_order);
                        }
                        expected.resize(static_cast<std::size_t>(keep));
                        for (int index = 0; index < keep; ++index) {
                            expected[static_cast<std::size_t>(index)] =
                                ranked[static_cast<std::size_t>(index)].second;
                        }
                        std::sort(expected.begin(), expected.end());
                        std::uint64_t overlap = 0;
                        int expected_index = 0;
                        int selected_index = 0;
                        while (expected_index < keep
                               && selected_index < keep) {
                            if (expected[static_cast<std::size_t>(
                                             expected_index)]
                                == selected[selected_index]) {
                                ++overlap;
                                ++expected_index;
                                ++selected_index;
                            } else if (expected[static_cast<std::size_t>(
                                                   expected_index)]
                                       < selected[selected_index]) {
                                ++expected_index;
                            } else {
                                ++selected_index;
                            }
                        }
                        return overlap;
                    };
                    float_overlap_total += count_overlap(float_ranked);
                    i32_overlap_total += count_overlap(i32_ranked);
                    selected_total += static_cast<std::uint64_t>(keep);
                }
                std::lock_guard<std::mutex> lock(diagnosticMutex());
                std::cerr << "[HMX_INT32_TOPK_RECALL] op="
                          << (request.diagnostic_label.empty()
                                  ? "unknown" : request.diagnostic_label)
                          << " matrix=" << matrix
                          << " float_recall="
                          << (selected_total == 0 ? 1.0
                              : static_cast<double>(float_overlap_total)
                                  / static_cast<double>(selected_total))
                          << " int32_match="
                          << (selected_total == 0 ? 1.0
                              : static_cast<double>(i32_overlap_total)
                                  / static_cast<double>(selected_total))
                          << " q_scale=" << score_q_scales_[matrix]
                          << " k_scale=" << score_k_scales_[matrix]
                          << std::endl;
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(diagnosticMutex());
            std::cerr << "[HMX_INT32_TOPK_RECALL] op="
                      << (request.diagnostic_label.empty()
                              ? "unknown" : request.diagnostic_label)
                      << " error=diagnostic-allocation-failed" << std::endl;
        }
    }

    void diagnoseExactRecall(const Request &request,
                             const ValidatedRequest &validated,
                             std::size_t matrix_begin,
                             std::size_t matrix_count) noexcept {
        try {
            for (std::size_t local = 0; local < matrix_count; ++local) {
                const std::size_t matrix = matrix_begin + local;
                if (topk_preselected_[matrix] != 0) continue;
                const std::int32_t *offsets = rowOffsetsForMatrix(
                    request, matrix);
                const std::size_t query_stride =
                    request.query_f32_row_stride == 0
                    ? static_cast<std::size_t>(request.head_dim)
                    : request.query_f32_row_stride;
                const float *query_base = request.query_f32_heads[matrix];
                const Half *key_base = request.key_heads[matrix];
                std::vector<std::pair<float, std::int32_t>> ranked(
                    static_cast<std::size_t>(request.key_len));
                std::vector<std::int32_t> exact;
                std::uint64_t selected_total = 0;
                std::uint64_t overlap_total = 0;
                std::uint64_t score_total = 0;
                std::uint64_t zero_total = 0;
                std::uint64_t saturated_total = 0;
                std::uint64_t score_mismatch_total = 0;
                std::uint8_t observed[256] = {};
                MatrixScales measured;
                for (int row = 0; row < request.query_len; ++row) {
                    const float *query = query_base
                        + static_cast<std::size_t>(row) * query_stride;
                    for (int dimension = 0; dimension < request.head_dim;
                         ++dimension) {
                        measured.q = std::max(
                            measured.q,
                            std::abs(query[dimension]) / 127.0F);
                    }
                }
                for (int key_index = 0; key_index < request.key_len;
                     ++key_index) {
                    const Half *key = key_base
                        + static_cast<std::size_t>(key_index)
                            * request.head_dim;
                    for (int dimension = 0; dimension < request.head_dim;
                         ++dimension) {
                        measured.k = std::max(
                            measured.k,
                            std::abs(halfToFloat(key[dimension]))
                                / 127.0F);
                    }
                }
                const OperatorInfo &diagnostic_info =
                    bucket_operators_.empty()
                    ? info_
                    : bucket_operators_[closestBucket(measured)].info;
                const auto score_order = [](const auto &left,
                                            const auto &right) {
                    if (left.first != right.first) {
                        return left.first > right.first;
                    }
                    return left.second < right.second;
                };
                for (int row = 0; row < request.query_len; ++row) {
                    const int valid = std::min(
                        request.key_len,
                        validated.causal_prefix_tokens + row + 1);
                    const int keep = offsets[row + 1] - offsets[row];
                    const float *query = query_base
                        + static_cast<std::size_t>(row) * query_stride;
                    for (int key_index = 0; key_index < valid; ++key_index) {
                        const Half *key = key_base
                            + static_cast<std::size_t>(key_index)
                                * request.head_dim;
                        float score = 0.0F;
                        std::int32_t quantized_score = 0;
                        for (int dimension = 0;
                             dimension < request.head_dim; ++dimension) {
                            const float key_value = halfToFloat(
                                key[dimension]);
                            score += query[dimension] * key_value;
                            quantized_score += static_cast<std::int32_t>(
                                    quantize(query[dimension],
                                         score_q_scales_[matrix] > 0.0F
                                             ? score_q_scales_[matrix]
                                             : diagnostic_info.q_scale))
                                * static_cast<std::int32_t>(
                                    quantize(key_value,
                                             score_k_scales_[matrix] > 0.0F
                                                 ? score_k_scales_[matrix]
                                                 : diagnostic_info.k_scale));
                        }
                        ranked[static_cast<std::size_t>(key_index)] = {
                            score, key_index};
                        const float effective_requant =
                            score_requant_scales_.size()
                                    == validated.matrix_count
                                        * static_cast<std::size_t>(
                                            request.query_len)
                                && score_requant_scales_[
                                       matrix * request.query_len + row] > 0.0F
                            ? score_requant_scales_[
                                  matrix * request.query_len + row]
                            : diagnostic_info.requant_scale;
                        const float requantized =
                            static_cast<float>(quantized_score)
                            * effective_requant;
                        const int rounded = static_cast<int>(std::nearbyint(
                            requantized));
                        const std::int8_t expected_score =
                            static_cast<std::int8_t>(std::max(
                                -128, std::min(127, rounded)));
                        const std::int8_t actual_score = scores_[
                            (matrix * request.query_len + row)
                                * score_row_stride_
                            + static_cast<std::size_t>(key_index)];
                        score_mismatch_total +=
                            expected_score != actual_score;
                    }
                    if (keep < valid) {
                        std::nth_element(
                            ranked.begin(), ranked.begin() + keep,
                            ranked.begin() + valid, score_order);
                    }
                    exact.resize(static_cast<std::size_t>(keep));
                    for (int index = 0; index < keep; ++index) {
                        exact[static_cast<std::size_t>(index)] =
                            ranked[static_cast<std::size_t>(index)].second;
                    }
                    std::sort(exact.begin(), exact.end());
                    const std::int32_t *selected = request.topk_indices
                        + matrix * validated.topk_head_stride + offsets[row];
                    int exact_index = 0;
                    int selected_index = 0;
                    while (exact_index < keep && selected_index < keep) {
                        if (exact[static_cast<std::size_t>(exact_index)]
                            == selected[selected_index]) {
                            ++overlap_total;
                            ++exact_index;
                            ++selected_index;
                        } else if (exact[static_cast<std::size_t>(exact_index)]
                                   < selected[selected_index]) {
                            ++exact_index;
                        } else {
                            ++selected_index;
                        }
                    }
                    selected_total += static_cast<std::uint64_t>(keep);
                    const std::int8_t *score_row = scores_.data()
                        + (matrix * request.query_len + row)
                            * score_row_stride_;
                    for (int key_index = 0; key_index < valid; ++key_index) {
                        const std::int8_t score = score_row[key_index];
                        ++score_total;
                        zero_total += score == 0;
                        saturated_total +=
                            score == INT8_MIN || score == INT8_MAX;
                        observed[static_cast<unsigned int>(
                            static_cast<int>(score) + 128)] = 1;
                    }
                }
                int unique_scores = 0;
                for (const std::uint8_t seen : observed) {
                    unique_scores += seen != 0;
                }
                std::lock_guard<std::mutex> lock(diagnosticMutex());
                std::cerr << "[HMX_INT8_RECALL] op="
                          << (request.diagnostic_label.empty()
                                  ? "unknown" : request.diagnostic_label)
                          << " layer=" << request.layer_id
                          << " key_len=" << request.key_len
                          << " matrix=" << matrix
                          << " recall="
                          << (selected_total == 0 ? 1.0
                              : static_cast<double>(overlap_total)
                                  / static_cast<double>(selected_total))
                          << " unique_scores=" << unique_scores
                          << " zero_fraction="
                          << (score_total == 0 ? 0.0
                              : static_cast<double>(zero_total)
                                  / static_cast<double>(score_total))
                          << " saturation_fraction="
                          << (score_total == 0 ? 0.0
                              : static_cast<double>(saturated_total)
                                  / static_cast<double>(score_total))
                          << " score_mismatch_fraction="
                          << (score_total == 0 ? 0.0
                              : static_cast<double>(score_mismatch_total)
                                  / static_cast<double>(score_total))
                          << " measured_q=" << measured.q
                          << " measured_k=" << measured.k
                          << " bucket_q=" << diagnostic_info.q_scale
                          << " bucket_k=" << diagnostic_info.k_scale
                          << " used_q="
                          << (score_q_scales_[matrix] > 0.0F
                                  ? score_q_scales_[matrix]
                                  : diagnostic_info.q_scale)
                          << " used_k="
                          << (score_k_scales_[matrix] > 0.0F
                                  ? score_k_scales_[matrix]
                                  : diagnostic_info.k_scale)
                          << std::endl;
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(diagnosticMutex());
            std::cerr << "[HMX_INT8_RECALL] op="
                      << (request.diagnostic_label.empty()
                              ? "unknown" : request.diagnostic_label)
                      << " layer=" << request.layer_id
                      << " key_len=" << request.key_len
                      << " error=diagnostic-allocation-failed" << std::endl;
        }
    }

    static std::mutex &diagnosticMutex() {
        static std::mutex mutex;
        return mutex;
    }

    static float topKOversample() noexcept {
        const char *value = std::getenv("MLLM_HMX_INT8_TOPK_OVERSAMPLE");
        if (value == nullptr || value[0] == '\0') return 1.0F;
        char *end = nullptr;
        const float parsed = std::strtof(value, &end);
        if (end == value || *end != '\0' || !std::isfinite(parsed)
            || parsed != 1.0F) {
            return 0.0F;
        }
        return parsed;
    }

    bool selectDirectThreeStagePipelined(
        const Request &request, const ValidatedRequest &validated,
        bool profile_enabled, std::string *error) {
        if (!request.topk_group_ready) {
            return fail("direct three-stage execution requires the sparse "
                        "callback", error);
        }
        if (bucket_operators_.empty()
            || topKOversample() != 1.0F || dspInt32TopKEnabled()
            || dynamicQKScaleEnabled() || dynamicOutputScaleEnabled()
            || outlierFallbackEnabled() || recallDiagnosticsEnabled()) {
            return fail("direct three-stage execution requires the static "
                        "bucket CPU INT8 quality path, oversample=1, no "
                        "fallback, and recall diagnostics disabled", error);
        }
        const int sparse_assist_percent = directSparseAssistPercent();
        if (sparse_assist_percent < 0) {
            return fail("MLLM_HMX_PIPELINE_DIRECT_SPARSE_ASSIST_PERCENT "
                        "must be in [0, 90]", error);
        }
        const bool sparse_assist = sparse_assist_percent > 0;
        if (sparse_assist && !request.topk_group_rows_ready) {
            return fail("direct sparse assistance requires the row-range "
                        "callback", error);
        }
        // Cooperative Top-k already occupies both of its persistent workers
        // for every head.  Running the sparse tail inline on worker 0 leaves
        // the configured sparse core idle and delays the next head's Top-k.
        // Keep the old placement as the default for reproducible A/Bs, but
        // allow one attention-scoped task on the persistent sparse pool to
        // consume the tail instead.  There is still no per-head submission.
        const bool dedicated_sparse_assist = sparse_assist
            && (!cooperativeTopKEnabled()
                || directSparseWorkerAssistEnabled());
        const bool topk_inline_sparse_assist = sparse_assist
            && cooperativeTopKEnabled() && !dedicated_sparse_assist;

        const bool timeline_enabled = pipelineTimelineDiagnosticsEnabled();
        const bool timing_enabled = profile_enabled || timeline_enabled;
        const std::uint64_t begin_ns = timing_enabled
            ? CPUAttentionProfiler::nowNs() : 0;
        struct alignas(128) ProgressCounter {
            std::atomic<std::size_t> value{0};
        };
        ProgressCounter topk_published;
        // The producer constructs DirectScoreStream on its callback stack.
        // Keep a selector-owned copy so the CPU2 sparse helper may finish
        // after the callback returns and the H12 RPC is being retired.
        DirectScoreStream stable_stream;
        std::atomic<const DirectScoreStream *> stream_view{nullptr};
        std::atomic<bool> cancelled{false};
        std::atomic<bool> topk_done{false};
        std::atomic<bool> stream_seen{false};
        std::atomic<std::uint64_t> first_ready_ns{0};
        std::atomic<std::uint64_t> last_ready_ns{0};
        std::atomic<std::uint64_t> topk_finish_ns{0};
        std::atomic<std::uint64_t> sparse_caller_finish_ns{0};
        std::atomic<std::uint64_t> sparse_assist_finish_ns{0};
        std::atomic<std::uint64_t> sparse_finish_ns{0};
        std::atomic<int> sparse_assist_cpu{-1};
#if defined(__linux__)
        const int sparse_caller_cpu = timeline_enabled ? sched_getcpu() : -1;
#else
        const int sparse_caller_cpu = -1;
#endif
        std::uint64_t topk_busy_ns = 0;
        std::uint64_t sparse_busy_ns = 0;
        std::uint64_t sparse_assist_busy_ns = 0;
        std::atomic<bool> topk_failed{false};
        std::string topk_error;
        std::string sparse_error;
        std::string sparse_assist_error;

        std::vector<int> sparse_assist_row_begin(
            validated.matrix_count, request.query_len);
        if (sparse_assist) {
            for (std::size_t matrix = 0;
                 matrix < validated.matrix_count; ++matrix) {
                const std::int32_t *offsets = rowOffsetsForMatrix(
                    request, matrix);
                std::uint64_t total_cost = 0;
                for (int row = 0; row < request.query_len; ++row) {
                    const std::uint64_t keep = static_cast<std::uint64_t>(
                        offsets[row + 1] - offsets[row]);
                    total_cost += keep * static_cast<std::uint64_t>(
                        2 * request.head_dim + 8);
                    total_cost += static_cast<std::uint64_t>(
                        request.head_dim + 64);
                }
                const std::uint64_t target =
                    (total_cost * static_cast<std::uint64_t>(
                        sparse_assist_percent) + 99U) / 100U;
                std::uint64_t tail_cost = 0;
                int row_begin = request.query_len;
                while (row_begin > 0 && tail_cost < target) {
                    --row_begin;
                    const std::uint64_t keep = static_cast<std::uint64_t>(
                        offsets[row_begin + 1] - offsets[row_begin]);
                    tail_cost += keep * static_cast<std::uint64_t>(
                        2 * request.head_dim + 8);
                    tail_cost += static_cast<std::uint64_t>(
                        request.head_dim + 64);
                }
                sparse_assist_row_begin[matrix] = row_begin;
            }
        }

        const bool cooperative_topk = cooperativeTopKEnabled();
        CooperativeTaskExecutor *cooperative_executor = nullptr;
        std::size_t topk_worker_count = 1;
        if (cooperative_topk) {
            cooperative_executor = &cooperativeTopKTaskExecutor();
            topk_worker_count = static_cast<std::size_t>(
                cooperative_executor->workerCount());
            if (topk_worker_count != 2U) {
                return fail("direct cooperative Top-k requires exactly two "
                            "persistent workers", error);
            }
        }
        auto head_workers_done = std::make_unique<
            std::atomic<std::size_t>[]>(validated.matrix_count);
        auto head_topk_begin_ns = std::make_unique<
            std::atomic<std::uint64_t>[]>(validated.matrix_count);
        for (std::size_t matrix = 0; matrix < validated.matrix_count;
             ++matrix) {
            head_workers_done[matrix].store(0, std::memory_order_relaxed);
            head_topk_begin_ns[matrix].store(
                std::numeric_limits<std::uint64_t>::max(),
                std::memory_order_relaxed);
        }
        std::vector<std::uint64_t> topk_worker_busy_ns(
            topk_worker_count, 0);
        const auto report_topk_error = [&](std::string message) {
            bool expected = false;
            if (topk_failed.compare_exchange_strong(
                    expected, true, std::memory_order_acq_rel)) {
                topk_error = std::move(message);
            }
            cancelled.store(true, std::memory_order_release);
            pipelineStreamingNotify();
        };

        // Submit before the RPC so CPU4/CPU6 are resident and waiting when
        // the DSP publishes head 0. Each worker owns disjoint four-row
        // blocks; the second completion publishes the head to CPU7.
        const auto consume_topk = [&](std::size_t worker,
                                      std::size_t worker_count) {
            const DirectScoreStream *stream = nullptr;
            while ((stream = stream_view.load(std::memory_order_acquire))
                       == nullptr
                   && !cancelled.load(std::memory_order_acquire)) {
                pipelineStreamingPause();
            }
            if (stream == nullptr) return;
            const auto deadline = std::chrono::steady_clock::now()
                + std::chrono::seconds(6);
            std::uint64_t worker_busy_ns = 0;
            for (std::size_t local = 0; local < stream->count; ++local) {
                while (!cancelled.load(std::memory_order_acquire)) {
                    const std::int32_t state = __atomic_load_n(
                        stream->ready + local * stream->ready_stride,
                        __ATOMIC_ACQUIRE);
                    if (state == kLongRpcOutputReady) break;
                    if (state == kLongRpcOutputError) {
                        report_topk_error(
                            "long-lived HMX RPC reported a DSP failure "
                            "before direct Top-k head "
                            + std::to_string(local));
                        break;
                    }
                    if (std::chrono::steady_clock::now() >= deadline) {
                        report_topk_error(
                            "timed out waiting for direct HMX score head "
                            + std::to_string(local));
                        break;
                    }
                    pipelineStreamingPause();
                }
                if (cancelled.load(std::memory_order_acquire)) break;
                const std::uint64_t ready_ns = timing_enabled
                    ? CPUAttentionProfiler::nowNs() : 0;
                const std::size_t ready_observer =
                    topk_inline_sparse_assist
                    ? worker_count - 1U : 0U;
                if (worker == ready_observer) {
                    std::uint64_t expected = 0;
                    first_ready_ns.compare_exchange_strong(
                        expected, ready_ns, std::memory_order_release,
                        std::memory_order_relaxed);
                    last_ready_ns.store(ready_ns, std::memory_order_release);
                }
                const std::uint64_t topk_begin = timing_enabled
                    ? CPUAttentionProfiler::nowNs() : 0;
                if (hmx_pipeline::LatencyRecorder::enabled()) {
                    std::uint64_t current =
                        head_topk_begin_ns[local].load(
                            std::memory_order_relaxed);
                    const std::uint64_t observed =
                        CPUAttentionProfiler::nowNs();
                    while (observed < current
                           && !head_topk_begin_ns[local]
                                   .compare_exchange_weak(
                                       current, observed,
                                       std::memory_order_relaxed)) {
                    }
                }
                const std::size_t matrix = stream->matrices[local];
                if (!selectTopKDirectMatrixPartition(
                        request, validated, matrix,
                        stream->scores
                            + local * stream->score_head_stride,
                        stream->score_row_stride, worker, worker_count)) {
                    report_topk_error(
                        "direct cooperative rpcmem INT8 Top-k failed for "
                        "head " + std::to_string(matrix));
                    break;
                }
                if (profile_enabled) {
                    worker_busy_ns += CPUAttentionProfiler::nowNs()
                        - topk_begin;
                }
                const std::size_t completed_workers =
                    head_workers_done[local].fetch_add(
                        1U, std::memory_order_acq_rel) + 1U;
                if (completed_workers == worker_count) {
                    if (hmx_pipeline::LatencyRecorder::enabled()) {
                        const int head = static_cast<int>(
                            matrix % static_cast<std::size_t>(request.heads));
                        const float retention =
                            request.head_retentions == nullptr
                            ? 1.0F : request.head_retentions[head];
                        hmx_pipeline::LatencyRecorder::head(
                            "topk", request.key_len, request.layer_id, head,
                            retention,
                            CPUAttentionProfiler::nowNs()
                                - head_topk_begin_ns[local].load(
                                    std::memory_order_relaxed));
                    }
                    topk_published.value.store(
                        local + 1U, std::memory_order_release);
                    pipelineStreamingNotify();
                }
                if (topk_inline_sparse_assist && worker == 0U) {
                    while (head_workers_done[local].load(
                               std::memory_order_acquire) < worker_count
                           && !cancelled.load(std::memory_order_acquire)) {
                        pipelineStreamingPause();
                    }
                    if (cancelled.load(std::memory_order_acquire)) break;
                    const int row_begin =
                        sparse_assist_row_begin[matrix];
                    const std::uint64_t assist_begin = timing_enabled
                        ? CPUAttentionProfiler::nowNs() : 0;
                    try {
                        request.topk_group_rows_ready(
                            matrix, 1U, row_begin,
                            request.query_len - row_begin);
                        if (profile_enabled) {
                            sparse_assist_busy_ns +=
                                CPUAttentionProfiler::nowNs() - assist_begin;
                        }
                    } catch (const std::exception &exception) {
                        sparse_assist_error = exception.what();
                        cancelled.store(true, std::memory_order_release);
                        pipelineStreamingNotify();
                        break;
                    } catch (...) {
                        sparse_assist_error =
                            "unknown exception in fused sparse assistant";
                        cancelled.store(true, std::memory_order_release);
                        pipelineStreamingNotify();
                        break;
                    }
                }
            }
            topk_worker_busy_ns[worker] = worker_busy_ns;
        };
        const auto finish_topk = [&]() {
            if (timing_enabled) {
                topk_finish_ns.store(
                    CPUAttentionProfiler::nowNs(), std::memory_order_release);
            }
            topk_done.store(true, std::memory_order_release);
            pipelineStreamingNotify();
        };

        std::future<void> topk_consumer;
        if (cooperative_topk) {
            topk_consumer = cooperative_executor->submit(
                consume_topk, finish_topk);
        } else {
            topk_consumer = topKTaskExecutor().submit([&]() {
                consume_topk(0U, 1U);
                finish_topk();
            });
        }

        std::future<void> sparse_assistant;
        if (dedicated_sparse_assist) {
            sparse_assistant = sparseTaskExecutor().submit([&]() {
#if defined(__linux__)
                if (timeline_enabled) {
                    sparse_assist_cpu.store(
                        sched_getcpu(), std::memory_order_relaxed);
                }
#endif
                const DirectScoreStream *stream = nullptr;
                while ((stream = stream_view.load(std::memory_order_acquire))
                           == nullptr
                       && !cancelled.load(std::memory_order_acquire)) {
                    pipelineStreamingPause();
                }
                try {
                    if (stream != nullptr) {
                        for (std::size_t local = 0;
                             local < stream->count; ++local) {
                            while (topk_published.value.load(
                                       std::memory_order_acquire) <= local
                                   && !cancelled.load(
                                       std::memory_order_acquire)) {
                                pipelineStreamingPause();
                            }
                            if (cancelled.load(std::memory_order_acquire)) {
                                break;
                            }
                            const std::size_t matrix =
                                stream->matrices[local];
                            const int row_begin =
                                sparse_assist_row_begin[matrix];
                            const std::uint64_t assist_begin = timing_enabled
                                ? CPUAttentionProfiler::nowNs() : 0;
                            request.topk_group_rows_ready(
                                matrix, 1U, row_begin,
                                request.query_len - row_begin);
                            if (profile_enabled) {
                                sparse_assist_busy_ns +=
                                    CPUAttentionProfiler::nowNs()
                                    - assist_begin;
                            }
                        }
                    }
                } catch (const std::exception &exception) {
                    sparse_assist_error = exception.what();
                    cancelled.store(true, std::memory_order_release);
                } catch (...) {
                    sparse_assist_error =
                        "unknown exception in direct sparse assistant";
                    cancelled.store(true, std::memory_order_release);
                }
                if (timing_enabled) {
                    sparse_assist_finish_ns.store(
                        CPUAttentionProfiler::nowNs(),
                        std::memory_order_release);
                }
                pipelineStreamingNotify();
            });
        }

        const DirectScoreStreamCallback on_direct_stream =
            [&](const DirectScoreStream &stream,
                std::string &stream_error) -> bool {
                if (stream.count != validated.matrix_count
                    || stream.ready == nullptr || stream.scores == nullptr
                    || stream.matrices == nullptr) {
                    stream_error = "invalid direct long-RPC score stream";
                    cancelled.store(true, std::memory_order_release);
                    pipelineStreamingNotify();
                    return false;
                }
                stable_stream = stream;
                stream_seen.store(true, std::memory_order_release);
                stream_view.store(&stable_stream, std::memory_order_release);
                pipelineStreamingNotify();
                std::size_t consumed = 0;
                try {
                    while (true) {
                        const std::size_t available =
                            topk_published.value.load(
                                std::memory_order_acquire);
                        if (consumed < available) {
                            const std::uint64_t sparse_begin =
                                (timing_enabled
                                 || hmx_pipeline::LatencyRecorder::enabled())
                                ? CPUAttentionProfiler::nowNs() : 0;
                            const std::size_t matrix =
                                stream.matrices[consumed];
                            if (sparse_assist) {
                                const int row_count =
                                    sparse_assist_row_begin[matrix];
                                if (row_count > 0) {
                                    request.topk_group_rows_ready(
                                        matrix, 1U, 0, row_count);
                                }
                            } else {
                                request.topk_group_ready(matrix, 1U);
                            }
                            if (hmx_pipeline::LatencyRecorder::enabled()) {
                                const int head = static_cast<int>(
                                    matrix % static_cast<std::size_t>(
                                        request.heads));
                                const float retention =
                                    request.head_retentions == nullptr
                                    ? 1.0F
                                    : request.head_retentions[head];
                                hmx_pipeline::LatencyRecorder::head(
                                    "sparse", request.key_len,
                                    request.layer_id, head, retention,
                                    CPUAttentionProfiler::nowNs()
                                        - sparse_begin);
                            }
                            if (profile_enabled) {
                                sparse_busy_ns += CPUAttentionProfiler::nowNs()
                                    - sparse_begin;
                            }
                            ++consumed;
                            continue;
                        }
                        if (cancelled.load(std::memory_order_acquire)
                            || (topk_done.load(std::memory_order_acquire)
                                && consumed == available)) {
                            break;
                        }
                        pipelineStreamingPause();
                    }
                } catch (const std::exception &exception) {
                    sparse_error = exception.what();
                    cancelled.store(true, std::memory_order_release);
                } catch (...) {
                    sparse_error =
                        "unknown exception in direct sparse consumer";
                    cancelled.store(true, std::memory_order_release);
                }
                while (!topk_done.load(std::memory_order_acquire)) {
                    pipelineStreamingPause();
                }
                if (timing_enabled) {
                    sparse_caller_finish_ns.store(
                        CPUAttentionProfiler::nowNs(),
                        std::memory_order_release);
                }
                pipelineStreamingNotify();
                if (!topk_error.empty() && stream_error.empty()) {
                    stream_error = topk_error;
                }
                if (!sparse_error.empty() && stream_error.empty()) {
                    stream_error = sparse_error;
                }
                return !cancelled.load(std::memory_order_acquire)
                    && consumed == stream.count;
            };

        std::string detail;
        const bool produced = produceBucketedGroups(
            request, validated, profile_enabled, ReadyGroupCallback(),
            detail, nullptr, on_direct_stream);
        if (!produced || !stream_seen.load(std::memory_order_acquire)) {
            cancelled.store(true, std::memory_order_release);
            pipelineStreamingNotify();
        }
        try {
            topk_consumer.get();
        } catch (const std::exception &exception) {
            if (topk_error.empty()) topk_error = exception.what();
        } catch (...) {
            if (topk_error.empty()) {
                topk_error = "unknown exception in direct Top-k consumer";
            }
        }
        if (sparse_assistant.valid()) {
            try {
                sparse_assistant.get();
            } catch (const std::exception &exception) {
                if (sparse_assist_error.empty()) {
                    sparse_assist_error = exception.what();
                }
            } catch (...) {
                if (sparse_assist_error.empty()) {
                    sparse_assist_error =
                        "unknown exception in direct sparse assistant";
                }
            }
        }
        if (timing_enabled) {
            sparse_finish_ns.store(
                std::max(
                    sparse_caller_finish_ns.load(std::memory_order_acquire),
                    sparse_assist_finish_ns.load(std::memory_order_acquire)),
                std::memory_order_release);
        }
        topk_busy_ns = *std::max_element(
            topk_worker_busy_ns.begin(), topk_worker_busy_ns.end());
        sparse_busy_ns = std::max(
            sparse_busy_ns, sparse_assist_busy_ns);
        if (!sparse_assist_error.empty()) {
            return fail(sparse_assist_error, error);
        }
        if (!produced) return fail(detail, error);
        if (!stream_seen.load(std::memory_order_acquire)) {
            return fail("HMX long RPC did not expose a direct score stream",
                        error);
        }
        if (!topk_error.empty()) return fail(topk_error, error);
        if (!sparse_error.empty()) return fail(sparse_error, error);

        const std::uint64_t finish_ns = timing_enabled
            ? CPUAttentionProfiler::nowNs() : 0;
        if (profile_enabled) {
            const std::uint64_t last_ready = last_ready_ns.load(
                std::memory_order_acquire);
            CPUAttentionProfiler::add(
                AttentionProfileStage::HMX_NPU_PRODUCE_WALL,
                last_ready > begin_ns ? last_ready - begin_ns : 0);
            CPUAttentionProfiler::add(
                AttentionProfileStage::HMX_TOPK, topk_busy_ns);
            CPUAttentionProfiler::add(
                AttentionProfileStage::HMX_TOPK_STAGE_WALL,
                topk_finish_ns.load(std::memory_order_acquire) - begin_ns);
            CPUAttentionProfiler::add(
                AttentionProfileStage::HMX_SPARSE_STAGE_WALL,
                sparse_busy_ns);
        }
        if (timeline_enabled) {
            const std::uint64_t first_ready = first_ready_ns.load(
                std::memory_order_acquire);
            const std::uint64_t last_ready = last_ready_ns.load(
                std::memory_order_acquire);
            std::cerr << "[HMX_DIRECT_THREE_STAGE] layer="
                      << request.layer_id << " key_len=" << request.key_len
                      << " heads=" << validated.matrix_count
                      << " topk_workers=" << topk_worker_count
                      << " sparse_assist_percent="
                      << sparse_assist_percent
                      << " sparse_assist_lane="
                      << (dedicated_sparse_assist
                              ? "persistent-sparse-worker"
                              : (topk_inline_sparse_assist
                                     ? "topk-worker-0" : "none"))
                      << " sparse_caller_cpu=" << sparse_caller_cpu
                      << " sparse_assist_cpu="
                      << sparse_assist_cpu.load(std::memory_order_relaxed)
                      << " first_ready_us="
                      << (first_ready - begin_ns) / 1000.0
                      << " last_ready_us="
                      << (last_ready - begin_ns) / 1000.0
                      << " topk_finish_us="
                      << (topk_finish_ns.load(std::memory_order_acquire)
                          - begin_ns) / 1000.0
                      << " sparse_finish_us="
                      << (sparse_finish_ns.load(std::memory_order_acquire)
                          - begin_ns) / 1000.0
                      << " total_us=" << (finish_ns - begin_ns) / 1000.0
                      << " topk_busy_us=" << topk_busy_ns / 1000.0
                      << " sparse_busy_us=" << sparse_busy_ns / 1000.0
                      << std::endl;
        }
        return succeed(error);
    }

    // Keep NPU production coarse grained, then pipeline only the two CPU
    // stages.  One attention-scoped task occupies the persistent Top-k
    // worker while the already-persistent attention caller becomes the
    // sparse lane; individual heads cross the boundary through a
    // release/acquire progress word.  This avoids per-head futures, queue
    // operations and wakeups while preserving the useful Top-k(head i) /
    // sparse(head i - 1) overlap.
    bool selectPipelined(const Request &request,
                         const ValidatedRequest &validated,
                         bool profile_enabled, std::string *error) {
        if (!directThreeStageExecutorEnabled()) {
            return fail("MLLM_HMX_PIPELINE_EXECUTOR must be direct-three-stage", error);
        }
        return selectDirectThreeStagePipelined(
            request, validated, profile_enabled, error);
    }

    static bool directThreeStageExecutorEnabled() noexcept {
        const char *value = std::getenv("MLLM_HMX_PIPELINE_EXECUTOR");
        return value != nullptr
            && std::strcmp(value, "direct-three-stage") == 0;
    }

    static int directSparseAssistPercent() noexcept {
        const char *value = std::getenv(
            "MLLM_HMX_PIPELINE_DIRECT_SPARSE_ASSIST_PERCENT");
        if (value == nullptr || value[0] == '\0') return 0;
        char *end = nullptr;
        const long parsed = std::strtol(value, &end, 10);
        if (end == value || *end != '\0' || parsed < 0 || parsed > 90) {
            return -1;
        }
        return static_cast<int>(parsed);
    }

    static bool directSparseWorkerAssistEnabled() noexcept {
        const char *value = std::getenv(
            "MLLM_HMX_PIPELINE_DIRECT_SPARSE_WORKER_ASSIST");
        return value != nullptr && value[0] != '\0'
            && std::strcmp(value, "0") != 0;
    }

    static bool pipelineTimelineDiagnosticsEnabled() noexcept {
        const char *value = std::getenv(
            "MLLM_HMX_PIPELINE_TIMELINE_DIAGNOSTICS");
        return value != nullptr && value[0] != '\0'
            && std::strcmp(value, "0") != 0;
    }

    static void pipelineStreamingPause() noexcept {
#if defined(__aarch64__)
        const char *spin_wait = std::getenv(
            "MLLM_HMX_PIPELINE_STREAMING_SPIN_WAIT");
        if (spin_wait != nullptr && spin_wait[0] != '\0'
            && std::strcmp(spin_wait, "0") != 0) {
            // DSP writes to the shared ready word cannot issue an ARM SEV.
            // Busy-poll on the already-dedicated pipeline lanes when strict
            // low-latency hand-off is requested; otherwise WFE can sleep
            // until an unrelated interrupt and create periodic long tails.
            __asm__ __volatile__("yield" ::: "memory");
            return;
        }
        // WFE/SEV keeps the hand-off in userspace while allowing the
        // dedicated core to stop issuing instructions between head-ready
        // publications. The event register closes the check/sleep race: an
        // SEV that arrives just before WFE makes WFE return immediately.
        __asm__ __volatile__("wfe" ::: "memory");
#else
        std::this_thread::yield();
#endif
    }

    static void pipelineStreamingNotify() noexcept {
#if defined(__aarch64__)
        __asm__ __volatile__("sev" ::: "memory");
#else
        std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
    }

    static bool pipelinePlanDiagnosticsEnabled() noexcept {
        const char *value = std::getenv(
            "MLLM_HMX_PIPELINE_PLAN_DIAGNOSTICS");
        return value != nullptr && value[0] != '\0'
            && std::strcmp(value, "0") != 0;
    }

    static bool cooperativeTopKEnabled() noexcept {
        const char *value = std::getenv("MLLM_HMX_PIPELINE_TOPK_MODE");
        return value != nullptr
            && std::strcmp(value, "cooperative") == 0;
    }

    static std::int8_t quantize(float value, float scale) noexcept {
        return quantizeWithInverse(value, 1.0F / scale);
    }

    static std::int8_t quantizeWithInverse(
        float value, float inverse_scale) noexcept {
        if (!std::isfinite(value)) return 0;
        const float scaled = value * inverse_scale;
        if (scaled >= 127.0F) return INT8_MAX;
        if (scaled <= -127.0F) return -127;
        return static_cast<std::int8_t>(std::nearbyint(scaled));
    }

    static int topKThreadCount(int request_threads) noexcept {
        const char *value = std::getenv("MLLM_HMX_TOPK_THREADS");
        if (value == nullptr || value[0] == '\0') return request_threads;
        char *end = nullptr;
        const long parsed = std::strtol(value, &end, 10);
        if (end == value || *end != '\0' || parsed <= 0
            || parsed > std::numeric_limits<int>::max()) {
            return 0;
        }
        return static_cast<int>(parsed);
    }

    static float halfToFloat(Half value) noexcept {
        // Half is an IEEE-FP16 bit container, not an arithmetic uint16_t.
        // In particular, do not pass it directly to MLLM_FP16_TO_FP32 on
        // ARM, where that macro expects __fp16 and would numerically convert
        // the integer bit pattern instead of decoding it.
        const std::uint32_t sign
            = static_cast<std::uint32_t>(value & 0x8000U) << 16;
        std::uint32_t exponent = (value >> 10) & 0x1fU;
        std::uint32_t mantissa = value & 0x03ffU;
        std::uint32_t bits = 0;
        if (exponent == 0) {
            if (mantissa == 0) {
                bits = sign;
            } else {
                int shift = 0;
                while ((mantissa & 0x0400U) == 0) {
                    mantissa <<= 1;
                    ++shift;
                }
                mantissa &= 0x03ffU;
                bits = sign
                    | (static_cast<std::uint32_t>(127 - 14 - shift) << 23)
                    | (mantissa << 13);
            }
        } else if (exponent == 0x1fU) {
            bits = sign | 0x7f800000U | (mantissa << 13);
        } else {
            bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
        }
        float result = 0.0F;
        std::memcpy(&result, &bits, sizeof(result));
        return result;
    }

    static const std::int32_t *rowOffsetsForMatrix(
        const Request &request, std::size_t matrix) noexcept {
        if (request.row_offsets_head_stride == 0) return request.row_offsets;
        const std::size_t head = matrix
            % static_cast<std::size_t>(request.heads);
        return request.row_offsets + head * request.row_offsets_head_stride;
    }

    static bool checkedProduct(std::size_t left, std::size_t right,
                               std::size_t &result) noexcept {
        if (left != 0
            && right > std::numeric_limits<std::size_t>::max() / left) {
            return false;
        }
        result = left * right;
        return true;
    }

    std::string shapeError(int matrices, int query_len, int key_len,
                           int head_dim) const {
        return "INT8 HMX shape mismatch: requested matrices="
            + std::to_string(matrices) + ", M="
            + std::to_string(query_len) + ", K="
            + std::to_string(head_dim) + ", N="
            + std::to_string(key_len) + "; operator group H="
            + std::to_string(info_.heads) + ", tile M="
            + std::to_string(info_.m) + ", K="
            + std::to_string(info_.k) + ", capacity N="
            + std::to_string(info_.n);
    }

    std::string callError(const char *operation, int status) const {
        std::string result(operation);
        result += " failed with status " + std::to_string(status);
        if (status_string_ != nullptr) {
            const char *description = status_string_(status);
            if (description != nullptr && description[0] != '\0') {
                result += " (";
                result += description;
                result += ")";
            }
        }
        return result;
    }

    static bool nearlyEqual(float left, float right) noexcept {
        const float tolerance = 1.0e-5F
            * std::max({1.0F, std::abs(left), std::abs(right)});
        return std::abs(left - right) <= tolerance;
    }

#if defined(__ANDROID__)
    template<typename Function>
    static bool loadSymbolFrom(void *library, const char *name,
                               Function &function, std::string &error) {
        dlerror();
        void *symbol = dlsym(library, name);
        const char *message = dlerror();
        if (message != nullptr || symbol == nullptr) {
            error = std::string("dlsym(") + name + ") failed";
            if (message != nullptr) error += ": " + std::string(message);
            return false;
        }
        function = reinterpret_cast<Function>(symbol);
        return true;
    }

    template<typename Function>
    static bool loadOptionalSymbolFrom(void *library, const char *name,
                                      Function &function) {
        dlerror();
        void *symbol = dlsym(library, name);
        const char *message = dlerror();
        if (message != nullptr || symbol == nullptr) {
            function = nullptr;
            return false;
        }
        function = reinterpret_cast<Function>(symbol);
        return true;
    }

    template<typename Function>
    bool loadSymbol(const char *name, Function &function,
                    std::string &error) {
        return loadSymbolFrom(library_, name, function, error);
    }

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

    bool loadBucketCatalog(const std::string &manifest_path,
                           std::string &error) {
        error.clear();
        std::ifstream input(manifest_path);
        if (!input) {
            error = "failed to open INT8 HMX bucket manifest: "
                + manifest_path;
            return false;
        }
        const char *model_value = std::getenv(
            "MLLM_HMX_INT8_OPERATOR_MODEL");
        const std::string requested_model = model_value == nullptr
            ? std::string() : std::string(model_value);
        const std::filesystem::path manifest_directory
            = std::filesystem::path(manifest_path).parent_path();
        std::vector<BucketOperator> loaded;
        std::string line;
        std::size_t line_number = 0;
        while (std::getline(input, line)) {
            ++line_number;
            if (line.empty() || line[0] == '#') continue;
            const std::vector<std::string> fields = splitTabs(line);
            if (fields[0] == "model") continue;
            if (fields.size() < 11) {
                error = "invalid INT8 HMX bucket manifest row "
                    + std::to_string(line_number);
                break;
            }
            try {
                const int heads = std::stoi(fields[1]);
                const int m = std::stoi(fields[4]);
                const int k = std::stoi(fields[5]);
                const int n = std::stoi(fields[6]);
                const float q_scale = std::stof(fields[7]);
                const float k_scale = std::stof(fields[8]);
                const float output_scale = std::stof(fields[9]);
                if ((!requested_model.empty()
                     && fields[0] != requested_model)
                    || heads <= 0 || m != info_.m
                    || k != info_.k || n != info_.n) {
                    continue;
                }

                BucketOperator bucket;
                std::filesystem::path operator_path(fields[2]);
                if (operator_path.is_relative()) {
                    operator_path = manifest_directory / operator_path;
                }
                bucket.path = operator_path.lexically_normal().string();
                dlerror();
                bucket.library = dlopen(
                    bucket.path.c_str(), RTLD_NOW | RTLD_LOCAL);
                if (bucket.library == nullptr) {
                    const char *message = dlerror();
                    error = "dlopen(" + bucket.path + ") failed";
                    if (message != nullptr) {
                        error += ": " + std::string(message);
                    }
                    break;
                }
                ApiVersionFn api_version = nullptr;
                GetInfoFn get_info = nullptr;
                if (!loadSymbolFrom(bucket.library,
                                    "hmx_qk_i8_operator_api_version",
                                    api_version, error)
                    || !loadSymbolFrom(bucket.library,
                                       "hmx_qk_i8_operator_get_info",
                                       get_info, error)
                    || !loadSymbolFrom(bucket.library, "hmx_i8_create",
                                       bucket.create, error)
                    || !loadSymbolFrom(bucket.library, "hmx_i8_destroy",
                                       bucket.destroy, error)
                    || !loadSymbolFrom(bucket.library,
                                       "hmx_qk_i8_operator_matmul",
                                       bucket.matmul, error)
                    || !loadSymbolFrom(bucket.library,
                                       "hmx_qk_i8_operator_query_data",
                                       bucket.query_data, error)
                    || !loadSymbolFrom(bucket.library,
                                       "hmx_qk_i8_operator_key_data",
                                       bucket.key_data, error)
                    || !loadSymbolFrom(bucket.library,
                                       "hmx_qk_i8_operator_key_sums_data",
                                       bucket.key_sums_data, error)
                    || !loadSymbolFrom(bucket.library,
                                       "hmx_qk_i8_operator_scores_data",
                                       bucket.scores_data, error)
                    || !loadSymbolFrom(bucket.library,
                                       "hmx_qk_i8_operator_raw_query_data",
                                       bucket.raw_query_data, error)
                    || !loadSymbolFrom(bucket.library,
                                       "hmx_qk_i8_operator_raw_key_data",
                                       bucket.raw_key_data, error)
                    || !loadSymbolFrom(bucket.library,
                                       "hmx_qk_i8_operator_execute_hn",
                                       bucket.execute_hn, error)
                    || !loadSymbolFrom(
                           bucket.library,
                           "hmx_qk_i8_operator_prepare_raw_key_hn",
                           bucket.prepare_raw_key, error)
                    || !loadSymbolFrom(
                           bucket.library,
                           "hmx_qk_i8_operator_profile_raw_scales_hn",
                           bucket.profile_raw, error)
                    || !loadSymbolFrom(
                           bucket.library,
                           "hmx_qk_i8_operator_execute_raw_hn",
                           bucket.execute_raw_hn, error)
                    || !loadSymbolFrom(bucket.library,
                                       "hmx_qk_i8_operator_begin",
                                       bucket.begin, error)
                    || !loadSymbolFrom(bucket.library,
                                       "hmx_qk_i8_operator_end",
                                       bucket.end, error)
                    || !loadSymbolFrom(bucket.library,
                                       "hmx_i8_status_string",
                                       bucket.status_string, error)) {
                    (void)dlclose(bucket.library);
                    bucket.library = nullptr;
                    break;
                }
                (void)loadOptionalSymbolFrom(
                    bucket.library,
                    "hmx_qk_i8_operator_execute_raw_scaled_hn",
                    bucket.execute_raw_scaled_hn);
                (void)loadOptionalSymbolFrom(
                    bucket.library,
                    "hmx_qk_i8_operator_prepare_execute_raw_hn",
                    bucket.prepare_execute_raw_hn);
                (void)loadOptionalSymbolFrom(
                    bucket.library,
                    "hmx_qk_i8_operator_prepare_execute_raw_per_head_hn",
                    bucket.prepare_execute_raw_per_head_hn);
                (void)loadOptionalSymbolFrom(
                    bucket.library,
                    "hmx_qk_i8_operator_prepare_execute_raw_per_head_requant_hn",
                    bucket.prepare_execute_raw_per_head_requant_hn);
                (void)loadOptionalSymbolFrom(
                    bucket.library,
                    "hmx_qk_i8_operator_cpu_pack_ready_data",
                    bucket.cpu_pack_ready_data);
                (void)loadOptionalSymbolFrom(
                    bucket.library,
                    "hmx_qk_i8_operator_execute_cpu_packed_per_head_hn",
                    bucket.execute_cpu_packed_per_head_hn);
                (void)loadOptionalSymbolFrom(
                    bucket.library,
                    "hmx_qk_i8_operator_prepare_raw_key_scaled_hn",
                    bucket.prepare_raw_key_scaled);
                (void)loadOptionalSymbolFrom(
                    bucket.library,
                    "hmx_qk_i8_operator_execute_raw_dynamic_hn",
                    bucket.execute_raw_dynamic_hn);
                (void)loadOptionalSymbolFrom(
                    bucket.library,
                    "hmx_qk_i8_operator_topk_indices_data",
                    bucket.topk_indices_data);
                (void)loadOptionalSymbolFrom(
                    bucket.library,
                    "hmx_qk_i8_operator_execute_raw_i32_topk_hn",
                    bucket.execute_raw_i32_topk_hn);
                bucket.info = {};
                bucket.info.struct_size = sizeof(bucket.info);
                const int status = get_info(&bucket.info);
                bucket.context_key =
                    std::filesystem::path(bucket.path).parent_path().string()
                    + "|h=" + std::to_string(bucket.info.heads)
                    + "|m=" + std::to_string(bucket.info.m)
                    + "|k=" + std::to_string(bucket.info.k)
                    + "|n=" + std::to_string(bucket.info.n);
                const float expected_requant = bucket.info.q_scale
                    * bucket.info.k_scale / bucket.info.output_scale;
                if (api_version() != kApiVersion || status != 0
                    || bucket.info.struct_size != sizeof(bucket.info)
                    || bucket.info.heads != heads
                    || bucket.info.m != info_.m
                    || bucket.info.k != info_.k
                    || bucket.info.n != info_.n
                    || !nearlyEqual(bucket.info.q_scale, q_scale)
                    || !nearlyEqual(bucket.info.k_scale, k_scale)
                    || !nearlyEqual(bucket.info.output_scale, output_scale)
                    || (bucket.info.flags & kRequiredDirectFlags)
                           != kRequiredDirectFlags
                    || !nearlyEqual(bucket.info.requant_scale,
                                    expected_requant)) {
                    error = "INT8 HMX bucket metadata mismatch at manifest row "
                        + std::to_string(line_number);
                    (void)dlclose(bucket.library);
                    bucket.library = nullptr;
                    break;
                }
                loaded.push_back(std::move(bucket));
            } catch (const std::exception &exception) {
                error = "invalid INT8 HMX bucket manifest row "
                    + std::to_string(line_number) + ": " + exception.what();
                break;
            }
        }
        if (!error.empty()) {
            for (auto &bucket : loaded) {
                if (bucket.library != nullptr) (void)dlclose(bucket.library);
            }
            return false;
        }
        std::sort(loaded.begin(), loaded.end(), [](const BucketOperator &left,
                                                   const BucketOperator &right) {
            if (left.info.q_scale != right.info.q_scale) {
                return left.info.q_scale < right.info.q_scale;
            }
            if (left.info.k_scale != right.info.k_scale) {
                return left.info.k_scale < right.info.k_scale;
            }
            if (left.info.output_scale != right.info.output_scale) {
                return left.info.output_scale < right.info.output_scale;
            }
            return left.info.heads < right.info.heads;
        });
        std::vector<std::pair<float, float>> scale_pairs;
        for (const BucketOperator &bucket : loaded) {
            bool found_pair = false;
            for (const auto &pair : scale_pairs) {
                if (nearlyEqual(pair.first, bucket.info.q_scale)
                    && nearlyEqual(pair.second, bucket.info.k_scale)) {
                    found_pair = true;
                    break;
                }
            }
            if (!found_pair) {
                scale_pairs.emplace_back(bucket.info.q_scale,
                                         bucket.info.k_scale);
            }
        }
        if (scale_pairs.empty()) {
            error = "INT8 HMX bucket manifest does not provide any matching "
                "Q/K scale pairs";
            for (auto &bucket : loaded) {
                if (bucket.library != nullptr) (void)dlclose(bucket.library);
            }
            return false;
        }
        bucket_operator_max_heads_ = 0;
        for (const BucketOperator &bucket : loaded) {
            bucket_operator_max_heads_ = std::max(
                bucket_operator_max_heads_, bucket.info.heads);
        }
        bucket_operators_ = std::move(loaded);
        return true;
    }
#endif

    void clearSymbols() noexcept {
        api_version_ = nullptr;
        get_info_ = nullptr;
        create_ = nullptr;
        destroy_ = nullptr;
        matmul_ = nullptr;
        query_data_ = nullptr;
        key_data_ = nullptr;
        key_sums_data_ = nullptr;
        scores_data_ = nullptr;
        raw_query_data_ = nullptr;
        raw_key_data_ = nullptr;
        profile_raw_ = nullptr;
        profile_raw_incremental_ = nullptr;
        last_dsp_timing_ = nullptr;
        execute_hn_ = nullptr;
        prepare_raw_key_ = nullptr;
        execute_raw_hn_ = nullptr;
        begin_ = nullptr;
        end_ = nullptr;
        status_string_ = nullptr;
        raw_path_supported_ = false;
    }

    bool fail(const std::string &message, std::string *error) {
        last_error_ = message;
        if (error != nullptr) *error = message;
        return false;
    }

    bool succeed(std::string *error) {
        last_error_.clear();
        if (error != nullptr) error->clear();
        return true;
    }

    std::size_t reserve_hint_tokens_ = 0;
    bool load_attempted_ = false;
    bool ready_ = false;
    bool raw_path_supported_ = false;
    std::string load_error_;
    std::string last_error_;
    std::string operator_path_;
    std::string context_key_;
#if defined(__ANDROID__)
    void *library_ = nullptr;
#endif
    OperatorInfo info_{};
    ApiVersionFn api_version_ = nullptr;
    GetInfoFn get_info_ = nullptr;
    CreateFn create_ = nullptr;
    DestroyFn destroy_ = nullptr;
    MatmulFn matmul_ = nullptr;
    DataFn query_data_ = nullptr;
    DataFn key_data_ = nullptr;
    Int32DataFn key_sums_data_ = nullptr;
    ConstDataFn scores_data_ = nullptr;
    RawQueryDataFn raw_query_data_ = nullptr;
    RawKeyDataFn raw_key_data_ = nullptr;
    ProfileRawFn profile_raw_ = nullptr;
    ProfileRawIncrementalFn profile_raw_incremental_ = nullptr;
    LastDspTimingFn last_dsp_timing_ = nullptr;
    ExecuteHnFn execute_hn_ = nullptr;
    PrepareRawKeyFn prepare_raw_key_ = nullptr;
    ExecuteRawHnFn execute_raw_hn_ = nullptr;
    BeginFn begin_ = nullptr;
    EndFn end_ = nullptr;
    StatusStringFn status_string_ = nullptr;
    void *context_ = nullptr;
    std::vector<BucketOperator> bucket_operators_;
    int bucket_operator_max_heads_ = 0;
    std::vector<std::int8_t> query_i8_;
    std::vector<std::int8_t> key_i8_;
    std::vector<std::int8_t> group_scores_;
    UninitializedScoreBuffer &scores_;
    int score_row_stride_ = 0;
    std::vector<float> score_requant_scales_;
    std::vector<float> score_q_scales_;
    std::vector<float> score_k_scales_;
    std::vector<float> adaptive_requant_cache_;
    std::vector<std::uint8_t> topk_preselected_;
    std::vector<PackedKeyCache> packed_key_cache_;
    std::vector<LayerPackedKeyCache> layer_packed_key_cache_;
    std::vector<LayerKeyScaleCache> layer_key_scale_cache_;
};

inline bool HMXInt8Selector::preloadLatencyProfile(std::string &error) {
    const bool greedy_schedule = greedyPipelineRequested(error);
    if (!error.empty()) return false;
    return !greedy_schedule || loadLatencyProfile(error) != nullptr;
}

} // namespace mllm

#endif // MLLM_CPU_HMX_INT8_SELECTOR_HPP
