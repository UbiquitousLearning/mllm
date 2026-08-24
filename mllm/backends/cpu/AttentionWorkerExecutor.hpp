#ifndef MLLM_CPU_ATTENTION_WORKER_EXECUTOR_HPP
#define MLLM_CPU_ATTENTION_WORKER_EXECUTOR_HPP

#include "PipelineTaskExecutor.hpp"
#include "Types.hpp"

#include <cstdlib>
#include <cstdio>
#include <mutex>
#include <stdexcept>
#include <string>

namespace mllm {

// Runs explicitly marked dense-attention operators on one process-level,
// persistent worker.  QK, softmax, and PV keep their existing kernels and
// ordering; only the thread that enters those kernels changes.  Sparse HMX
// attention is deliberately not marked because its final stage already owns
// the big-core sparse executor.
class AttentionWorkerExecutor final {
public:
    template <typename Function>
    static ErrorCode run(bool attention_operator, Function &&function) {
        if (!attention_operator || !enabled() || onWorker()) {
            return function();
        }

        ErrorCode result = MLLM_NO_ERROR;
        auto future = executor().submit([&]() {
            WorkerScope scope;
            reportPlacementOnce();
            result = function();
        });
        future.get();
        return result;
    }

    static void warmUp(bool attention_operator) {
        if (!attention_operator || !enabled()) return;
        executor().submit([]() {}).get();
    }

    static bool enabled() {
        static const bool value = []() {
            const char *env = std::getenv("MLLM_CPU_ATTENTION_WORKER");
            if (env == nullptr || env[0] == '\0' || std::string(env) == "0") {
                return false;
            }
            if (std::string(env) != "1") {
                throw std::invalid_argument(
                    "MLLM_CPU_ATTENTION_WORKER must be 0 or 1");
            }
            return true;
        }();
        return value;
    }

private:
    class WorkerScope final {
    public:
        WorkerScope() { onWorker() = true; }
        ~WorkerScope() { onWorker() = false; }
    };

    static bool &onWorker() {
        static thread_local bool value = false;
        return value;
    }

    static PipelineTaskExecutor &executor() {
        static auto *value = new PipelineTaskExecutor(
            "MLLM_CPU_ATTENTION_WORKER_CPU", nullptr,
            "dense attention worker", 1, nullptr, nullptr,
            "MLLM_CPU_ATTENTION_WORKER_SPIN_US");
        return *value;
    }

    static void reportPlacementOnce() {
        static std::once_flag once;
        std::call_once(once, []() {
#if defined(__linux__)
            std::fprintf(stderr, "[CPU_ATTN_WORKER] actual_cpu=%d\n",
                         sched_getcpu());
#else
            std::fprintf(stderr, "[CPU_ATTN_WORKER] actual_cpu=unsupported\n");
#endif
        });
    }
};

} // namespace mllm

#endif // MLLM_CPU_ATTENTION_WORKER_EXECUTOR_HPP
