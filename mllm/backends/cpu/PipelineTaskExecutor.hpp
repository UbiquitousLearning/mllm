#ifndef MLLM_CPU_PIPELINE_TASK_EXECUTOR_HPP
#define MLLM_CPU_PIPELINE_TASK_EXECUTOR_HPP

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <future>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#endif

namespace mllm {

// Process-level persistent workers for a single attention stage. Submitting a
// task never creates a thread; all layers share the same executor instance.
class PipelineTaskExecutor final {
public:
    PipelineTaskExecutor(const char *pin_env_name,
                         const char *thread_count_env_name,
                         const char *stage_name, int requested_threads,
                         const char *tail_pin_env_name = nullptr,
                         const char *tail_thread_count_env_name = nullptr,
                         const char *spin_microseconds_env_name = nullptr,
                         const char *event_wait_env_name = nullptr)
        : pin_env_name_(pin_env_name),
          thread_count_env_name_(thread_count_env_name),
          stage_name_(stage_name == nullptr ? "pipeline" : stage_name),
          requested_threads_(std::max(1, requested_threads)),
          tail_pin_env_name_(tail_pin_env_name),
          tail_thread_count_env_name_(tail_thread_count_env_name),
          spin_microseconds_env_name_(spin_microseconds_env_name) {
        (void)event_wait_env_name;
    }

    ~PipelineTaskExecutor() { shutdown(); }

    PipelineTaskExecutor(const PipelineTaskExecutor &) = delete;
    PipelineTaskExecutor &operator=(const PipelineTaskExecutor &) = delete;

    std::future<void> submit(std::function<void()> task) {
        if (!task) {
            return exceptionalFuture(std::make_exception_ptr(
                std::invalid_argument(stage_name_ + " received empty task")));
        }
        auto job = std::make_shared<Job>(std::move(task));
        auto future = job->completion.get_future();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopped_) {
                job->completion.set_exception(std::make_exception_ptr(
                    std::runtime_error(stage_name_ + " is stopped")));
                return future;
            }
            try {
                ensureStarted();
            } catch (...) {
                job->completion.set_exception(std::current_exception());
                return future;
            }
            jobs_.push_back(job);
        }
        cv_.notify_one();
        return future;
    }

    void shutdown() noexcept {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopped_) return;
            stopped_ = true;
        }
        cv_.notify_all();
        for (auto &worker : workers_) {
            if (worker.joinable()) worker.join();
        }
    }

private:
    struct Job {
        explicit Job(std::function<void()> value) : task(std::move(value)) {}
        std::function<void()> task;
        std::promise<void> completion;
    };

    static std::future<void> exceptionalFuture(std::exception_ptr error) {
        std::promise<void> promise;
        auto future = promise.get_future();
        promise.set_exception(error);
        return future;
    }

    static int readCount(const char *name, int fallback,
                         bool allow_zero = false, long maximum = 64) {
        if (name == nullptr) return fallback;
        const char *value = std::getenv(name);
        if (value == nullptr || value[0] == '\0') return fallback;
        char *end = nullptr;
        const long parsed = std::strtol(value, &end, 10);
        const long minimum = allow_zero ? 0 : 1;
        if (end == value || *end != '\0' || parsed < minimum
            || parsed > maximum) {
            throw std::invalid_argument(
                std::string(name) + " must be in ["
                + std::to_string(minimum) + ", "
                + std::to_string(maximum) + "]");
        }
        return static_cast<int>(parsed);
    }

    static std::vector<int> readCpus(const char *name, int count) {
        if (name == nullptr || count == 0) return {};
        const char *value = std::getenv(name);
        if (value == nullptr || value[0] == '\0') return {};
        std::vector<int> cpus;
        const char *cursor = value;
        while (*cursor != '\0') {
            char *end = nullptr;
            const long parsed = std::strtol(cursor, &end, 10);
#if defined(__linux__)
            const bool valid = end != cursor && parsed >= 0
                && parsed < CPU_SETSIZE;
#else
            const bool valid = end != cursor && parsed >= 0
                && parsed <= std::numeric_limits<int>::max();
#endif
            if (!valid || (*end != '\0' && *end != ',')) {
                throw std::invalid_argument(
                    std::string(name) + " must contain valid CPU ids");
            }
            cpus.push_back(static_cast<int>(parsed));
            if (*end == '\0') break;
            cursor = end + 1;
        }
        if (cpus.size() != 1U
            && cpus.size() != static_cast<std::size_t>(count)) {
            throw std::invalid_argument(
                std::string(name) + " needs one CPU or one per worker");
        }
        return cpus;
    }

    static void pinCurrentThread(int cpu, const std::string &stage) {
        if (cpu < 0) return;
#if defined(__linux__)
        cpu_set_t affinity;
        CPU_ZERO(&affinity);
        CPU_SET(cpu, &affinity);
        if (sched_setaffinity(0, sizeof(affinity), &affinity) != 0) {
            throw std::runtime_error(
                "failed to pin " + stage + " to CPU " + std::to_string(cpu)
                + ": " + std::strerror(errno));
        }
#else
        (void)stage;
        throw std::runtime_error("CPU pinning requires Linux/Android");
#endif
    }

    int spinMicroseconds() const {
        return readCount(spin_microseconds_env_name_, 0, true, 10000000);
    }

    void ensureStarted() {
        if (started_) return;
        const int primary_count = readCount(
            thread_count_env_name_, requested_threads_);
        const int tail_count = readCount(
            tail_thread_count_env_name_, 0, true);
        auto primary_cpus = readCpus(pin_env_name_, primary_count);
        auto tail_cpus = readCpus(tail_pin_env_name_, tail_count);
        const int spin_us = spinMicroseconds();
        const auto start_group = [&](int count, const std::vector<int> &cpus) {
            for (int worker = 0; worker < count; ++worker) {
                const int cpu = cpus.empty() ? -1
                    : cpus[cpus.size() == 1U ? 0U
                                              : static_cast<std::size_t>(worker)];
                workers_.emplace_back(
                    [this, cpu, spin_us]() { workerLoop(cpu, spin_us); });
            }
        };
        start_group(primary_count, primary_cpus);
        start_group(tail_count, tail_cpus);
        started_ = true;
    }

    void workerLoop(int cpu, int spin_us) noexcept {
        std::exception_ptr initialization_error;
        try {
            pinCurrentThread(cpu, stage_name_);
        } catch (...) {
            initialization_error = std::current_exception();
        }
        while (true) {
            std::shared_ptr<Job> job;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                if (jobs_.empty() && !stopped_ && spin_us > 0) {
                    lock.unlock();
                    const auto deadline = std::chrono::steady_clock::now()
                        + std::chrono::microseconds(spin_us);
                    do {
                        std::this_thread::yield();
                        lock.lock();
                        const bool ready = stopped_ || !jobs_.empty();
                        lock.unlock();
                        if (ready) break;
                    } while (std::chrono::steady_clock::now() < deadline);
                    lock.lock();
                }
                cv_.wait(lock, [&]() { return stopped_ || !jobs_.empty(); });
                if (stopped_ && jobs_.empty()) return;
                job = std::move(jobs_.front());
                jobs_.pop_front();
            }
            try {
                if (initialization_error) {
                    std::rethrow_exception(initialization_error);
                }
                job->task();
                job->completion.set_value();
            } catch (...) {
                try {
                    job->completion.set_exception(std::current_exception());
                } catch (...) {
                }
            }
        }
    }

    const char *pin_env_name_;
    const char *thread_count_env_name_;
    std::string stage_name_;
    int requested_threads_;
    const char *tail_pin_env_name_;
    const char *tail_thread_count_env_name_;
    const char *spin_microseconds_env_name_;
    bool started_ = false;
    bool stopped_ = false;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<std::shared_ptr<Job>> jobs_;
    std::vector<std::thread> workers_;
};

} // namespace mllm

#endif // MLLM_CPU_PIPELINE_TASK_EXECUTOR_HPP
