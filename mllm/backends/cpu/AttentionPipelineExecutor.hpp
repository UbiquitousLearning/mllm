#ifndef MLLM_CPU_ATTENTION_PIPELINE_EXECUTOR_HPP
#define MLLM_CPU_ATTENTION_PIPELINE_EXECUTOR_HPP

#include <algorithm>
#include <atomic>
#include <cerrno>
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

// A persistent gang executor. Every submitted job is split across all workers;
// workers synchronize at the end of the job before any worker may start the
// next one. This gives the pipeline one logical Top-k lane while allowing the
// lane to use multiple little cores cooperatively.
class CooperativeTaskExecutor final {
public:
    using Task = std::function<void(std::size_t, std::size_t)>;

    CooperativeTaskExecutor(int worker_count, std::vector<int> worker_cpus,
                            std::string stage_name)
        : worker_count_(worker_count), worker_cpus_(std::move(worker_cpus)),
          stage_name_(std::move(stage_name)) {
        validateConfiguration();
    }

    CooperativeTaskExecutor(const char *pin_env_name,
                            const char *worker_env_name,
                            const char *stage_name, int fallback_workers)
        : worker_count_(readWorkerCount(worker_env_name, fallback_workers)),
          worker_cpus_(readWorkerCpus(pin_env_name, worker_count_)),
          stage_name_(stage_name == nullptr ? "cooperative" : stage_name) {
        validateConfiguration();
    }

    ~CooperativeTaskExecutor() { shutdown(); }

    CooperativeTaskExecutor(const CooperativeTaskExecutor &) = delete;
    CooperativeTaskExecutor &operator=(const CooperativeTaskExecutor &) = delete;

    std::future<void> submit(Task task,
                             std::function<void()> on_complete = {}) {
        if (!task) {
            std::promise<void> promise;
            auto future = promise.get_future();
            promise.set_exception(std::make_exception_ptr(
                std::invalid_argument(stage_name_ + " received empty task")));
            return future;
        }
        std::lock_guard<std::mutex> submit_lock(submit_mutex_);
        try {
            ensureStarted();
        } catch (...) {
            std::promise<void> promise;
            auto future = promise.get_future();
            promise.set_exception(std::current_exception());
            return future;
        }
        auto job = std::make_shared<Job>(
            std::move(task), std::move(on_complete), worker_count_);
        auto future = job->completion.get_future();
        for (std::size_t worker = 0; worker < queues_.size(); ++worker) {
            Queue &queue = *queues_[worker];
            {
                std::lock_guard<std::mutex> lock(queue.mutex);
                if (stopped_) {
                    job->setException(std::make_exception_ptr(
                        std::runtime_error(stage_name_ + " is stopped")));
                    job->finishOne();
                    continue;
                }
                queue.jobs.push_back(job);
            }
            queue.cv.notify_one();
        }
        return future;
    }

    void shutdown() noexcept {
        {
            std::lock_guard<std::mutex> submit_lock(submit_mutex_);
            if (stopped_) return;
            stopped_ = true;
            for (auto &queue : queues_) {
                std::lock_guard<std::mutex> lock(queue->mutex);
                queue->cv.notify_all();
            }
        }
        for (std::thread &worker : workers_) {
            if (worker.joinable()) worker.join();
        }
    }

    int workerCount() const noexcept { return worker_count_; }

private:
    struct Job {
        Job(Task task_value, std::function<void()> completion_callback,
            int worker_count)
            : task(std::move(task_value)),
              on_complete(std::move(completion_callback)),
              remaining(worker_count) {}

        void setException(std::exception_ptr value) noexcept {
            std::lock_guard<std::mutex> lock(mutex);
            if (!exception) exception = value;
        }

        void finishOne() noexcept {
            if (remaining.fetch_sub(1) != 1) return;
            std::exception_ptr completion_error;
            {
                std::lock_guard<std::mutex> lock(mutex);
                completion_error = exception;
            }
            if (!completion_error && on_complete) {
                try {
                    on_complete();
                } catch (...) {
                    completion_error = std::current_exception();
                }
            }
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (!exception) exception = completion_error;
                completed = true;
                try {
                    if (exception) completion.set_exception(exception);
                    else completion.set_value();
                } catch (...) {
                }
            }
            cv.notify_all();
        }

        void waitForPeers() noexcept {
            std::unique_lock<std::mutex> lock(mutex);
            cv.wait(lock, [&]() { return completed; });
        }

        Task task;
        std::function<void()> on_complete;
        std::atomic<int> remaining;
        std::promise<void> completion;
        std::mutex mutex;
        std::condition_variable cv;
        std::exception_ptr exception;
        bool completed = false;
    };

    struct Queue {
        std::mutex mutex;
        std::condition_variable cv;
        std::deque<std::shared_ptr<Job>> jobs;
    };

    static int readWorkerCount(const char *env_name, int fallback) {
        if (env_name == nullptr) return std::max(1, fallback);
        const char *value = std::getenv(env_name);
        if (value == nullptr || value[0] == '\0') return std::max(1, fallback);
        char *end = nullptr;
        const long parsed = std::strtol(value, &end, 10);
        if (end == value || *end != '\0' || parsed < 1 || parsed > 64) {
            throw std::invalid_argument(
                std::string(env_name) + " must be in [1, 64]");
        }
        return static_cast<int>(parsed);
    }

    static std::vector<int> readWorkerCpus(const char *env_name,
                                           int worker_count) {
        if (env_name == nullptr) return {};
        const char *value = std::getenv(env_name);
        if (value == nullptr || value[0] == '\0') return {};
        std::vector<int> cpus;
        const char *cursor = value;
        while (*cursor != '\0') {
            char *end = nullptr;
            const long parsed = std::strtol(cursor, &end, 10);
#if defined(__linux__)
            if (end == cursor || parsed < 0 || parsed >= CPU_SETSIZE) {
#else
            if (end == cursor || parsed < 0
                || parsed > std::numeric_limits<int>::max()) {
#endif
                throw std::invalid_argument(
                    std::string(env_name)
                    + " must contain valid comma-separated CPU ids");
            }
            cpus.push_back(static_cast<int>(parsed));
            if (*end == '\0') break;
            if (*end != ',' || end[1] == '\0') {
                throw std::invalid_argument(
                    std::string(env_name)
                    + " must contain valid comma-separated CPU ids");
            }
            cursor = end + 1;
        }
        if (cpus.size() != 1U
            && cpus.size() != static_cast<std::size_t>(worker_count)) {
            throw std::invalid_argument(
                std::string(env_name)
                + " must contain either one CPU id or one id per worker");
        }
        return cpus;
    }

    void validateConfiguration() {
        if (worker_count_ <= 0 || worker_count_ > 64) {
            throw std::invalid_argument(
                "cooperative worker count must be in [1, 64]");
        }
        if (!worker_cpus_.empty() && worker_cpus_.size() != 1U
            && worker_cpus_.size()
                != static_cast<std::size_t>(worker_count_)) {
            throw std::invalid_argument(
                "cooperative CPU list must have one CPU or one per worker");
        }
    }

    static void pinWorker(int cpu, const std::string &stage_name) {
#if defined(__linux__)
        if (cpu < 0) return;
        cpu_set_t affinity;
        CPU_ZERO(&affinity);
        CPU_SET(cpu, &affinity);
        if (sched_setaffinity(0, sizeof(affinity), &affinity) != 0) {
            throw std::runtime_error(
                "failed to pin " + stage_name + " to CPU "
                + std::to_string(cpu) + ": " + std::strerror(errno));
        }
#else
        (void)cpu;
        (void)stage_name;
        throw std::runtime_error(
            "cooperative CPU pinning requires Linux/Android");
#endif
    }

    void ensureStarted() {
        if (started_) return;
        if (stopped_) {
            throw std::runtime_error(stage_name_ + " is stopped");
        }
        queues_.reserve(static_cast<std::size_t>(worker_count_));
        workers_.reserve(static_cast<std::size_t>(worker_count_));
        for (int worker = 0; worker < worker_count_; ++worker) {
            queues_.push_back(std::make_unique<Queue>());
        }
        for (int worker = 0; worker < worker_count_; ++worker) {
            const int cpu = worker_cpus_.empty() ? -1
                : worker_cpus_[worker_cpus_.size() == 1U
                                   ? 0U : static_cast<std::size_t>(worker)];
            workers_.emplace_back([this, worker, cpu]() {
                try {
                    pinWorker(cpu, stage_name_);
                } catch (...) {
                    setInitializationError(std::current_exception());
                }
                Queue &queue = *queues_[static_cast<std::size_t>(worker)];
                while (true) {
                    std::shared_ptr<Job> job;
                    {
                        std::unique_lock<std::mutex> lock(queue.mutex);
                        queue.cv.wait(lock, [&]() {
                            return stopped_ || !queue.jobs.empty();
                        });
                        if (stopped_ && queue.jobs.empty()) break;
                        if (queue.jobs.empty()) continue;
                        job = std::move(queue.jobs.front());
                        queue.jobs.pop_front();
                    }
                    try {
                        std::exception_ptr init_error;
                        {
                            std::lock_guard<std::mutex> lock(init_mutex_);
                            init_error = initialization_error_;
                        }
                        if (init_error) std::rethrow_exception(init_error);
                        job->task(static_cast<std::size_t>(worker),
                                  static_cast<std::size_t>(worker_count_));
                    } catch (...) {
                        job->setException(std::current_exception());
                    }
                    job->finishOne();
                    job->waitForPeers();
                }
            });
        }
        started_ = true;
    }

    void setInitializationError(std::exception_ptr error) noexcept {
        std::lock_guard<std::mutex> lock(init_mutex_);
        if (!initialization_error_) initialization_error_ = error;
    }

    int worker_count_ = 1;
    std::vector<int> worker_cpus_;
    std::string stage_name_;
    bool started_ = false;
    std::atomic<bool> stopped_{false};
    std::mutex submit_mutex_;
    std::mutex init_mutex_;
    std::exception_ptr initialization_error_;
    std::vector<std::unique_ptr<Queue>> queues_;
    std::vector<std::thread> workers_;
};

} // namespace mllm

#endif // MLLM_CPU_ATTENTION_PIPELINE_EXECUTOR_HPP
