#ifndef MLLM_STRASSENMATMUL_HPP
#define MLLM_STRASSENMATMUL_HPP

#include "Backend.hpp"
#include "Types.hpp"
#include "VecDot.hpp"
#include "Matmul.hpp"
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>

namespace mllm{

class StrassenMatmul{
public:
    StrassenMatmul(Backend *bn, int maxDepth, int threadCount);

    ErrorCode onReshape(Tensor *src0, Tensor *src1, Tensor *dst, bool support_bias, Tensor *bias = nullptr, bool transpose0 = false, bool transpose1 = false, int scale1 = 1, int scale2 = 1);
    ErrorCode onExecute();
    void onReset();

private:
    int max_depth_;
    int thread_count;
    Backend *backend_;
    std::vector<std::pair<std::function<void(int tId)>, int>> functions_;

    ErrorCode generateTrivalMatmul(Tensor *src0, Tensor *src1, Tensor *dst, bool support_bias, Tensor *bias, bool transpose0, bool transpose1, float scale1 = 1, float scale2 = 1);
    ErrorCode generateStrassenMatmul(int m, int k, int n, Tensor *src0, Tensor *src1, Tensor *dst, bool support_bias, Tensor *bias, bool transpose0, bool transpose1, float scale1 = 1, float scale2 = 1, int depth = 0);

    class ThreadPool {
    public:
        ThreadPool(size_t num_threads);
        ~ThreadPool();
        void enqueue(std::function<void()> f);

    private:
        std::vector<std::thread> workers_;
        std::queue<std::function<void()>> tasks_;
        std::mutex queue_mutex_;
        std::condition_variable condition_;
        bool stop_;
    };
};

} // namespace mllm

#endif // MLLM_STRASSENMATMUL_HPP