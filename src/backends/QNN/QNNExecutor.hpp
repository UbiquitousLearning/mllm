#ifndef MLLM_QNNEXECUTOR_H
#define MLLM_QNNEXECUTOR_H
#include "Net.hpp"
#include "Executor.hpp"
#include "Types.hpp"
#include "express/ExpressBase.hpp"
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>

namespace mllm {
class QNNExecutor : public Executor {
public:
    QNNExecutor(ParamLoader *data_loader) :
        Executor(data_loader) {
    }
    ~QNNExecutor() = default;

    /**
     * \brief Setup graphs in net
     * \param net  An instance of the Net class
     */
    void setup(Net *net) override;

    /**
     * \brief Executes the foreword propagation of provided network
     * \param net       An instance of the Net class representing the network to be run
     * \param input_tensors     A vector of input tensors to be processed by the network
     */
    void run(Net *net, vector<shared_ptr<Tensor>> input_tensors) override;

    // used for assigning graph backends execuation
    void run(Context *ctx, Net *net, vector<shared_ptr<Tensor>> input_tensor);

    /**
     * \brief Setup&Executes the foreword propagation of provided network
     * \param net       An instance of the Net class representing the network to be run
     * \param input_tensors     A vector of input tensors to be processed by the network
     *
     * execute(net, input_tensors) is equivalent to setup(net) + run(net, input_tensors)
     */
    void execute(Net *net, vector<shared_ptr<Tensor>> input_tensor) override;

    // void QNNGraphThreadExecute(int id, Net* net);

protected:
    QNNExecutionType executionType_ = PROMPT;

    uint autoregressive_seq_pos_ = 0;

    // int threadVar_[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    // uint threadNum_ = 100;
};

class QNNPipelineExecutor : public QNNExecutor {
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

public:
    QNNPipelineExecutor(ParamLoader *data_loader) :
        QNNExecutor(data_loader) {
    }
    // used for assigning graph backends execuation
    void run(Context *ctx, Net *net, vector<shared_ptr<Tensor>> input_tensors);
};

} // namespace mllm

#endif // MLLM_EXECUTOR_H
