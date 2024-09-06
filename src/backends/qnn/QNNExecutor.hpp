#ifndef MLLM_QNNEXECUTOR_H
#define MLLM_QNNEXECUTOR_H
#include "Net.hpp"
#include "Executor.hpp"
#include "Types.hpp"
#include "express/ExpressBase.hpp"
#include <condition_variable>
#include <iostream>
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
    virtual void run(Context *ctx, Net *net, vector<shared_ptr<Tensor>> input_tensor);
    virtual void runExp(Context *ctx, Net *net, vector<shared_ptr<Tensor>> input_tensor) {};

    /**
     * \brief Setup&Executes the foreword propagation of provided network
     * \param net       An instance of the Net class representing the network to be run
     * \param input_tensors     A vector of input tensors to be processed by the network
     *
     * execute(net, input_tensors) is equivalent to setup(net) + run(net, input_tensors)
     */
    void execute(Net *net, vector<shared_ptr<Tensor>> input_tensor) override {
        std::cerr << "QNNExecutor::execute Not implemented" << std::endl;
    };

    // graph offload rule for qnn execution, used in setup and execution
    static BackendType graphOffloadRule(BackendType expectedBackend, int graphIndex);

    // graph naming rule for qnn execution
    string graphNamingRule(int graphIndex) {
        switch (executionType_) {
        case PROMPT:
            return "Prompt_Graph." + std::to_string(graphIndex);
        case AUTOREGRESSIVE:
            return "Autoregressive_Graph." + std::to_string(graphIndex);
        }
    };

protected:
    QNNExecutionType executionType_ = PROMPT;
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
    void run(Context *ctx, Net *net, vector<shared_ptr<Tensor>> input_tensors) override;
    virtual void runExp(Context *ctx, Net *net, vector<shared_ptr<Tensor>> input_tensor) override;
};

} // namespace mllm

#endif // MLLM_EXECUTOR_H
