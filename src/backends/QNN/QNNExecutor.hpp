#ifndef MLLM_QNNEXECUTOR_H
#define MLLM_QNNEXECUTOR_H
#include "Net.hpp"
#include "Executor.hpp"
#include "Types.hpp"
#include <numeric>

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

    /**
     * \brief Setup&Executes the foreword propagation of provided network
     * \param net       An instance of the Net class representing the network to be run
     * \param input_tensors     A vector of input tensors to be processed by the network
     *
     * execute(net, input_tensors) is equivalent to setup(net) + run(net, input_tensors)
     */
    void execute(Net *net, vector<shared_ptr<Tensor>> input_tensor) override;

private:
    QNNExecutionType executionType_ = PROMPT;

};

} // namespace mllm

#endif // MLLM_EXECUTOR_H
