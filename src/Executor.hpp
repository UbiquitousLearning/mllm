#ifndef MLLM_EXECUTOR_H
#define MLLM_EXECUTOR_H
#include "Net.hpp"
#include <numeric>

namespace mllm {
class Executor {
public:
    Executor(ParamLoader *data_loader) :
        data_loader_(data_loader) {
    }
    ~Executor() = default;

    /**
     * \brief Setup graphs in net
     * \param net  An instance of the Net class
     */
    virtual void setup(Net *net);

    /**
     * \brief Executes the foreword propagation of provided network
     * \param net       An instance of the Net class representing the network to be run
     * \param input_tensors     A vector of input tensors to be processed by the network
     */
    virtual void run(Net *net, vector<shared_ptr<Tensor>> input_tensors);

    /**
     * \brief Setup&Executes the foreword propagation of provided network
     * \param net       An instance of the Net class representing the network to be run
     * \param input_tensors     A vector of input tensors to be processed by the network
     *
     * execute(net, input_tensors) is equivalent to setup(net) + run(net, input_tensors)
     */
    virtual void execute(Net *net, vector<shared_ptr<Tensor>> input_tensor);

    bool checkSame(vector<shared_ptr<Tensor>> input_tensor) {
        if (input_tensor.size() != input_sizes_.size()) {
            return false;
        }
        bool same = true;
        for (int i = 0; i < input_tensor.size(); ++i) {
            if (input_tensor[i]->shape() != input_sizes_[i]) {
                same = false;
                break;
            }
        }
        return same;
    }
    /**
     * \brief Checks whether the input tensors have the same shape as the previous input tensors.
     *        Change init & reshape flags accordingly.
     * \param init    whether to initialize the input_sizes_ vector
     * \param reshape    whether to reshape the input tensors
     * \param input_tensor   A vector of input tensors to be processed by the network
     * \return
     */
    bool checkReshape(bool &init, bool &reshape, vector<shared_ptr<Tensor>> input_tensor) {
        if (input_sizes_.empty()) {
            for (auto &t : input_tensor) {
                input_sizes_.push_back(t->shape());
            }
            init = true;
        } else if (checkSame(input_tensor)) {
            reshape = false;
        } else {
            input_sizes_.clear();
            for (auto &t : input_tensor) {
                input_sizes_.push_back(t->shape());
            }
            reshape = true;
        }
        return init || reshape;
    }

    vector<shared_ptr<Tensor>> &result() {
        return result_;
    }

    void perf() const {
        std::cout << "load time: " << load_time_ << " ms" << std::endl;
        double sum_time = std::accumulate(std::begin(run_time_), std::end(run_time_), 0.0);
        double mean_time = sum_time / run_time_.size();
        std::cout << "token time: " << mean_time << " ms" << std::endl;
        std::cout << "inference speed: " << 1000 / mean_time << " tokens/s" << std::endl;
    }

protected:
    vector<vector<int>> input_sizes_;
    vector<shared_ptr<Tensor>> result_;
    ParamLoader *data_loader_;

    double load_time_ = 0;
    vector<double> run_time_;

    bool paramloaded = false;
    bool freeGraph = false;
};

} // namespace mllm

#endif // MLLM_EXECUTOR_H
