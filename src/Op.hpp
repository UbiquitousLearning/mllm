#ifndef MLLM_OP_H
#define MLLM_OP_H

#include "Tensor.hpp"
#include "Types.hpp"
#include <functional>
#include "ParamLoader.hpp"
using std::function;
namespace mllm {

class Backend;
class Tensor;
class ParamLoader;

class Op {
public:
    /**
     * @brief initializer.
     * @param backend   backend that exection will running on.
     */
    Op(){};
    // Op(const BackendType betype): backend_type_(betype) {};
    // Op() = delete;
    // Op(shared_ptr<Backend> bn) : backend_(bn) {
    //     // nothing to do
    // }
    Op(Backend *bn) :
        backend_(bn) {
        // nothing to do
    }
    virtual ~Op() = default;

    /**
     * @brief response shape change of input or output tensors.
     * 设定输入输出的tensor(已经to_cpu)
     * @param inputs    input tensors
     * @param outputs   output tensors
     * @return reshapeOutputs result
     */
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
        // check inputs shape
        // reshape outputs
        return NO_ERROR;
    }
    virtual ErrorCode reshapeOutputs(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
        // check inputs shape
        // reshape outputs
        reshape(inputs, outputs);
        {
            for (auto &t : outputs) {
                t->alloc();
            }
        }
        return NO_ERROR;
    }

    /**
     * @brief response shape change of input or output tensors.
     * 设定输入输出的tensor(已经to_cpu)
     * @param inputs    input tensors
     * @param outputs   output tensors
     * @return reshapeOutputs result
     */
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
        //        for (auto &t : outputs) {
        // t->SetName("Input0"+"_out");
        //        }
        // Weight malloc set
        return NO_ERROR;
    }

    virtual ErrorCode load(ParamLoader &loader) {
        // check inputs shape
        // reshape outputs
        return NO_ERROR;
    }

    /**
     * @brief perform execution.
     * @param inputs    input tensors
     * @param outputs   output tensors
     * @return execution result
     */
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) = 0;
    /**
     * @brief get backend.
     * @return backend.
     */
    // shared_ptr<Backend> backend() const {
    //     return backend_;
    // }
    Backend *backend() const {
        return backend_;
    }
    string name() const {
        return name_;
    }
    void setName(string name) {
        name_ = name;
    }

private:
    // shared_ptr<Backend> backend_;
    Backend *backend_;
    vector<Tensor *> inputs_;
    vector<Tensor *> outputs_;
    string name_;
    // BackendType backend_type_;
    // tensor w
    // vector<>
};

// unordered_map<OpType, function<shared_ptr<Op>(Backend*)>> opMap;
} // namespace mllm

#endif // MLLM_OP_H