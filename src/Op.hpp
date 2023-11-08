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
    Op(Backend *bn, string name = "") :
        backend_(bn) , name_(name) {
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
#ifdef TEST
        std::cout << "*"<<name_<<" reshape*" << std::endl;
        for (auto input:inputs) {
            std::cout << "Input "<< input->name() <<" shape: " << input->ShapeString() << std::endl;
        }
        for (auto output:outputs) {
            std::cout << "Output "<< output->name() <<" shape: " << output->ShapeString() << std::endl;
        }
        //std::cout << "*"<<name_<<" reshape*" << std::endl;
#elif DEBUG
        std::cout << "*"<<name_<<" reshape*" << std::endl;
#endif
        return NO_ERROR;
    }
    /*
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
     */

    /**
     * @brief response shape change of input or output tensors.
     * 设定输入输出的tensor(已经to_cpu)
     * @param inputs    input tensors
     * @param outputs   output tensors
     * @return reshapeOutputs result
     */
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
        for (auto &input :inputs) {
            if (!input->allocted()) {
                input->alloc(); // TODO remove
            }
        }
        for (auto &output :outputs) {
            output->setDtype(activation_dtype_);
            output->alloc();
        }
#ifdef DEBUG
        std::cout << "*"<<name_<<" setUp*" << std::endl;
#endif
        return NO_ERROR;
    }

    virtual ErrorCode load(ParamLoader &loader) {
        // check inputs shape
        // reshape outputs
#ifdef DEBUG
        std::cout << "*"<<name_<<" load*" << std::endl;
#endif
        return NO_ERROR;
    }
    /*
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs, ParamLoader &loader) {
        setUp(inputs, outputs);
        load(loader);
        return NO_ERROR;
    }*/

    /**
     * @brief perform execution.
     * @param inputs    input tensors
     * @param outputs   output tensors
     * @return execution result
     */
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
#ifdef DEBUG
        std::cout << "*"<<name_<<" execute*" << std::endl;
#endif
        return NO_ERROR;
    }

    /**
     * @brief perform free.
     * @param inputs    input tensors
     * @param outputs   output tensors
     * @return execution result
     */
    virtual ErrorCode free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
#ifdef DEBUG
        std::cout << "*"<<name_<<" free*" << std::endl;
#endif
        return NO_ERROR;
    }

//    virtual ErrorCode setDtype(DataType activation_dtype) {
//        activation_dtype_ = activation_dtype;
//        return NO_ERROR;
//    }
//    DataType activationDtype() const {
//        return activation_dtype_;
//    }
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
    DataType activation_dtype_ = MLLM_TYPE_F32;
};

// unordered_map<OpType, function<shared_ptr<Op>(Backend*)>> opMap;
} // namespace mllm

#endif // MLLM_OP_H