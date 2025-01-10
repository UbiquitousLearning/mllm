#ifndef MLLM_OP_H
#define MLLM_OP_H
// #define DEBUGPRINT
#include "Backend.hpp"
#include "Tensor.hpp"
#include "Types.hpp"
#include <cassert>
#include <functional>
#include <iostream>
#include <memory>
#include "ParamLoader.hpp"
#include "Timing.hpp"
using std::function;
namespace mllm {

class Backend;
class Tensor;
class ParamLoader;

class Op {
public:
    /**
     * @brief initializer.
     * @param bn   backend that exection will running on.
     * @param name name of Op.
     */
    Op(Backend *bn, string name = "") :
        backend_(bn), name_(name) {
        // nothing to do
    }
    virtual ~Op() = default;

    /**
     * @brief set the shape of output tensors.
     * @param inputs    input tensors
     * @param outputs   output tensors
     * @return MLLM_NO_ERROR
     */
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
#ifdef DEBUGPRINT
        std::cout << "" << name() << "     reshape:";
        std::cout << "\n    || ";
        for (auto input : inputs) {
            std::cout << "Input " << input->name() << " shape: " << input->shapeString() << " |";
        }
        std::cout << "\n    || ";
        for (auto output : outputs) {
            std::cout << "Output " << output->name() << " shape: " << output->shapeString() << " |";
        }
        std::cout << std::endl;
#endif
        return MLLM_NO_ERROR;
    }

    /**
     * @brief alloc the memory of output tensors.
     * @param inputs    input tensors
     * @param outputs   output tensors
     * @return MLLM_NO_ERROR
     */
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
        for (auto &output : outputs) {
            output->setDtype(activation_dtype_);
            output->alloc();
        }
        return MLLM_NO_ERROR;
    }

    /**
     * \brief load the weights/bias of this Op.
     * \param loader A Paramloader
     * \return MLLM_NO_ERROR
     */
    virtual ErrorCode load(AbstructLoader &loader) {
#ifdef DEBUGPRINT
        std::cout << "" << name() << "      load" << std::endl;
#endif
        return MLLM_NO_ERROR;
    }

    /**
     * @brief perform execution.
     * @param inputs    input tensors
     * @param outputs   output tensors
     * @return MLLM_NO_ERROR
     */
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
        return MLLM_NO_ERROR;
    }

    /**
     * @brief perform free.
     * @param inputs    input tensors
     * @param outputs   output tensors
     * @return MLLM_NO_ERROR
     */
    virtual ErrorCode free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
        return MLLM_NO_ERROR;
    }

    Backend *backend() const {
        return backend_;
    }
    string name() const {
        return name_;
    }
    void setName(string name) {
        name_ = name;
    }
    DataType activation_dtype() const {
        return activation_dtype_;
    }
    OpType type() const {
        return type_;
    }
    void setOpType(OpType type) {
        type_ = type;
    }

    virtual int getCacheSeqLen() {
        assert(type_ == OpType::KVCACHE || type_ == OpType::KVCACHENPU);
        std::cout << "only for KVCache" << std::endl;
        return -1;
    }
    virtual void clearCache() {
        assert(type_ == OpType::KVCACHE || type_ == OpType::KVCACHENPU || type_ == OpType::IROPE || type_ == OpType::ROPE);
        std::cout << "only for KVCache" << std::endl;
    }

    static DataType &noLoadWeightsDtype() {
        return no_load_weights_dtype_;
    }

protected:
    Backend *backend_;
    vector<Tensor *> inputs_;
    vector<Tensor *> outputs_;
    string name_;
    DataType activation_dtype_ = MLLM_TYPE_F32;
    OpType type_;
    static DataType no_load_weights_dtype_;
};

class Callable {
public:
    CallableType type_;
    vector<shared_ptr<Tensor>> opInputs;
    vector<shared_ptr<Tensor>> opOutputs;
    Op *op;
    vector<Tensor *> tensorInputs;
    vector<Tensor *> tensorOutputs;
    vector<float> args;
    TensorFunction *tensorFunc;

    Callable(CallableType type) :
        type_(type) {
    }

    void reshape(){
        if (type_ == CallableType::OP) {
            op->reshape(opInputs, opOutputs);
        }
    }

    void setUp() {
        if (type_ == CallableType::OP) {
            op->setUp(opInputs, opOutputs);
        } else {
            tensorFunc->setup(tensorOutputs, tensorInputs, args);
        }
    }

    void execute() {
        if (type_ == CallableType::OP) {
            op->execute(opInputs, opOutputs);
        } else {
            tensorFunc->execute(tensorOutputs, tensorInputs, args);
        }
    }

    vector<shared_ptr<Tensor>> outputs() {
        if (type_ == CallableType::OP) {
            return opOutputs;
        } else {
            vector<shared_ptr<Tensor>> outputs;
            for (auto tensor : tensorOutputs) {
                outputs.push_back(std::shared_ptr<Tensor>(tensor, [](Tensor *) {}));
            }
            return outputs;
        }
    }
};

} // namespace mllm

#endif // MLLM_OP_H