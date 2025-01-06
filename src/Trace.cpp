#include "Trace.hpp"
#include "Module.hpp"
#include "Tensor.hpp"
#include "Types.hpp"
#include <memory>

namespace mllm {

vector<std::shared_ptr<Module>> Tracer::model_;
bool Tracer::isToCreateNewModule = true;

void Tracer::addModule(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs, string name) {
    model_.push_back(std::make_shared<QNNModuleWrapper>());
    auto *wrapperPtr = static_cast<QNNModuleWrapper *>(model_.back().get());
    wrapperPtr->name_ = name;
    wrapperPtr->inputs_ = inputs;
    wrapperPtr->outputs_ = outputs;
    isToCreateNewModule = true;
}

void Tracer::addOp(Op *op, vector<std::shared_ptr<Tensor>> inputs, vector<std::shared_ptr<Tensor>> outputs) {
    if (isToCreateNewModule) {
        model_.push_back(std::make_shared<CPUModuleWrapper>());
        isToCreateNewModule = false;
    }
    auto wrapper_module = std::dynamic_pointer_cast<CPUModuleWrapper>(model_.back());
    assert(wrapper_module != nullptr);
    wrapper_module->addOp(op, inputs, outputs);
}

void Tracer::addTensorFunction(TensorFunction *func,
                               vector<Tensor *> inputs, vector<Tensor *> outputs, vector<float> args) {
    if (isToCreateNewModule) {
        model_.push_back(std::make_shared<CPUModuleWrapper>());
        isToCreateNewModule = false;
    }
    // cast model_.back() to wrapper module
    auto wrapper_module = std::dynamic_pointer_cast<CPUModuleWrapper>(model_.back());
    assert(wrapper_module != nullptr);
    wrapper_module->addTensorFunction(func, inputs, outputs, args);
}

void Tracer::trace(Module *model, vector<Tensor> inputs) {
    inputs[0].setTtype(TensorType::NORMAL_TENSOR);
    model->activation_tensors[inputs[0].name()] = std::shared_ptr<Tensor>(&inputs[0], [](Tensor *) {});
    model->activation_tensors[inputs[0].name()]->setName(inputs[0].name());
    model->activation_tensors[inputs[0].name()]->setModule(model);

    Module::llm_model_ptr = model;

    Tensor::tensor_status = TENSOR_STATIC_INIT;
    model->Forward(inputs, {});
    Tensor::tensor_status = TENSOR_STATIC_TRACE;
    model->Forward(inputs, {});
}

void Tracer::refleshInputTensor(vector<shared_ptr<Tensor>> inputs) {
    auto cpuPtr = std::dynamic_pointer_cast<CPUModuleWrapper>(model_[0]);
    assert(cpuPtr != nullptr);
    cpuPtr->traces_[0]->opInputs = inputs;
}

} // namespace mllm
