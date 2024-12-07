#ifndef TRACE_HPP
#define TRACE_HPP

#include "Tensor.hpp"
#include "Types.hpp"
#include <memory>

namespace mllm {

class Tracer {
public:
    static vector<std::shared_ptr<Module>> model_;

    static bool isToCreateNewModule;

    static void addModule(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs, string name);

    static void addOp(Op *op, vector<std::shared_ptr<Tensor>> inputs, vector<std::shared_ptr<Tensor>> outputs);

    static void addTensorFunction(TensorFunction *func,
                                  vector<Tensor *> inputs, vector<Tensor *> outputs, vector<float> args);

    static void trace(Module *model, vector<Tensor> inputs);

    static void refleshInputTensor(vector<shared_ptr<Tensor>> inputs);

};

} // namespace mllm

#endif // TRACE_HPP