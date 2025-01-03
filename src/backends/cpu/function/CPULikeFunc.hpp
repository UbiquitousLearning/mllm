//
// Created by Rongjie Yi on 24-12-16.
//

#ifndef CPULIKEFUNC_HPP
#define CPULIKEFUNC_HPP
#include "Tensor.hpp"
#include "Types.hpp"
namespace mllm {
class Tensor;

class CPUlikeFunction : public TensorFunction {
public:
    void setup(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        float like_value = args[0];
        outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
        outputs[0]->setDtype(inputs[0]->dtype()); // like_values
        outputs[0]->alloc();
        memset(outputs[0]->hostPtr<float>(), like_value, outputs[0]->count() * sizeof(float));
    }
    void execute(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        float like_value = args[0];
        memset(outputs[0]->hostPtr<float>(), like_value, outputs[0]->count() * sizeof(float));
    }
};
} // namespace mllm
#endif // CPULIKEFUNC_HPP