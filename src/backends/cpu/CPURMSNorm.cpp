#include <cmath>
#include "CPURMSNorm.hpp"
#include "Tensor.hpp"

namespace mllm {

// template class CPURMSNorm;
// template class CPURMSNorm;

CPURMSNorm::CPURMSNorm(Backend *bn, string opName, bool multiThread, float epsilon) :
    Op(bn, opName), epsilon_(epsilon), support_multi_thread_(multiThread) {
    weight_.setBackend(bn);
}

ErrorCode CPURMSNorm::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // RMSNorm 类似于LayerNorm作用于channel维度
    weight_.reshape(1, 1, 1, inputs[0]->dimension()); // (C, 1, 1, 1)
    weight_.setName(name() + ".weight");
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->shape(1), inputs[0]->shape(2), inputs[0]->shape(3));
    std::cout << name() << "  CPURMSNorm  reshape" << std::endl;
    return NO_ERROR;
}

ErrorCode CPURMSNorm::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    if (!inputs[0]->allocted()) {
        inputs[0]->alloc(); // TODO remove
    }
    outputs[0]->alloc();
    weight_.alloc();

    // TEST
    //    weight_.fullData<float>(2.0);
    //    inputs[0]->fullDataTest();

    std::cout << name() << "  CPURMSNorm  setUp" << std::endl;
    return NO_ERROR;
}

ErrorCode CPURMSNorm::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto input = inputs[0];
    int batch = input->batch();
    int dim = input->dimension();
    int seq = input->sequence();
    int head = input->head();
    for (int h = 0; h < head; h++) {
        for (int s = 0; s < seq; s++) {
            for (int n = 0; n < batch; n++) {
                float sum_squares = 0.0F;
                // sum
                for (int d = 0; d < dim; d++) {
                    float value = input->dataAt<float>(n, h, s, d);
                    sum_squares += value * value;
                }
                float rms = std::sqrt(sum_squares / dim); //+ epsilon_);
                // use memset to set the value of the memory block
                for (int d = 0; d < dim; d++) {
                    float value = input->dataAt<float>(n, h, s, d);
                    outputs[0]->setDataAt<float>(n, h, s, d, weight_.dataAt<float>(0, 0, 0, d) * value / rms);
                }
            }
        }
    }
    //    input->printData<float>();
    //    weight_.printData<float>();
    //    outputs[0]->printData<float>();

    std::cout << name() << "  CPURMSNorm()" << std::endl;
    return NO_ERROR;
}
ErrorCode CPURMSNorm::load(ParamLoader &loader) {
    return Op::load(loader);
}
} // namespace mllm