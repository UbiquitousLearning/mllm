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
    normSize_ = inputs[0]->dimension();
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->shape(1), inputs[0]->shape(2), inputs[0]->shape(3));
    //outputs[0]->setDtype(activationDtype());
    //std::cout << name() << "  CPURMSNorm  reshape" << std::endl;
    return Op::reshape(inputs, outputs);
}

ErrorCode CPURMSNorm::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto input = inputs[0];
    int batch = input->batch();
    int dim = input->dimension();
    int seq = input->sequence();
    int head = input->head();
    for (int h = 0; h < head; h++) {
        for (int n = 0; n < batch; n++) {
            #pragma omp parallel for num_threads(8)
            for (int s = 0; s < seq; s++) {
                float sum_squares = 0.0F;
                // sum
                for (int d = 0; d < dim; d++) {
                    float value = input->dataAt<float>(n, h, s, d);
                    sum_squares += value * value;
                }
                float rms = std::sqrt(sum_squares / dim + epsilon_);
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

    //std::cout << name() << "  CPURMSNorm()" << std::endl;
    return Op::execute(inputs, outputs);
}
ErrorCode CPURMSNorm::load(AbstructLoader &loader) {
    weight_.setName(name() + ".weight");
    weight_.reshape(1, 1, 1, normSize_); //
    weight_.setDtype(loader.getDataType(weight_.name()));
    weight_.alloc();
    // TEST
    //    weight_.fullData<float>(2.0);
    //    inputs[0]->fullDataTest();
    loader.load(&weight_);
    return Op::load(loader);
}
ErrorCode CPURMSNorm::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    weight_.free();
    return Op::free(inputs, outputs);
}
} // namespace mllm