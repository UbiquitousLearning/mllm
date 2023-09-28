
#include "CPURMSNorm.hpp"
#include "Tensor.hpp"

namespace mllm {

// template class CPURMSNorm;
// template class CPURMSNorm;

CPURMSNorm::CPURMSNorm(Backend *bn, bool multiThread, float epsilon) :
    Op(bn), epsilon_(epsilon), support_multi_thread_(multiThread) {
    weight_.setBackend(bn);
}

ErrorCode CPURMSNorm::reshape(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
    // TODO: Shape of Weights
    weight_.reshape(inputs[0]->shape(-1), 1, 1, 1); // (C, 1, 1, 1)
    weight_.setName(name() + ".weight");
    outputs[0]->reshape(inputs[0]->num(), inputs[0]->channels(), inputs[0]->height(), inputs[0]->width());
    std::cout << "CPURMSNorm  reshape" << std::endl;
    return NO_ERROR;
}

ErrorCode CPURMSNorm::setUp(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
    if (!inputs[0]->allocted()) {
        inputs[0]->alloc();
    }
    outputs[0]->alloc();
    weight_.alloc();

    std::cout << "CPURMSNorm  setUp" << std::endl;
    return NO_ERROR;
}

ErrorCode CPURMSNorm::execute(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
    auto input = inputs[0];
    int batch = input->num();
    int n1 = input->legacyShape(1);
    int n2 = input->legacyShape(2);
    int n3 = input->legacyShape(3);
    for (int w = 0; w < n3; w++) {
        for (int h = 0; h < n2; h++) {
            for (int c = 0; c < n1; c++) {
                float sum_squares = 0.0F;

                for (int n = 0; n < batch; n++) {
                    float value = input->dataAt<float>(n, c, h, w);
                    sum_squares += value * value;
                }
                float rms = std::sqrt(sum_squares / batch + epsilon_);
                // use memset to set the value of the memory block
                for (int n = 0; n < batch; n++) {
                    float value = input->dataAt<float>(n, c, h, w);
                    outputs[0]->setDataAt<float>(n, c, h, w, value / rms);
                }
            }
        }
    }

    std::cout
        << "CPURMSNorm()" << std::endl;
    return NO_ERROR;
}

ErrorCode CPURMSNorm::load(ParamLoader &loader) {
    std::cout << "CPURMSNorm load" << std::endl;
    return NO_ERROR;
}
} // namespace mllm
