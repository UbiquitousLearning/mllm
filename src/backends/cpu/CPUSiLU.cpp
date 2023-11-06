
#include "CPUSiLU.hpp"
#include <cmath>

namespace mllm {

// template class CPUSiLU;
// template class CPUSiLU;

CPUSiLU::CPUSiLU(Backend *bn, string opName, bool multiThread) :
    Op(bn, opName) {
}

ErrorCode CPUSiLU::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    outputs[0]->reshape(inputs[0]->num(), inputs[0]->channels(), inputs[0]->height(), inputs[0]->width());
    //outputs[0]->setDtype(activationDtype());
    std::cout<<name() << "  CPUSiLU  reshape" << std::endl;
    return NO_ERROR;
}

ErrorCode CPUSiLU::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto input = inputs[0];
    int batch = input->num();
    int n1 = input->shape(1);
    int n2 = input->shape(2);
    int n3 = input->shape(3);
    for (int w = 0; w < n3; w++) {
        for (int h = 0; h < n2; h++) {
            for (int c = 0; c < n1; c++) {
                for (int n = 0; n < batch; n++) {
                    float value = input->dataAt<float>(n, c, h, w);
                    outputs[0]->setDataAt<float>(n, c, h, w, value / (1 + std::exp(-value)));
                }
            }
        }
    }
    std::cout<<name() << "  CPUSiLU()" << std::endl;
    return NO_ERROR;
}

ErrorCode CPUSiLU::load(ParamLoader &loader) {
    std::cout<<name() << "  CPUSiLU load" << std::endl;
    return NO_ERROR;
}
} // namespace mllm
