
#include "CPUCausalMask.hpp"
#include <cmath>

namespace mllm {

// template class CPUMask;
// template class CPUMask;

CPUCausalMask::CPUCausalMask(Backend *bn, string opName, bool multiThread) :
    Op(bn, opName) {
}

ErrorCode CPUCausalMask::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    //std::cout << "CPUMask  reshape" << std::endl;
    CHECK_EQ(inputs.size(), 1);
    CHECK_EQ(outputs.size(), 1);
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    return Op::reshape(inputs, outputs);
}

ErrorCode CPUCausalMask::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    if(inputs[0]->sequence() >1 ) {
        int batch_size = inputs[0]->batch();
        int head_num = inputs[0]->head();
        int sequence = inputs[0]->sequence();
        int dimension = inputs[0]->dimension();
        for (int n = 0; n < batch_size; ++n) {
            for (int h = 0; h < head_num; ++h) {
                for (int s = 0; s < sequence; ++s) {
                    #pragma omp parallel for num_threads(4)
                    for (int d = 0; d < dimension; ++d) {
                        if (d > s) {
                            outputs[0]->setDataAt<float>({n, h, s, d}, -INFINITY);
                        }
                        else{
                            outputs[0]->setDataAt<float>({n, h, s, d}, inputs[0]->dataAt<float>(n, h, s, d));
                        }
                    }
                }
            }
        }
    }
    else{
        outputs[0]->copyFrom(inputs[0]);
    }
    return Op::execute(inputs, outputs);
}

} // namespace mllm
