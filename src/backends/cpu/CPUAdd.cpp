
#include "CPUAdd.hpp"

namespace mllm {

// template class CPUAdd;
// template class CPUAdd;

CPUAdd::CPUAdd(Backend *bn,  string opName, int threadCount) : thread_count(threadCount),
    Op(bn, opName) {
}

ErrorCode CPUAdd::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    //std::cout<<name() << "  CPUAdd  reshape" << std::endl;
    CHECK_EQ(inputs.size(), 2);
    CHECK_EQ(outputs.size(), 1);
    if(inputs[0]->batch() == 1 || inputs[1]->batch() == 1){
    }else {
        CHECK_EQ(inputs[0]->batch(), inputs[1]->batch());
    }
    CHECK_EQ(inputs[0]->head(), inputs[1]->head());
    CHECK_EQ(inputs[0]->sequence(), inputs[1]->sequence());
    CHECK_EQ(inputs[0]->dimension(), inputs[1]->dimension());
    outputs[0]->reshape(std::max(inputs[0]->batch(), inputs[1]->batch()), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    //outputs[0]->setDtype(activationDtype());
    return Op::reshape(inputs, outputs);
}


ErrorCode CPUAdd::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    //std::cout<<name() << "  CPUAdd()" << std::endl;
    int N = std::max(inputs[0]->batch(), inputs[1]->batch());
    int C = inputs[0]->head();
    int H = inputs[0]->sequence();
    int W = inputs[0]->dimension();
    for (int n = 0; n < N; ++n) {
        auto n_0 = std::min(n, inputs[0]->batch() - 1);
        auto n_1 = std::min(n, inputs[1]->batch() - 1);
        if(inputs[0]->masterTensor() == nullptr && inputs[1]->masterTensor() == nullptr && inputs[0]->ctype() == inputs[1]->ctype()) {
            auto copy_size = C * H * W;
            auto in0_ptr = inputs[0]->ptrAt<float>(n_0, 0, 0, 0);
            auto in1_ptr = inputs[1]->ptrAt<float>(n_1, 0, 0, 0);
            auto out_ptr = outputs[0]->ptrAt<float>(n, 0, 0, 0);
#pragma omp parallel for num_threads(thread_count)
            for (int is = 0; is < copy_size; ++is) {
                out_ptr[is] = in0_ptr[is] + in1_ptr[is];
            }
        }else {
            for (int c = 0; c < C; ++c) {
                for (int h = 0; h < H; ++h) {
#pragma omp parallel for num_threads(thread_count)
                    for (int w = 0; w < W; ++w) {
                        outputs[0]->setDataAt<float>(n, c, h, w, inputs[0]->dataAt<float>(n_0, c, h, w) + inputs[1]->dataAt<float>(n_1, c, h, w));
                    }
                }
            }
        }
    }
    return Op::execute(inputs, outputs);
}
} // namespace mllm
