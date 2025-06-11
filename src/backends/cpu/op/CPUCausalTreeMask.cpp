
#include "CPUCausalTreeMask.hpp"
#include <cmath>

namespace mllm {

CPUCausalTreeMask::CPUCausalTreeMask(Backend *bn, string opName, int threadCount) :
    thread_count(threadCount),
    Op(bn, opName) {
}

ErrorCode CPUCausalTreeMask::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // std::cout << "CPUMask  reshape" << std::endl;
    //  assert(inputs.size() == 1);
    assert(outputs.size() == 1);
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    return Op::reshape(inputs, outputs);
}

/*
* inputs: hidden_values, tree_ancestor
* e.g. hidden_values [6 x 6], tree_ancestor [-1, 0, 1, 1, 3], draft_size = 5
* Target Mask (usual):
* TFFFFF
* TTFFFF
* TTTFFF
* TTTTFF
* TTTTTF
* TTTTTT

* Target Mask (tree):
* cR0113
* TFFFFF
* TTFFFF
* TTTFFF
* TTTTFF
* TTTFTF
* TTTFTT
* 第s行(s>=c)和第tree_ancestor_d，即t[s-c]+c行的mask一样，并在[s, s]位置为T
*/
ErrorCode CPUCausalTreeMask::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    if (inputs[0]->sequence() > 1) {
        int batch_size = inputs[0]->batch();
        int head_num = inputs[0]->head();
        int sequence = inputs[0]->sequence();
        int dimension = inputs[0]->dimension();
        int old_dim = 0;

        auto &tree_ancestor = inputs[2];
        int draft_size = tree_ancestor->sequence() - 1;
        int context_size = sequence - draft_size;
        if (inputs.size() > 1) {
            old_dim = (int)inputs[1]->dataAt<float>(0, 0, 0, 0) - sequence;
        } else {
#ifndef LLAMAFILE_SGEMM
            old_dim = dimension - sequence;
#endif
        }
#pragma omp parallel for collapse(3) num_threads(thread_count)
        for (int n = 0; n < batch_size; ++n) {
            for (int h = 0; h < head_num; ++h) {
                for (int s = 0; s < sequence; ++s) {
                    // 非draft的部分（之前的上下文）按正常的mask计算
                    if (s < context_size) {
                        for (int d = 0; d < inputs[0]->dimension(); ++d) {
                            if (d > s + old_dim) {
                                outputs[0]->setDataAt<float>({n, h, s, d}, -INFINITY);
                            } else {
                                outputs[0]->setDataAt<float>({n, h, s, d}, inputs[0]->dataAt<float>(n, h, s, d));
                            }
                        }
                    }
                    // draft的部分（树的部分）按树的mask计算
                    else {
                        int tree_ancestor_s = tree_ancestor->dataAt<int32_t>({0, 0, s - context_size + 1, 0}); // + context_size;
                        for (int d = 0; d < inputs[0]->dimension(); ++d) {
                            if (d == s) {
                                outputs[0]->setDataAt<float>({n, h, s, d}, inputs[0]->dataAt<float>(n, h, s, d));
                            } else {
                                if (outputs[0]->dataAt<float>({n, h, tree_ancestor_s, d}) == -INFINITY) {
                                    outputs[0]->setDataAt<float>({n, h, s, d}, -INFINITY);
                                } else {
                                    outputs[0]->setDataAt<float>({n, h, s, d}, inputs[0]->dataAt<float>(n, h, s, d));
                                }
                            }
                        }
                    }
                }
            }
        }
    } else {
        outputs[0]->copyFrom(inputs[0]);
    }
    return Op::execute(inputs, outputs);
}

// ErrorCode CPUCausalTreeMask::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
//     // assert(inputs.size() == 1);
//     assert(outputs.size() == 1);
//     if(inputs[0]->masterTensor() == nullptr) {
//         inputs[0]->free(); // TODO remove
//     }
//     outputs[0]->setDtype(activation_dtype());
//     outputs[0]->alloc();
//     inputs[0]->shallowCopyFrom(outputs[0].get(), false);
//     return MLLM_NO_ERROR;
// }
} // namespace mllm
