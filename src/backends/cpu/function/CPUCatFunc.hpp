//
// Created by Rongjie Yi on 24-2-26.
//

#ifndef CPUCATFUNC_HPP
#define CPUCATFUNC_HPP
#include "Tensor.hpp"
#include "Types.hpp"
#include "CPUBackend.hpp"
#include <cassert>

namespace mllm {
class Tensor;

class CPUcatFunction : public TensorFunction {
public:
    void setup(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        Chl axis = (Chl)args[0];
        int expd_batch_ = inputs[0]->batch();
        for (int ii = 0; ii < inputs.size(); ++ii) {
            auto input = inputs[ii];
            if (input->batch() > expd_batch_) {
                expd_batch_ = input->batch();
            }
        }
        int dim_b = expd_batch_;
        int dim_h = inputs[0]->head();
        int dim_s = inputs[0]->sequence();
        int dim_d = inputs[0]->dimension();
        int sizes[] = {0, 0, 0, 0};
        Chl axes[] = {BATCH, HEAD, SEQUENCE, DIMENSION};
        int *dims[] = {&dim_b, &dim_h, &dim_s, &dim_d};
        for (int i = 0; i < 4; i++) {
            if (axis == axes[i]) {
                for (auto input : inputs) {
                    sizes[i] += (i == 0) ? input->batch() : (i == 1) ? input->head() :
                                                        (i == 2)     ? input->sequence() :
                                                                       input->dimension();
                }
                *dims[i] = sizes[i];
                break;
            }
        }
        outputs[0]->reshape(dim_b, dim_h, dim_s, dim_d);
        outputs[0]->setDtype(inputs[0]->dtype());
        outputs[0]->alloc();
        if (axis == HEAD){
            int cbatch = 0;
            int chead = 0;
            int cseq = 0;
            int cdim = 0;
            if(inputs[0]->hostPtr<float>() == inputs[1]->hostPtr<float>()){
                if (inputs[0]->masterTensor() == nullptr) {
                    inputs[0]->free();
                }
                inputs[0]->shallowCopyFrom(outputs[0], false, {cbatch, chead, cseq, cdim});
            }else{
                for (int idx = 0; idx < inputs.size(); idx++) {
                    if (inputs[idx]->masterTensor() == nullptr) {
                        inputs[idx]->free();
                    }
                    if (idx > 0) {
                        chead += inputs[idx - 1]->head();
                    }
                    inputs[idx]->shallowCopyFrom(outputs[0], false, {cbatch, chead, cseq, cdim}); // b,h,s,d
                }
            }
        }else if (axis == SEQUENCE && inputs[0]->head() != 1) {
            int cbatch = 0;
            int chead = 0;
            int cseq = 0;
            int cdim = 0;
            for (int idx = 0; idx < inputs.size(); idx++) {
                if (inputs[idx]->masterTensor() == nullptr) {
                    inputs[idx]->free();
                }
                if (idx > 0) {
                    cseq += inputs[idx - 1]->sequence();
                }
                inputs[idx]->shallowCopyFrom(outputs[0], false, {cbatch, chead, cseq, cdim}); // b,h,s,d
            }
        } else if (axis == DIMENSION && inputs[0]->head() != 1) {
            int cbatch = 0;
            int chead = 0;
            int cseq = 0;
            int cdim = 0;
            for (int idx = 0; idx < inputs.size(); idx++) {
                if (inputs[idx]->masterTensor() == nullptr) {
                    inputs[idx]->free();
                }
                if (idx > 0) {
                    cdim += inputs[idx - 1]->dimension();
                }
                int tmp_agg_idx;
                if (inputs[idx]->deaggregatedTensor() != nullptr) {
                    for (int t = 0; t < inputs[idx]->deaggregatedTensor()->aggregatedTensors().size(); t++) {
                        if (inputs[idx]->deaggregatedTensor()->aggregatedTensors()[t].get() == inputs[idx]) {
                            tmp_agg_idx = t;
                            continue;
                        }
                    }
                }
                inputs[idx]->shallowCopyFrom(outputs[0], false, {cbatch, chead, cseq, cdim}); // b,h,s,d
                if (inputs[idx]->deaggregatedTensor() != nullptr) {
                    vector<shared_ptr<Tensor>> shared_outputs = {};
                    for (int t = 0; t < inputs[idx]->deaggregatedTensor()->aggregatedTensors().size(); t++) {
                        if (t == tmp_agg_idx) {
                            inputs[idx]->deaggregatedTensor()->aggregatedTensors()[t] =
                                std::shared_ptr<Tensor>(inputs[idx], [](Tensor *) {});
                        }
                    }
                }
            }
        }
    }
    void execute(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        Chl axis = (Chl)args[0];
        int expd_batch_ = inputs[0]->batch();
        int expd_batch_input_idx = 0;
        for (int ii = 0; ii < inputs.size(); ++ii) {
            auto input = inputs[ii];
            if (input->batch() > expd_batch_) {
                expd_batch_ = input->batch();
                expd_batch_input_idx = ii;
            }
        }
        if (axis == BATCH) {
            for (int n = 0; n < inputs.size(); ++n) {
                auto copysize = inputs[0]->batch() * inputs[0]->head() * inputs[0]->sequence() * inputs[0]->dimension();
                memcpy(outputs[0]->ptrAt<float>(n * inputs[0]->batch(), 0, 0, 0),
                       inputs[n]->ptrAt<float>(0, 0, 0, 0),
                       sizeof(float) * copysize);
            }
        } else if (axis == DIMENSION) {
            for (int n = 0; n < expd_batch_; ++n) {
                for (int c = 0; c < inputs[0]->head(); ++c) {
                    for (int h = 0; h < inputs[0]->sequence(); ++h) {
                        int w = 0;
                        for (int idx = 0; idx < inputs.size(); idx++) {
                            int dim_size = inputs[idx]->dimension();
                            auto n_ = n;
                            if (idx != expd_batch_input_idx) {
                                n_ = 0;
                            }
                            assert(inputs[0]->dtype()==outputs[0]->dtype());
                            if(inputs[0]->dtype() == MLLM_TYPE_F32){
                                memcpy(outputs[0]->ptrAt<float>(n, c, h, w),
                                   inputs[idx]->ptrAt<float>(n_, c, h, 0),
                                   sizeof(float) * (dim_size));
                            } else if(inputs[0]->dtype() == MLLM_TYPE_F16) {
                                memcpy(outputs[0]->ptrAt<mllm_fp16_t>(n, c, h, w),
                                   inputs[idx]->ptrAt<mllm_fp16_t>(n_, c, h, 0),
                                   sizeof(mllm_fp16_t) * (dim_size));
                            }
                            w += dim_size;
                        }
                    }
                }
            }
        } else if ((axis == SEQUENCE) && inputs[0]->head() != 1) {
        } else if ((axis == SEQUENCE) && inputs[0]->head() == 1) {
            for (int n = 0; n < expd_batch_; ++n) {
                int h = 0;
                for (int idx = 0; idx < inputs.size(); idx++) {
                    auto n_ = n;
                    if (idx != expd_batch_input_idx) {
                        n_ = 0;
                    }
                    memcpy(outputs[0]->ptrAt<float>(n, 0, h, 0),
                           inputs[idx]->ptrAt<float>(n_, 0, 0, 0),
                           sizeof(float) * (inputs[idx]->sequence() * inputs[idx]->dimension()));
                    h += inputs[idx]->sequence();
                }
            }
        } else if (axis == HEAD) {
            if(inputs[0]->hostPtr<float>() == inputs[1]->hostPtr<float>()){
                for (int b = 0; b < outputs[0]->batch(); ++b) {
                    for (int s = 0; s < inputs[0]->sequence(); ++s) {
                        for (int h_ = 1; h_ < outputs[0]->head(); ++h_) {
                            int dim_size = inputs[0]->dimension();
                            memcpy(outputs[0]->ptrAt<float>(b, h_, s, 0),
                                    outputs[0]->ptrAt<float>(b, 0, s, 0),
                                    sizeof(float) * (dim_size));
                        }
                    }
                }
                return;
            }
            for (int b = 0; b < expd_batch_; ++b) {
#pragma omp parallel for collapse(1) num_threads(CPUBackend::cpu_threads)
                for (int s = 0; s < inputs[0]->sequence(); ++s) {
                    // for (int h = 0; h < inputs[0]->head(); ++h) {
                    int head_i = 0;
                    for (int idx = 0; idx < inputs.size(); idx++) {
                        auto b_ = b;
                        if (idx != expd_batch_input_idx) {
                            b_ = 0;
                        }
                        int dim_size = inputs[idx]->dimension() * inputs[idx]->head();
                        memcpy(outputs[0]->ptrAt<float>(b, head_i, s, 0),
                               inputs[idx]->ptrAt<float>(b_, 0, s, 0),
                               sizeof(float) * (dim_size));
                        head_i += inputs[idx]->head();
                    }
                    // }
                }
            }
        }
    }
};

} // namespace mllm
#endif // CPUCATFUNC_HPP