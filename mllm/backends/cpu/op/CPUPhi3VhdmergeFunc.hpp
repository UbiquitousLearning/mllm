//
// Created by Rongjie Yi on 24-2-26.
//

#ifndef CPUPHI3VHDMERGEEFUNC_HPP
#define CPUPHI3VHDMERGEEFUNC_HPP

#include "Tensor.hpp"
#include "Types.hpp"
#include "CPUBackend.hpp"
#include <memory>
#include <cmath> // For std::sqrt

namespace mllm {
class Tensor;

class CPUPhi3VhdmergeFunction : public Op {
private:
    int thread_count = 4;
    int h_crop_;
    int w_crop_;

public:
    CPUPhi3VhdmergeFunction(Backend *bn, string name, int threadCount, int h_crop, int w_crop)
        : Op(bn, name), thread_count(threadCount), h_crop_(h_crop), w_crop_(w_crop) {}

    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        int N = inputs[0]->batch();
        int L = inputs[0]->sequence();
        int C = inputs[0]->dimension();
        assert(L == 24 * 24);
        assert(C == 1024);
        assert(N % (h_crop_ * w_crop_) == 0);
        
        int num_images = N / (h_crop_ * w_crop_);
        int H = static_cast<int>(std::sqrt(L));

        int b = num_images;
        int s = h_crop_ * H / 2;
        int h = w_crop_ * H / 2;
        int d = 4 * C;

        outputs[0]->reshape(b, h, s, d);
        outputs[0]->setDtype(inputs[0]->dtype());
        // 遵从原始 reshape 逻辑，在这里 alloc
        // outputs[0]->alloc();
        return MLLM_NO_ERROR;
    }

    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        int N = inputs[0]->batch();
        int L = inputs[0]->sequence();
        int C = inputs[0]->dimension();
        int num_images = N / (h_crop_ * w_crop_);
        int H = static_cast<int>(std::sqrt(L));

        int b = num_images;
        int s = h_crop_ * H / 2;
        int h = w_crop_ * H / 2;
        int d = 4 * C;

        #pragma omp parallel for collapse(3) num_threads(thread_count)
        for (int ob = 0; ob < b; ob++) {
            for (int os = 0; os < s; os++) {
                for (int oh = 0; oh < h; oh++) {
                    int base_s = ((oh / 12) * (24 * 24)) + (os * 48) + (2 * (oh % 12));
                    int hed = base_s % L;
                    int btch = (ob * (h_crop_ * w_crop_)) + (base_s / L);
                    
                    auto i_ptr_0 = inputs[0]->ptrAt<char>(btch, 0, hed, 0);
                    auto i_ptr_1 = inputs[0]->ptrAt<char>(btch, 0, hed + 1, 0);
                    auto i_ptr_2 = inputs[0]->ptrAt<char>(btch, 0, hed + 24, 0);
                    auto i_ptr_3 = inputs[0]->ptrAt<char>(btch, 0, hed + 25, 0);
                    
                    size_t copy_size = (size_t)C * inputs[0]->dtypeSize();

                    memcpy(outputs[0]->ptrAt<char>(ob, oh, os, 0),
                           i_ptr_0, copy_size);
                    memcpy(outputs[0]->ptrAt<char>(ob, oh, os, C),
                           i_ptr_1, copy_size);
                    memcpy(outputs[0]->ptrAt<char>(ob, oh, os, C * 2),
                           i_ptr_2, copy_size);
                    memcpy(outputs[0]->ptrAt<char>(ob, oh, os, C * 3),
                           i_ptr_3, copy_size);
                }
            }
        }
        return MLLM_NO_ERROR;
    }
};

class CPUPhi3VhdmergeFunctionCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const override {
        int h_crop = static_cast<int>(op_param.at("h_crop"));
        int w_crop = static_cast<int>(op_param.at("w_crop"));
        return new CPUPhi3VhdmergeFunction(bn, name, threadCount, h_crop, w_crop);
    }
};

} // namespace mllm
#endif // CPUPHI3VHDMERGEEFUNC_HPP