//
// Created by Rongjie Yi on 24-2-26.
//

#ifndef CPUPHI3VHDMERGEEFUNC_HPP
#define CPUPHI3VHDMERGEEFUNC_HPP
#include "Tensor.hpp"
#include "Types.hpp"
#include "CPUBackend.hpp"

namespace mllm {
class Tensor;

class CPUPhi3VhdmergeFunction : public TensorFunction {
public:
    void reshape(vector<shared_ptr<Tensor>> outputs, vector<shared_ptr<Tensor>> inputs, vector<float> args) override {
        assert(args.size() == 2);
        int h_crop = (int)args[0];
        int w_crop = (int)args[1];
        int N = inputs[0]->batch();
        int L = inputs[0]->sequence();
        int C = inputs[0]->dimension();
        assert(L == 24 * 24);
        assert(C == 1024);
        assert(N % (h_crop * w_crop) == 0);
        int num_images = N / (h_crop * w_crop);
        int H = static_cast<int>(std::sqrt(L));

        int b = num_images;
        int s = h_crop * H / 2;
        int h = w_crop * H / 2;
        int d = 4 * C;

        outputs[0]->reshape(b, h, s, d);
        outputs[0]->setDtype(inputs[0]->dtype());
        outputs[0]->alloc();
    }
    void execute(vector<shared_ptr<Tensor>> outputs, vector<shared_ptr<Tensor>> inputs, vector<float> args) override {
        int h_crop = (int)args[0];
        int w_crop = (int)args[1];
        int N = inputs[0]->batch();
        int L = inputs[0]->sequence();
        int C = inputs[0]->dimension();
        int num_images = N / (h_crop * w_crop);
        int H = static_cast<int>(std::sqrt(L));

        int b = num_images;
        int s = h_crop * H / 2;
        int h = w_crop * H / 2;
        int d = 4 * C;

#pragma omp parallel for collapse(3) num_threads(CPUBackend::cpu_threads)
        for (int ob = 0; ob < b; ob++) {
            for (int os = 0; os < s; os++) {
                for (int oh = 0; oh < h; oh++) {
                    int base_s = int(oh / 12) * (24 * 24) + os * 48 + 2 * (oh % 12);
                    int hed = base_s % L;
                    int btch = int(base_s / L);
                    auto i_ptr_0 = inputs[0]->ptrAt<float>(btch, hed, 0, 0);
                    auto i_ptr_1 = inputs[0]->ptrAt<float>(btch, hed + 1, 0, 0);
                    auto i_ptr_2 = inputs[0]->ptrAt<float>(btch, hed + 24, 0, 0);
                    auto i_ptr_3 = inputs[0]->ptrAt<float>(btch, hed + 25, 0, 0);
                    memcpy(outputs[0]->ptrAt<float>(ob, oh, os, 0),
                           i_ptr_0, C * sizeof(float));
                    memcpy(outputs[0]->ptrAt<float>(ob, oh, os, C),
                           i_ptr_1, C * sizeof(float));
                    memcpy(outputs[0]->ptrAt<float>(ob, oh, os, C * 2),
                           i_ptr_2, C * sizeof(float));
                    memcpy(outputs[0]->ptrAt<float>(ob, oh, os, C * 3),
                           i_ptr_3, C * sizeof(float));
                }
            }
        }
    }
};

} // namespace mllm
#endif // CPUPHI3VHDMERGEEFUNC_HPP