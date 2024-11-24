//
// Created by Rongjie Yi on 24-2-26.
//

#ifndef CPUMATMULFUNC_HPP
#define CPUMATMULFUNC_HPP
#include "CPUBackend.hpp"
#include "Tensor.hpp"
#include "Types.hpp"
#include "../compute/Matmul.hpp"
#include <cassert>

namespace mllm {
class Tensor;

class CPUmmFunction : public TensorFunction {
    static void tranTensorChl(Tensor &input) {
        assert(input.ctype() == BSHD);
        auto b = input.batch();
        auto h = input.head();
        auto d = input.dimension();
        auto s = input.sequence();
        auto ori_seq_idx = input.chls()[SEQUENCE];
        auto ori_head_idx = input.chls()[HEAD];
        auto ori_dim_idx = input.chls()[DIMENSION];
        input.chls()[HEAD] = ori_seq_idx;
        input.chls()[DIMENSION] = ori_head_idx;
        input.chls()[SEQUENCE] = ori_dim_idx;
        input.changeCtype();
        input.reshape(b, h, s, d);
        input.transed() = true;
        input.undiffusion() = false;
        // if no TENSOR_STATIC_SHAPED
        if (input.masterTensor() != nullptr) {
            auto b = input.masterTensor()->batch();
            auto h = input.masterTensor()->head();
            auto d = input.masterTensor()->dimension();
            auto s = input.masterTensor()->sequence();
            input.masterTensor()->chls() = input.chls();
            input.masterTensor()->changeCtype();
            input.masterTensor()->reshape(b, h, s, d);
            for (auto child : input.masterTensor()->childTensors()) {
                auto b = child->batch();
                auto h = child->head();
                auto d = child->dimension();
                auto s = child->sequence();
                child->chls() = input.chls();
                child->changeCtype();
                child->reshape(b, h, s, d);
            }
        } else {
            for (auto child : input.childTensors()) {
                auto b = child->batch();
                auto h = child->head();
                auto d = child->dimension();
                auto s = child->sequence();
                child->chls() = input.chls();
                child->changeCtype();
                child->reshape(b, h, s, d);
            }
        }
    }

public:
    void setup(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        if (inputs[1]->chls()[SEQUENCE] != 3) {
            tranTensorChl(*inputs[1]);
        }
        assert(inputs[0]->dimension() == inputs[1]->sequence());
        outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[1]->dimension());
        outputs[0]->setDtype(inputs[0]->dtype());
        outputs[0]->alloc();
    }
    void execute(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        bool isSame = std::equal(inputs[0]->chls().begin(), inputs[0]->chls().end(), inputs[1]->chls().begin());
        assert(inputs[0]->dtype() == MLLM_TYPE_F32);
        mat_mul(inputs[0], inputs[1], outputs[0], false, nullptr, false, isSame, CPUBackend::cpu_threads);
    }
};
} // namespace mllm
#endif // CPUMATMULFUNC_HPP