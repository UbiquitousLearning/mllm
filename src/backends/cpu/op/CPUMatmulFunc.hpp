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
#include <vector>
#include <memory>
#include <algorithm> // For std::equal

namespace mllm {
class Tensor;

class CPUmmFunction : public Op {
private:
    int thread_count = 4;

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

        // [FIX] Correctly handle the master tensor and its children
        if (auto master = input.masterTensor()) { // master is now a shared_ptr
            master->chls() = input.chls();
            master->changeCtype();
            master->reshape(master->batch(), master->head(), master->sequence(), master->dimension());

            // Loop through the master's children
            for (auto &child_wp : master->childTensors()) {
                // Lock the weak_ptr to get a shared_ptr
                if (auto child_sp = child_wp.lock()) {
                    // Now, use the shared_ptr to access members
                    auto b_c = child_sp->batch();
                    auto h_c = child_sp->head();
                    auto d_c = child_sp->dimension();
                    auto s_c = child_sp->sequence();
                    child_sp->chls() = input.chls();
                    child_sp->changeCtype();
                    child_sp->reshape(b_c, h_c, s_c, d_c);
                }
            }
        } else {
            // [FIX] Correctly handle this tensor's own children
            for (auto &child_wp : input.childTensors()) {
                // Lock the weak_ptr to get a shared_ptr
                if (auto child_sp = child_wp.lock()) {
                    // Now, use the shared_ptr to access members
                    auto b_c = child_sp->batch();
                    auto h_c = child_sp->head();
                    auto d_c = child_sp->dimension();
                    auto s_c = child_sp->sequence();
                    child_sp->chls() = input.chls();
                    child_sp->changeCtype();
                    child_sp->reshape(b_c, h_c, s_c, d_c);
                }
            }
        }
    }

public:
    CPUmmFunction(Backend *bn, string name, int threadCount) :
        Op(bn, name), thread_count(threadCount) {
    }

    ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        if (inputs[1]->chls()[SEQUENCE] != 3) {
            tranTensorChl(*inputs[1]);
        }
        if (!inputs[1]->shape().empty() && !inputs[0]->shape().empty()) {
            assert(inputs[0]->dimension() == inputs[1]->sequence());
        }
        outputs[0]->alloc();
        return MLLM_NO_ERROR;
    }

    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        if (inputs[1]->chls()[SEQUENCE] != 3) {
            tranTensorChl(*inputs[1]);
            assert(inputs[1]->chls()[SEQUENCE] == 3);
        }
        assert(inputs[0]->dimension() == inputs[1]->sequence());
        outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[1]->dimension());
        outputs[0]->setDtype(inputs[0]->dtype());
        // 遵从原始 reshape 逻辑，在这里 alloc
        // outputs[0]->alloc();
        return MLLM_NO_ERROR;
    }

    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        bool isSame = std::equal(inputs[0]->chls().begin(), inputs[0]->chls().end(), inputs[1]->chls().begin());
        assert(inputs[0]->dtype() == MLLM_TYPE_F32);
        mat_mul(inputs[0].get(), inputs[1].get(), outputs[0].get(), false, nullptr, false, isSame, thread_count);
        return MLLM_NO_ERROR;
    }
};

class CPUmmFunctionCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const override {
        return new CPUmmFunction(bn, name, threadCount);
    }
};

} // namespace mllm
#endif // CPUMATMULFUNC_HPP