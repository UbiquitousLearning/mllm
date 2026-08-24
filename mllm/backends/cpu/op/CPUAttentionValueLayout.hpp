#ifndef CPU_ATTENTION_VALUE_LAYOUT_HPP
#define CPU_ATTENTION_VALUE_LAYOUT_HPP

#include "Tensor.hpp"

#include <cassert>

namespace mllm {

// Reinterpret a physically sequence-contiguous NPU value-cache tensor as BHDS
// and propagate the metadata transition to its cache master and sibling views.
// CPU dense AV matmul and fused sparse AV must use the same transition.
inline void transposeAttentionValueChannels(Tensor &input) {
    assert(input.ctype() == BSHD);
    const auto b = input.batch();
    const auto h = input.head();
    const auto d = input.dimension();
    const auto s = input.sequence();
    const auto ori_seq_idx = input.chls()[SEQUENCE];
    const auto ori_head_idx = input.chls()[HEAD];
    const auto ori_dim_idx = input.chls()[DIMENSION];
    input.chls()[HEAD] = ori_seq_idx;
    input.chls()[DIMENSION] = ori_head_idx;
    input.chls()[SEQUENCE] = ori_dim_idx;
    input.changeCtype();
    input.reshape(b, h, s, d);
    input.transed() = true;
    input.undiffusion() = false;

    if (auto master = input.masterTensor()) {
        const auto master_batch = master->batch();
        const auto master_head = master->head();
        const auto master_dimension = master->dimension();
        const auto master_sequence = master->sequence();
        master->chls() = input.chls();
        master->changeCtype();
        master->reshape(master_batch, master_head, master_sequence,
                        master_dimension);

        for (auto &child_wp : master->childTensors()) {
            if (auto child = child_wp.lock()) {
                const auto child_batch = child->batch();
                const auto child_head = child->head();
                const auto child_dimension = child->dimension();
                const auto child_sequence = child->sequence();
                child->chls() = input.chls();
                child->changeCtype();
                child->reshape(child_batch, child_head, child_sequence,
                               child_dimension);
            }
        }
    } else {
        for (auto &child_wp : input.childTensors()) {
            if (auto child = child_wp.lock()) {
                const auto child_batch = child->batch();
                const auto child_head = child->head();
                const auto child_dimension = child->dimension();
                const auto child_sequence = child->sequence();
                child->chls() = input.chls();
                child->changeCtype();
                child->reshape(child_batch, child_head, child_sequence,
                               child_dimension);
            }
        }
    }
}

} // namespace mllm

#endif // CPU_ATTENTION_VALUE_LAYOUT_HPP
