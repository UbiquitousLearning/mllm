#include "backends/xnnpack/Ops/XpRoPE.hpp"
#include "Types.hpp"
#include "xnnpack.h"
#include <array>

namespace mllm::xnnpack {

Tensor XpRoPE::sin_params_;
Tensor XpRoPE::cos_params_;
int XpRoPE::input_dims_previous_ = 0;

namespace {
void sinusoidal_position_embedding_huggingface(int seq_len, int output_dim, Tensor &sin, Tensor &cos, float base = 10000) {
#pragma omp parallel for num_threads(4)
    for (int s = 0; s < seq_len; ++s) {
        for (int d = 0; d < output_dim / 2; d += 1) {
            int i = (int)d / 1;
            float sin_value = sinf((float)s / (float)std::pow(base, 2.0 * i / output_dim));
            float cos_value = cosf((float)s / (float)std::pow(base, 2.0 * i / output_dim));
            sin.setDataAt<float>(1, s, 1, d, sin_value);
            cos.setDataAt<float>(1, s, 1, d, cos_value);
        }
        for (int d = output_dim / 2; d < output_dim; d += 1) {
            int i = (int)(d - output_dim / 2);
            float sin_value = sinf((float)s / (float)std::pow(base, 2.0 * i / output_dim));
            float cos_value = cosf((float)s / (float)std::pow(base, 2.0 * i / output_dim));
            sin.setDataAt<float>(1, s, 1, d, sin_value);
            cos.setDataAt<float>(1, s, 1, d, cos_value);
        }
    }
}
} // namespace

ErrorCode XpRoPE::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto xpb = (XnnpackBackend *)inputs[0]->backend();

    // the inputs should in [b, s, h, d] shape layout.
    auto b = inputs[0]->shape()[0];
    auto s = inputs[0]->shape()[1];
    auto h = inputs[0]->shape()[2];
    auto d = inputs[0]->shape()[3];

    // PENDING: making rope weight.
    if (sin_params_.rawHostPtr() == nullptr || cos_params_.rawHostPtr() == nullptr || input_dims_previous_ < inputs[0]->dimension()) {
        input_dims_previous_ = inputs[0]->dimension();

        sin_params_.reshape(1, max_position_embeddings_, 1, d);
        cos_params_.reshape(1, max_position_embeddings_, 1, d);
        sin_params_.alloc();
        cos_params_.alloc();
        defineWeightTensor(xpb, &sin_params_);
        defineWeightTensor(xpb, &cos_params_);

        sinusoidal_position_embedding_huggingface(max_position_embeddings_, d, sin_params_, cos_params_, rope_theta_);
    }
    return MLLM_NO_ERROR;
}

ErrorCode XpRoPE::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto xpb = (XnnpackBackend *)inputs[0]->backend();
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    return Op::reshape(inputs, outputs);
}

ErrorCode XpRoPE::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto xpb = (XnnpackBackend *)inputs[0]->backend();
    tryDefineAllXpTensors(xpb, inputs);
    tryDefineAllXpTensors(xpb, outputs);

    // ref: https://chenghuawang.github.io/keep-moving-forward/tech/fundamental_rope/#0x05-%E4%BB%A3%E7%A0%81%E5%AE%9E%E7%8E%B0
    // the inputs should in [b, s, h, d] shape layout.
    auto dtype = inputs[0]->dtype();
    size_t b = inputs[0]->shape()[0];
    size_t s = inputs[0]->shape()[1];
    size_t h = inputs[0]->shape()[2];
    size_t d = inputs[0]->shape()[3];

    // x1 = x[..., : x.shape[-1] // 2]
    // x2 = x[..., x.shape[-1] // 2 :]
    // torch.cat((-x2, x1), dim=-1)
    auto x1 = defineTemporaryTensor(xpb, {b, s, h, d / 2}, dtype);
    auto x2 = defineTemporaryTensor(xpb, {b, s, h, d / 2}, dtype);
    {
        std::array<size_t, 4> offsets{0, 0, 0, 0};
        std::array<size_t, 4> output_dim{b, s, h, d / 2};
        auto status = xnn_define_static_slice(xpb->getXnnSubgraph(), 4, offsets.data(), output_dim.data(), inputs[0]->uuid(), x1, 0);
        if (status != xnn_status_success) {
            Log::error("XpRoPE, xnn_define_static_slice for x1 failed");
            exit(-1);
        }
    }
    {
        std::array<size_t, 4> offsets{0, 0, 0, d / 2};
        std::array<size_t, 4> output_dim{b, s, h, d / 2};
        auto status = xnn_define_static_slice(xpb->getXnnSubgraph(), 4, offsets.data(), output_dim.data(), inputs[0]->uuid(), x2, 0);
        if (status != xnn_status_success) {
            Log::error("XpRoPE, xnn_define_static_slice for x2 failed");
            exit(-1);
        }
    }

    // torch.cat((-x2, x1), dim = -1)
    {
        auto status = xnn_define_negate(xpb->getXnnSubgraph(), x2, x2, 0);
        if (status != xnn_status_success) {
            Log::error("XpRoPE, xnn_define_negate failed");
            exit(-1);
        }
    }
    auto x_new = defineTemporaryTensor(xpb, {b, s, h, d}, dtype);
    {
        auto status = xnn_define_concatenate2(xpb->getXnnSubgraph(), 3, x2, x1, x_new, 0);
        if (status != xnn_status_success) {
            Log::error("XpRoPE, xnn_define_concatenate2 failed");
            exit(-1);
        }
    }

    // (x * cos) + (x_new * sin)
    auto x_cosined = defineTemporaryTensor(xpb, {b, s, h, d}, dtype);
    auto x_new_sined = defineTemporaryTensor(xpb, {b, s, h, d}, dtype);
    {
        auto status = xnn_define_binary(xpb->getXnnSubgraph(), xnn_binary_multiply, nullptr, inputs[0]->uuid(), cos_params_.uuid(), x_cosined, 0);
        if (status != xnn_status_success) {
            Log::error("XpRoPE, xnn_define_binary failed");
            exit(-1);
        }
    }
    {
        auto status = xnn_define_binary(xpb->getXnnSubgraph(), xnn_binary_multiply, nullptr, x_new, sin_params_.uuid(), x_new_sined, 0);
        if (status != xnn_status_success) {
            Log::error("XpRoPE, xnn_define_binary failed");
            exit(-1);
        }
    }
    {
        auto status = xnn_define_binary(xpb->getXnnSubgraph(), xnn_binary_add, nullptr, x_cosined, x_new_sined, outputs[0]->uuid(), 0);
        if (status != xnn_status_success) {
            Log::error("XpRoPE, xnn_define_binary failed");
            exit(-1);
        }
    }

    return MLLM_NO_ERROR;
}

ErrorCode XpRoPE::load(AbstructLoader &loader) {
    Log::warn("XpRoPE currently only supports the HF form of subspace partitioning method (halving partition), "
              "and does not support the partitioning method described in the original RoPE paper.");
    return MLLM_NO_ERROR;
}

ErrorCode XpRoPE::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    sin_params_.free();
    cos_params_.free();
    return MLLM_NO_ERROR;
}

} // namespace mllm::xnnpack