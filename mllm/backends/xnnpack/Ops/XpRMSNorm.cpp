#include "backends/xnnpack/Ops/XpRMSNorm.hpp"
#include "backends/xnnpack/XnnpackBackend.hpp"
#include "Types.hpp"
#include "xnnpack.h"

namespace mllm::xnnpack {

ErrorCode XpRMSNorm::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // do nothing
    return MLLM_NO_ERROR;
}

ErrorCode XpRMSNorm::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto xpb = (XnnpackBackend *)backend();
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    return Op::reshape(inputs, outputs);
}

ErrorCode XpRMSNorm::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto xpb = (XnnpackBackend *)backend();
    tryDefineAllXpTensors(xpb->getCurProcessingGraph(), inputs);
    tryDefineAllXpTensors(xpb->getCurProcessingGraph(), outputs);

    if (xpb->getCurProcessingGraph()->getExecCnt()) return MLLM_NO_ERROR;

    defineWeightTensor(xpb->getCurProcessingGraph(), &weight_params_);
    defineWeightTensor(xpb->getCurProcessingGraph(), &epsilon_param_);

    auto dtype = inputs[0]->dtype();
    size_t b = inputs[0]->shape()[0];
    size_t s = inputs[0]->shape()[1];
    size_t h = inputs[0]->shape()[2];
    size_t d = inputs[0]->shape()[3];

    // x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    auto x_powed = defineTemporaryTensor(xpb->getCurProcessingGraph(), {b, s, h, d}, dtype);
    {
        auto status = xnn_define_square(xpb->getCurProcessingGraph()->getXnnSubgraph(), inputs[0]->uuid(), x_powed, 0);
        if (status != xnn_status_success) {
            Log::error("XpRMSNorm: xnn_define_square failed");
            exit(-1);
        }
    }
    auto x_powed_mean = defineTemporaryTensor(xpb->getCurProcessingGraph(), {b, s, h, d}, dtype);
    {
        std::array<size_t, 1> along_axes{3};
        auto status = xnn_define_static_mean(xpb->getCurProcessingGraph()->getXnnSubgraph(), 1, along_axes.data(), x_powed, x_powed_mean, XNN_FLAG_KEEP_DIMS);
        if (status != xnn_status_success) {
            Log::error("XpRMSNorm: xnn_define_static_mean failed");
            exit(-1);
        }
    }
    auto x_pow_mean_eps = defineTemporaryTensor(xpb->getCurProcessingGraph(), {b, s, h, d}, dtype);
    {
        auto status = xnn_define_binary(xpb->getCurProcessingGraph()->getXnnSubgraph(), xnn_binary_add, nullptr, x_powed_mean, epsilon_param_.uuid(), x_pow_mean_eps, 0);
        if (status != xnn_status_success) {
            Log::error("XpRMSNorm: xnn_define_binary xnn_binary_add failed");
            exit(-1);
        }
    }
    auto x_pme_rsqrt = defineTemporaryTensor(xpb->getCurProcessingGraph(), {b, s, h, d}, dtype);
    {
        auto status = xnn_define_reciprocal_square_root(xpb->getCurProcessingGraph()->getXnnSubgraph(), x_pow_mean_eps, x_pme_rsqrt, 0);
        if (status != xnn_status_success) {
            Log::error("XpRMSNorm: xnn_define_reciprocal_square_root failed");
            exit(-1);
        }
    }
    auto x_1 = defineTemporaryTensor(xpb->getCurProcessingGraph(), {b, s, h, d}, dtype);
    {
        auto status = xnn_define_binary(xpb->getCurProcessingGraph()->getXnnSubgraph(), xnn_binary_multiply, nullptr, inputs[0]->uuid(), x_pme_rsqrt, x_1, 0);
        if (status != xnn_status_success) {
            Log::error("XpRMSNorm: xnn_define_binary xnn_binary_multiply x * epsed failed");
            exit(-1);
        }
    }

    {
        auto status = xnn_define_binary(xpb->getCurProcessingGraph()->getXnnSubgraph(), xnn_binary_multiply, nullptr, x_1, weight_params_.uuid(), outputs[0]->uuid(), 0);
        if (status != xnn_status_success) {
            Log::error("XpRMSNorm: xnn_define_binary xnn_binary_multiply x * weight failed");
            exit(-1);
        }
    }

    return MLLM_NO_ERROR;
}

ErrorCode XpRMSNorm::load(AbstructLoader &loader) {
    auto xpb = (XnnpackBackend *)backend();

    weight_params_.setName(name() + ".weight");
    weight_params_.reshape(1, 1, 1, norm_size_);
    if (loader.getDataType(weight_params_.name()) != MLLM_TYPE_COUNT) {
        weight_params_.setDtype(loader.getDataType(weight_params_.name()));
        weight_params_.alloc();
        loader.load(&weight_params_);
    } else {
        weight_params_.setDtype(MLLM_TYPE_F32);
        weight_params_.alloc();
    }

    // preset epsilon
    epsilon_param_.setName(name() + ".epsilon");
    epsilon_param_.reshape(1, 1, 1, 1);
    epsilon_param_.setDtype(MLLM_TYPE_F32);
    epsilon_param_.alloc();
    epsilon_param_.setDataAt(0, 0, 0, 0, epsilon_);

    // FIXME: make this process more efficient.
    if (add_unit_offset_) {
        int batch = weight_params_.batch();
        int dim = weight_params_.dimension();
        int seq = weight_params_.sequence();
        int head = weight_params_.head();
#pragma omp parallel for collapse(4) num_threads(thread_count)
        for (auto i = 0; i < batch * dim * seq * head; ++i) {
            *(weight_params_.hostPtr<float>()) = *(weight_params_.hostPtr<float>()) + 1;
        }
    }

    return Op::load(loader);
}

ErrorCode XpRMSNorm::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    weight_params_.free();
    epsilon_param_.free();
    return Op::free(inputs, outputs);
}

Op *XpRMSNormCreator::create(OpParam op_param, Backend *bk, const string &name, int thread_count) const {
    int normSize = (int)op_param["norm_size"];
    float epsilon = (float)op_param["epsilon"];
    bool add_unit_offset_ = (op_param.find("add_unit_offset") == op_param.end()) ? false : true;
    return new XpRMSNorm(bk, name, normSize, epsilon, add_unit_offset_, thread_count);
}
} // namespace mllm::xnnpack