#include "backends/xnnpack/Ops/XpSDPA.hpp"
#include "Types.hpp"
#include "xnnpack.h"
#include <cassert>
#include <cmath>

namespace mllm::xnnpack {

ErrorCode XpSDPA::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    Log::warn("XpSDPA: The inputs of XpSDPA suppose to be B, H, S, D");
    return MLLM_NO_ERROR;
}

ErrorCode XpSDPA::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto Q = inputs[0];
    auto K = inputs[1];
    auto V = inputs[2];

    int Q_BATCH = Q->shape()[0];
    int Q_HEAD = Q->shape()[1];
    int Q_SEQUENCE = Q->shape()[2];
    int Q_DIMENSION = Q->shape()[3];

    scale_params_.reshape(1, 1, 1, Q_DIMENSION);
    if (scale_params_.hostPtr<float>()) scale_params_.free();
    scale_params_.alloc();

    // scale = \sqrt{d_k}
    scale_params_.fullData<float>(1.f / (float)std::sqrt(V->shape()[3]));

    // mask
    //  mask_param_ reshape and alloc
    mask_params_.reshape(1, 1, Q_SEQUENCE, V->shape()[2]);
    if (mask_params_.hostPtr<float>()) mask_params_.free();
    mask_params_.alloc();

    memset(mask_params_.hostPtr<float>(), 0, mask_params_.count() * sizeof(float));

    // recompute mask
    int b = Q_BATCH;
    int h = Q_HEAD;
    int s = Q_SEQUENCE;
    int d = V->shape()[2];
    if (s > 1) {
#pragma omp parallel for collapse(4) num_threads(thread_count)
        for (int i_s = 0; i_s < s; ++i_s) {
            for (int i_d = 0; i_d < d; ++i_d) {
                if (i_d > i_s) {
                    mask_params_.setDataAt<float>({0, 0, i_s, i_d}, std::numeric_limits<float>::lowest());
                }
            }
        }
    }

    // BHSD.
    outputs[0]
        ->reshape(Q_BATCH, /*s*/ Q_SEQUENCE, /*h*/ Q_HEAD, Q_DIMENSION);
    return Op::reshape(inputs, outputs);
}

ErrorCode XpSDPA::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto xpb = (XnnpackBackend *)backend();
    tryDefineAllXpTensors(xpb->getCurProcessingGraph(), inputs);
    tryDefineAllXpTensors(xpb->getCurProcessingGraph(), outputs);

    if (xpb->getCurProcessingGraph()->getExecCnt()) return MLLM_NO_ERROR;

    auto Q = inputs[0];
    auto K = inputs[1];
    auto V = inputs[2];

    defineWeightTensor(xpb->getCurProcessingGraph(), &scale_params_, {(size_t)K->dimension()});
    // B, H, S, D
    defineWeightTensor(xpb->getCurProcessingGraph(), &mask_params_, {(size_t)Q->shape()[2], (size_t)K->shape()[2]});

    // B, H, S, D
    auto status = xnn_define_scaled_dot_product_attention(
        xpb->getCurProcessingGraph()->getXnnSubgraph(),
        xnn_attention_logits_cap_type_none,
        nullptr,
        Q->uuid(),
        K->uuid(),
        V->uuid(),
        scale_params_.uuid(),
        mask_params_.uuid(),
        outputs[0]->uuid(),
        0);

    if (status != xnn_status_success) {
        Log::error("XpSDPA xnn_define_scaled_dot_product_attention error");
        exit(-1);
    }

    return MLLM_NO_ERROR;
}

ErrorCode XpSDPA::load(AbstructLoader &loader) {
    return Op::load(loader);
}

ErrorCode XpSDPA::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    scale_params_.free();
    return Op::free(inputs, outputs);
}

Op *XpSDPACreator::create(OpParam op_param, Backend *bk, const string &name, int thread_count) const {
    return new XpSDPA(bk, name, thread_count);
}
} // namespace mllm::xnnpack