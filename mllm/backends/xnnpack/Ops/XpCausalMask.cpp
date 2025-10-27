#include "backends/xnnpack/Ops/XpCausalMask.hpp"
#include "xnnpack.h"
#include "backends/xnnpack/XnnpackBackend.hpp"

namespace mllm::xnnpack {
ErrorCode XpCausalMask::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return MLLM_NO_ERROR;
}

ErrorCode XpCausalMask::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    //  mask_param_ reshape and alloc
    mask_param_.reshape(1, 1, inputs[0]->sequence(), inputs[0]->dimension());
    mask_param_.setDtype(DataType::MLLM_TYPE_F32);
    if (mask_param_.hostPtr<float>()) mask_param_.free();
    mask_param_.alloc();

    memset(mask_param_.hostPtr<float>(), 0, mask_param_.count() * sizeof(float));

    // recompute mask
    int b = inputs[0]->batch();
    int h = inputs[0]->head();
    int s = inputs[0]->sequence();
    int d = inputs[0]->dimension();
    if (s > 1) {
#pragma omp parallel for collapse(4) num_threads(thread_count)
        for (int i_s = 0; i_s < s; ++i_s) {
            for (int i_d = 0; i_d < d; ++i_d) {
                if (i_d > i_s) {
                    mask_param_.setDataAt<float>({0, 0, i_s, i_d}, std::numeric_limits<float>::lowest());
                }
            }
        }
    }

    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    return Op::reshape(inputs, outputs);
}

ErrorCode XpCausalMask::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto xpb = (XnnpackBackend *)backend();
    tryDefineAllXpTensors(xpb->getCurProcessingGraph(), inputs);
    tryDefineAllXpTensors(xpb->getCurProcessingGraph(), outputs);
    if (xpb->getCurProcessingGraph()->getExecCnt()) return MLLM_NO_ERROR;

    defineWeightTensor(xpb->getCurProcessingGraph(), &mask_param_);

    auto statuts = xnn_define_binary(
        xpb->getCurProcessingGraph()->getXnnSubgraph(),
        xnn_binary_add,
        nullptr,
        inputs[0]->uuid(),
        mask_param_.uuid(),
        outputs[0]->uuid(),
        0);

    if (statuts != xnn_status_success) {
        Log::error("XpCausalMask xnn_define_binary error");
        exit(-1);
    }

    return MLLM_NO_ERROR;
}

Op *XpCausalMaskCreator::create(OpParam op_param, Backend *bk, const string &name, int thread_count) const {
    return new XpCausalMask(bk, name, thread_count);
}
} // namespace mllm::xnnpack
