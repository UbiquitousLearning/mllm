/**
 * @file XpMatmulFunc.cpp
 * @author your name (you@domain.com)
 * @version 0.1
 * @date 2024-10-23
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "backends/xnnpack/Functions/XpMatmulFunc.hpp"
#include "xnnpack.h"
#include "backends/xnnpack/Utils/Logger.hpp"
#include "backends/xnnpack/XnnpackBackend.hpp"

namespace mllm::xnnpack {

void XpMatmulFunction::reshape(vector<shared_ptr<Tensor>> outputs, vector<shared_ptr<Tensor>> inputs, vector<float> args) {
    assert(inputs[0]->dimension() == inputs[1]->sequence());
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[1]->dimension());
    outputs[0]->setDtype(inputs[0]->dtype());
}

void XpMatmulFunction::execute(vector<shared_ptr<Tensor>> outputs, vector<shared_ptr<Tensor>> inputs, vector<float> args) {
    auto xpb = (XnnpackBackend *)(inputs[0]->backend());
    tryDefineAllXpTensors(xpb->getCurProcessingGraph(), inputs);
    tryDefineAllXpTensors(xpb->getCurProcessingGraph(), outputs);

    if (xpb->getCurProcessingGraph()->getExecCnt()) return;

    auto status = xnn_define_batch_matrix_multiply(
        xpb->getCurProcessingGraph()->getXnnSubgraph(),
        inputs[0]->uuid(),
        inputs[1]->uuid(),
        outputs[0]->uuid(),
        0);
    if (status != xnn_status_success) {
        Log::error("XpMatmulFunction xnn_define_batch_matrix_multiply failed");
        exit(-1);
    }
}

} // namespace mllm::xnnpack