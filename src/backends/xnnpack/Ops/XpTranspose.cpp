#include "backends/xnnpack/Ops/XpTranspose.hpp"
#include "Types.hpp"
#include "xnnpack.h"
#include <utility>

namespace mllm::xnnpack {

ErrorCode XpTranspose::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return MLLM_NO_ERROR;
}

ErrorCode XpTranspose::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // inputs[0]->transShape(SEQUENCE, DIMENSION);
    if (axis0_ == SEQUENCE && axis1_ == DIMENSION) {
        if (inputs[0]->ctype() == BSHD) {
            outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->dimension(), inputs[0]->sequence());
        }
    } else if (axis0_ == THW && axis1_ == CHANNLE) {
        if (inputs[0]->ctype() == BCTHW) {
            outputs[0]->reshape(inputs[0]->batch(), inputs[0]->time(), inputs[0]->height(), inputs[0]->width(), inputs[0]->channel());
        }
    } else if (axis0_ == BATCH && axis1_ == SEQUENCE) {
        if (inputs[0]->ctype() == BSHD) {
            outputs[0]->reshape(inputs[0]->sequence(), inputs[0]->head(), inputs[0]->batch(), inputs[0]->dimension());
        }
    }
    return MLLM_NO_ERROR;
}

ErrorCode XpTranspose::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto xpb = (XnnpackBackend *)inputs[0]->backend();
    tryDefineAllXpTensors(xpb->getCurProcessingGraph(), inputs);
    tryDefineAllXpTensors(xpb->getCurProcessingGraph(), outputs);

    if (xpb->getCurProcessingGraph()->getExecCnt()) return MLLM_NO_ERROR;

    std::array<size_t, 4> perm{0, 1, 2, 3};

    // inputs[0]->transShape(SEQUENCE, DIMENSION);
    if (axis0_ == SEQUENCE && axis1_ == DIMENSION) {
        if (inputs[0]->ctype() == BSHD) {
            std::swap(perm[2], perm[3]);
        } else {
            Log::error("XpTransposeFunction NYI");
            exit(-1);
        }
    } else if (axis0_ == THW && axis1_ == CHANNLE) {
        if (inputs[0]->ctype() == BCTHW) {
            Log::error("XpTransposeFunction NYI");
            exit(-1);
            outputs[0]->reshape(inputs[0]->batch(), inputs[0]->time(), inputs[0]->height(), inputs[0]->width(), inputs[0]->channel());
        } else {
            Log::error("XpTransposeFunction NYI");
            exit(-1);
        }
    } else if (axis0_ == BATCH && axis1_ == SEQUENCE) {
        if (inputs[0]->ctype() == BSHD) {
            Log::error("XpTransposeFunction NYI");
            exit(-1);
            outputs[0]
                ->reshape(inputs[0]->sequence(), inputs[0]->head(), inputs[0]->batch(), inputs[0]->dimension());
        } else {
            Log::error("XpTransposeFunction NYI");
            exit(-1);
        }
    } else {
        Log::error("XpTransposeFunction NYI");
        exit(-1);
    }

    auto status = xnn_define_static_transpose(xpb->getCurProcessingGraph()->getXnnSubgraph(), 4, perm.data(), inputs[0]->uuid(), outputs[0]->uuid(), 0);

    if (status != xnn_status_success) {
        Log::error("XpTranspose::execute Error");
        exit(-1);
    }
    return MLLM_NO_ERROR;
}

Op *XpTransposeCreator::create(OpParam op_param, Backend *bk, const string &name, int thread_count) const {
    int axis0 = (int)op_param["axis0"];
    int axis1 = (int)op_param["axis1"];
    return new XpTranspose(bk, axis0, axis1, name, thread_count);
}
} // namespace mllm::xnnpack
