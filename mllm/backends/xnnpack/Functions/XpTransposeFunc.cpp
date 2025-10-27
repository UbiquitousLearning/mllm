#include <array>
#include "backends/xnnpack/Functions/XpTransposeFunc.hpp"
#include "Types.hpp"

namespace mllm::xnnpack {

void XpTransposeFunction::reshape(vector<shared_ptr<Tensor>> outputs, vector<shared_ptr<Tensor>> inputs, vector<float> args) {
    Chl axis0_ = (Chl)args[0];
    Chl axis1_ = (Chl)args[1];

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
    } else if (axis0_ == SEQUENCE && axis1_ == HEAD) {
        if (inputs[0]->ctype() == BSHD) {
            outputs[0]->reshape(inputs[0]->batch(), inputs[0]->sequence(), inputs[0]->head(), inputs[0]->dimension());
        }
    }
}

void XpTransposeFunction::execute(vector<shared_ptr<Tensor>> outputs, vector<shared_ptr<Tensor>> inputs, vector<float> args) {
    auto xpb = (XnnpackBackend *)inputs[0]->backend();
    tryDefineAllXpTensors(xpb->getCurProcessingGraph(), inputs);
    tryDefineAllXpTensors(xpb->getCurProcessingGraph(), outputs);

    if (xpb->getCurProcessingGraph()->getExecCnt()) return;

    std::array<size_t, 4> perm{0, 1, 2, 3};

    Chl axis0_ = (Chl)args[0];
    Chl axis1_ = (Chl)args[1];

    // inputs[0]->transShape(SEQUENCE, DIMENSION);
    // B, S, H, D
    if (axis0_ == SEQUENCE && axis1_ == DIMENSION) {
        if (inputs[0]->ctype() == BSHD) {
            std::swap(perm[1], perm[3]);
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
    } else if (axis0_ == SEQUENCE && axis1_ == HEAD) {
        if (inputs[0]->ctype() == BSHD) {
            std::swap(perm[1], perm[2]);
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
        Log::error("XpTransposeFunction::execute Error");
        exit(-1);
    }
}

} // namespace mllm::xnnpack