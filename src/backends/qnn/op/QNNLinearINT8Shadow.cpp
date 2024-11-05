
#include "QNNLinearINT8Shadow.hpp"
#include "QnnTypes.h"
#include "Types.hpp"
#include "QNNCommonOp.hpp"
#include <cstdint>
#include <memory>
#include "backends/cpu/compute/Matmul.hpp"

namespace mllm {
QNNLinearINT8Shadow::QNNLinearINT8Shadow(Backend *bn, string opName, int in_features, int out_features, bool bias) :
    QNNCommonOp(bn, opName) {
}

ErrorCode QNNLinearINT8Shadow::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(inputs.size() == 3);
    assert(outputs.size() == 3);

    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    outputs[1]->reshape(inputs[1]->batch(), inputs[1]->head(), inputs[1]->sequence(), inputs[1]->dimension());
    outputs[2]->reshape(inputs[2]->batch(), inputs[2]->head(), inputs[2]->sequence(), inputs[2]->dimension());
    return Op::reshape(inputs, outputs);
}

ErrorCode QNNLinearINT8Shadow::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    inputs[0]->setBackend(qnnBackend_);
    // inputs[0]->setDtype(MLLM_TYPE_F32);
    inputs[0]->alloc();

    qnnBackend_->pushOutputBuffers(inputs[0]->hostPtr<uint8_t>());

    inputs[1]->setBackend(qnnBackend_);
    // inputs[1]->setDtype(MLLM_TYPE_I8);
    inputs[1]->alloc();

    qnnBackend_->pushOutputBuffers(inputs[1]->hostPtr<uint8_t>());

    inputs[2]->setBackend(qnnBackend_);
    // inputs[2]->setDtype(MLLM_TYPE_F32);
    inputs[2]->alloc();

    qnnBackend_->pushOutputBuffers(inputs[2]->hostPtr<uint8_t>());

    outputs[0]->setDtype(inputs[0]->dtype());
    outputs[0]->deepCopyFrom(inputs[0].get(), true);
    outputs[1]->setDtype(inputs[1]->dtype());
    outputs[1]->deepCopyFrom(inputs[1].get(), true);
    outputs[2]->setDtype(inputs[2]->dtype());
    outputs[2]->deepCopyFrom(inputs[2].get(), true);

    return MLLM_NO_ERROR;
}

ErrorCode QNNLinearINT8Shadow::load(AbstructLoader &loader) {
    return Op::load(loader);
}

ErrorCode QNNLinearINT8Shadow::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return Op::free(inputs, outputs);
}

} // namespace mllm
