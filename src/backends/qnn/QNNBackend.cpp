#include "QNNBackend.hpp"
#include "QNNCommonOp.hpp"

namespace mllm {
QNNBackend::QNNBackend(shared_ptr<MemoryManager> &mm) :
    Backend(mm) {
    type_ = BackendType::MLLM_QNN;
    registerOps();
}

Op *QNNBackend::opCreate(const OpParam &op_param, string name, int threadCount) {
    OpType optype = OpType(op_param.find("type")->second);
    auto iter = map_creator_.find(optype);
    if (iter == map_creator_.end()) {
        printf("Don't support type \n");
        return nullptr;
    }
    Op *exe = iter->second->create(op_param, this, name);
    return exe;
}
void QNNBackend::registerOps() {
    addCreator(PARAMETER, (QNNBackend::Creator *)(new QNNCommCreator()));
    addCreator(ADD, (QNNBackend::Creator *)(new QNNCommCreator()));
    addCreator(CAUSALMASK, (QNNBackend::Creator *)(new QNNCommCreator()));
    addCreator(MATMUL, (QNNBackend::Creator *)(new QNNCommCreator()));
    addCreator(RMSNORM, (QNNBackend::Creator *)(new QNNCommCreator()));
    addCreator(ROPE, (QNNBackend::Creator *)(new QNNCommCreator()));
    addCreator(SCALE, (QNNBackend::Creator *)(new QNNCommCreator()));
    addCreator(SILU, (QNNBackend::Creator *)(new QNNCommCreator()));
    addCreator(SOFTMAX, (QNNBackend::Creator *)(new QNNCommCreator()));
    addCreator(LINEAR, (QNNBackend::Creator *)(new QNNCommCreator()));
    addCreator(EMBEDDING, (QNNBackend::Creator *)(new QNNCommCreator()));
    addCreator(MUL, (QNNBackend::Creator *)(new QNNCommCreator()));
    addCreator(VIEW, (QNNBackend::Creator *)(new QNNCommCreator()));
    addCreator(KVCACHE, (QNNBackend::Creator *)(new QNNCommCreator()));
    addCreator(RELU, (QNNBackend::Creator *)(new QNNCommCreator()));
    addCreator(RELU2, (QNNBackend::Creator *)(new QNNCommCreator()));
    addCreator(OP_GELU, (QNNBackend::Creator *)(new QNNCommCreator()));
    addCreator(QUICKGLUE, (QNNBackend::Creator *)(new QNNCommCreator()));
    addCreator(LAYERNORM, (QNNBackend::Creator *)(new QNNCommCreator()));
    addCreator(SPLIT, (QNNBackend::Creator *)(new QNNCommCreator()));
    addCreator(GATHER, (QNNBackend::Creator *)(new QNNCommCreator()));
    addCreator(CONVOLUTION2D, (QNNBackend::Creator *)(new QNNCommCreator()));
    addCreator(AVGPOOL2D, (QNNBackend::Creator *)(new QNNCommCreator()));
    addCreator(MAXPOOL2D, (QNNBackend::Creator *)(new QNNCommCreator()));
    addCreator(CONVOLUTION3D, (QNNBackend::Creator *)(new QNNCommCreator()));
    addCreator(CAT, (QNNBackend::Creator *)(new QNNCommCreator()));
    addCreator(TRANSPOSE, (QNNBackend::Creator *)(new QNNCommCreator()));
    addCreator(SUBDIM, (QNNBackend::Creator *)(new QNNCommCreator()));
    addCreator(DIVISION, (QNNBackend::Creator *)(new QNNCommCreator()));
    addCreator(NORM, (QNNBackend::Creator *)(new QNNCommCreator()));
    addCreator(SHAPE, (QNNBackend::Creator *)(new QNNCommCreator()));
    addCreator(MEAN, (QNNBackend::Creator *)(new QNNCommCreator()));
    addCreator(RANGE, (QNNBackend::Creator *)(new QNNCommCreator()));
    addCreator(WHERE, (QNNBackend::Creator *)(new QNNCommCreator()));
    addCreator(REPLACE, (QNNBackend::Creator *)(new QNNCommCreator()));
}

} // namespace mllm
