#include "backends/xnnpack/Ops/XpBinary.hpp"
#include "Types.hpp"

namespace mllm::xnnpack {

ErrorCode XpAdd::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // do not alloc memory.
    // setup tensor and grapgh in xnnpack subgraph.
    // TODO
    return MLLM_NO_ERROR;
}

ErrorCode XpAdd::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // xnnpack will do reshape for us.
    return MLLM_NO_ERROR;
}

ErrorCode XpAdd::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // do not execute this op
    return MLLM_NO_ERROR;
}

Op *XpAddCreator::create(OpParam op_param, Backend *bk, const string &name, int thread_count) const {
    return new XpAdd(bk, name, thread_count);
}
} // namespace mllm::xnnpack