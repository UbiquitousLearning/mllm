#include "backends/xnnpack/Ops/XpSubGraphStart.hpp"
#include "Types.hpp"

namespace mllm::xnnpack {

ErrorCode XpSubGraphStart::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return MLLM_NO_ERROR;
}

ErrorCode XpSubGraphStart::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return MLLM_NO_ERROR;
}

ErrorCode XpSubGraphStart::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return ErrorCode::MLLM_NO_ERROR;
}

Op *XpSubGraphStartCreator::create(OpParam op_param, Backend *bk, const string &name, int thread_count) const {
    return new XpSubGraphStart(bk, name, thread_count);
}
} // namespace mllm::xnnpack