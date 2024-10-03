#include "backends/Xnnpack/Ops/XpBinary.hpp"

namespace mllm::xnnpack {

Op *XpAddCreator::create(OpParam op_param, Backend *bk, const string &name, int thread_count) const {
    return new XpAdd(bk, name, thread_count);
}
} // namespace mllm::xnnpack