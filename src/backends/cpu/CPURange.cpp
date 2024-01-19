
#include "CPURange.hpp"

namespace mllm {

CPURange::CPURange(Backend *bn,  string opName, int start, int end, int threadCount) : thread_count(threadCount),
    Op(bn, opName) {
    start_ = start;
    end_ =  end;
}

ErrorCode CPURange::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    outputs[0]->reshape(1, 1,  end_- start_, 1);
    return Op::reshape(inputs, outputs);
}

ErrorCode CPURange::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    for (int i = 0; i < end_-start_; ++i) {
        outputs[0]->setDataAt<float>(0, 0, i+start_,0, (float)i);
    }
    return Op::execute(inputs, outputs);
}

} // namespace mllm

