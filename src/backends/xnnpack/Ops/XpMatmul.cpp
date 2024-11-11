#include "backends/xnnpack/Ops/XpMatmul.hpp"
#include "xnnpack.h"
#include <cassert>

namespace mllm::xnnpack {

ErrorCode XpMatMul::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // do nothing
    return MLLM_NO_ERROR;
}

ErrorCode XpMatMul::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(inputs.size() == 2);
    assert(outputs.size() == 1);

    if (transpose_c_ || transpose_a_) {
        Log::error("XpMatMul::The transpose of C or A is not support yet.");
        exit(-1);
    }

    auto A = inputs[0];
    auto B = inputs[1];
    auto C = outputs[0];

    int m = 0;
    int k = 0;
    int n = 0;

    // A: M x K
    // B: K x N
    if (!transpose_a_ && !transpose_b_) {
        assert(A->shape()[3] == B->shape()[2]);
        m = A->shape()[2];
        k = A->shape()[3];
        n = B->shape()[3];

        goto xp_matmul_process_outputs;
    }

    // A: M x K
    // B: N x K
    if (!transpose_a_ && transpose_b_) {
        assert(A->shape()[3] == B->shape()[3]);
        m = A->shape()[2];
        k = A->shape()[3];
        n = B->shape()[2];

        goto xp_matmul_process_outputs;
    }

    // A: K x M
    // B: K x N
    if (transpose_a_ && !transpose_b_) {
        assert(A->shape()[2] == B->shape()[2]);
        m = A->shape()[3];
        k = A->shape()[2];
        n = B->shape()[3];

        goto xp_matmul_process_outputs;
    }

    // A: K x M
    // B: N x K
    if (transpose_a_ && transpose_b_) {
        assert(A->shape()[2] == B->shape()[3]);

        m = A->shape()[3];
        k = A->shape()[2];
        n = B->shape()[2];

        goto xp_matmul_process_outputs;
    }

xp_matmul_process_outputs:
    int b = A->batch();
    int h = A->head();

    assert(b == B->batch());
    assert(h == B->head());

    if (!transpose_c_) {
        outputs[0]->reshape(b, h, m, n);
    } else {
        outputs[0]->reshape(b, h, n, m);
    }
    return Op::reshape(inputs, outputs);
}

ErrorCode XpMatMul::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto xpb = (XnnpackBackend *)inputs[0]->backend();
    tryDefineAllXpTensors(xpb->getCurProcessingGraph(), inputs);
    tryDefineAllXpTensors(xpb->getCurProcessingGraph(), outputs);

    if (xpb->getCurProcessingGraph()->getExecCnt()) return MLLM_NO_ERROR;

    uint32_t flag = 0;

    if (transpose_a_) flag |= XNN_FLAG_TRANSPOSE_A;
    if (transpose_b_) flag |= XNN_FLAG_TRANSPOSE_B;

    auto status = xnn_define_batch_matrix_multiply(xpb->getCurProcessingGraph()->getXnnSubgraph(), inputs[0]->uuid(), inputs[1]->uuid(), outputs[0]->uuid(), flag);

    if (status != xnn_status_success) {
        Log::error("XpMatMul::execute Error");
        exit(-1);
    }

    return MLLM_NO_ERROR;
}

Op *XpMatMulCreator::create(OpParam op_param, Backend *bk, const string &name, int thread_count) const {
    bool transpose0 = (bool)op_param["transpose0"];
    bool transpose1 = (bool)op_param["transpose1"];
    return new XpMatMul(bk, transpose0, transpose1, false, name, thread_count);
}

} // namespace mllm::xnnpack