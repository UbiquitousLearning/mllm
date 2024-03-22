
#include "CPUMatmulINT8.hpp"
#include "Types.hpp"
#include "compute/Matmul.hpp"

namespace mllm {

CPUMatmulINT8::CPUMatmulINT8(Backend *bn, string opName, bool transpose0, bool transpose1, int threadCount) :
    thread_count(threadCount),
    Op(bn, opName) {
    transpose0_ = transpose0;
    transpose1_ = transpose1;
    thread_count = threadCount;
}

ErrorCode CPUMatmulINT8::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(inputs.size() == 2);
    assert(outputs.size() == 1);
    assert(inputs[0]->head() == inputs[1]->head());
    //    assert(inputs[0]->head() == 1);
    // assert(inputs[0]->batch() == inputs[1]->batch());
    if (!transpose0_ && !transpose1_) {
        /*
         N     |    C       |   H                   |  W
         -----------------------------------------------
         batch |out_channel | in_channel            |  1
         -----------------------------------------------
         batch |in_channel  | seq_len               |  1
         -----------------------------------------------
         batch |out_channel | seq_len               |  1
         */
        assert(inputs[0]->dimension() == inputs[1]->sequence());
        inputs[1]->transShape(SEQUENCE, DIMENSION);
        outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[1]->dimension());
    } else if (transpose1_) {
        /*
         N     |    C       |   H                   |  W
         -----------------------------------------------
         batch |in_channel | out_channel            |  1
         -----------------------------------------------
         batch |in_channel  | seq_len               |  1
         -----------------------------------------------
         batch |out_channel | seq_len               |  1
         */
        assert(inputs[0]->dimension() == inputs[1]->dimension());
        outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[1]->sequence());
    } else {
        /*
         N     |    C       |   H                   |  W
         -----------------------------------------------
         batch |out_channel | in_channel            |  1
         -----------------------------------------------
         batch |seq_len     | in_channel            |  1
         -----------------------------------------------
         batch |out_channel | seq_len               |  1
         */
        assert(inputs[0]->sequence() == inputs[1]->sequence());
        inputs[0]->transShape(SEQUENCE, DIMENSION);
        inputs[1]->transShape(SEQUENCE, DIMENSION);
        outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->dimension(), inputs[1]->dimension());
    }
    // outputs[0]->setDtype(activationDtype());
    return Op::reshape(inputs, outputs);
}

ErrorCode CPUMatmulINT8::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(inputs[0]->dtype() == MLLM_TYPE_I8);
    assert(inputs[1]->dtype() == MLLM_TYPE_I8);

    mat_mul_i8(inputs[0].get(), inputs[1].get(), outputs[0].get(), false, nullptr, transpose0_, transpose1_, thread_count);

    return Op::execute(inputs, outputs);
}

ErrorCode CPUMatmulINT8::load(AbstructLoader &loader) {
    // std::cout << name() << "  CPULinear load" << std::endl;
    scale1_.setName(name() + ".scale");
    scale1_.reshape(1, 1, 1, 1);
    scale1_.setBackend(this->backend());
    if (loader.getDataType(scale1_.name()) != MLLM_TYPE_COUNT) {
        scale1_.setDtype(loader.getDataType(scale1_.name()));
        scale1_.alloc();
        loader.load(&scale1_);
    } else {
        scale1_.setDtype(MLLM_TYPE_F32);
        scale1_.alloc();
    }
    scale2_.setName(name() + ".scale");
    scale2_.reshape(1, 1, 1, 1);
    scale2_.setBackend(this->backend());
    if (loader.getDataType(scale2_.name()) != MLLM_TYPE_COUNT) {
        scale2_.setDtype(loader.getDataType(scale2_.name()));
        scale2_.alloc();
        loader.load(&scale2_);
    } else {
        scale2_.setDtype(MLLM_TYPE_F32);
        scale2_.alloc();
    }
    return Op::load(loader);
}

ErrorCode CPUMatmulINT8::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs){
    scale1_.free();
    scale2_.free();
    return Op::free(inputs, outputs);
}

} // namespace mllm
