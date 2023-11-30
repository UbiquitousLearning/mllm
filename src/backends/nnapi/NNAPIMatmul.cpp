#include "NNAPIMatmul.hpp"
#include "Check.hpp"
#include "NNAPICommonOp.hpp"
#include "Types.hpp"

namespace mllm {

NNAPIMatmul::NNAPIMatmul(Backend *bn, string opName) :
    NNAPICommonOp(bn, opName) {
}

ErrorCode NNAPIMatmul::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
#ifdef DEBUG
    std::cout << "*NNAPI " << name() << " reshape*" << std::endl;
#endif
    CHECK_EQ(inputs.size(), 2);
    CHECK_EQ(outputs.size(), 1);

    CHECK_EQ(inputs[0]->batch(), inputs[1]->batch());
    CHECK_EQ(inputs[0]->batch(), 1);
    CHECK_EQ(inputs[0]->head(), inputs[1]->head());
    CHECK_EQ(inputs[0]->head(), 1);
    // different from cpu version, we assume that the input is 2D and already transposed
    CHECK_EQ(inputs[0]->width(), inputs[1]->height());

    bias_.reshape(1, 1, 1, outputs[0]->dimension());
    outputs[0]->reshape(inputs[0]->batch(),
                        inputs[0]->head(),
                        inputs[0]->sequence(),
                        inputs[1]->dimension());

    return NO_ERROR;
}

ErrorCode NNAPIMatmul::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
#ifdef DEBUG
    std::cout << "*NNAPI " << name() << " setUp*" << std::endl;
#endif

    bias_.alloc();
    bias_.fullData(0);
    outputs[0]->alloc();

    unsigned int batch_size = inputs[0]->batch() * inputs[0]->head() * inputs[0]->sequence();
    unsigned int num_units = inputs[1]->width();
    unsigned int input_size = inputs[0]->height();

    std::vector<uint32_t> inputIdxs;
    inputIdxs.push_back(getTensorIdx(inputs[0].get(), true, {batch_size, input_size})); // reshape to 2D
    inputIdxs.push_back(getTensorIdx(inputs[1].get(), true, {num_units, input_size}));
    inputIdxs.push_back(getTensorIdx(&bias_, true, {num_units}));
    inputIdxs.push_back(buildScalar(ANEURALNETWORKS_FUSED_NONE));

    return buildOperation(ANEURALNETWORKS_FULLY_CONNECTED, inputIdxs, {getTensorIdx(outputs[0].get(), true, {batch_size, num_units})});
}
} // namespace mllm
