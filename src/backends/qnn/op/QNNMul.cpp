
#include "QNNMul.hpp"
#include "Types.hpp"
#include "QNNCommonOp.hpp"

namespace mllm {
QNNMul::QNNMul(Backend *bn, string opName) :
    QNNCommonOp(bn, opName) {

    scale_.setBackend(bn);
}

ErrorCode QNNMul::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(inputs.size() == 2);
    assert(outputs.size() == 1);
    assert(inputs[0]->batch() == inputs[1]->batch());
    assert(inputs[0]->head() == inputs[1]->head());
    assert(inputs[0]->sequence() == inputs[1]->sequence());
    assert(inputs[0]->dimension() == inputs[1]->dimension());
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    return Op::reshape(inputs, outputs);
}

ErrorCode QNNMul::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {

    if (inputs[0]->dtype() == MLLM_TYPE_I8) {

        outputs[0]->setDtype(MLLM_TYPE_I8);
        return graphAddNode(name(), "ElementWiseMultiply", inputs, outputs, {}, "qti.aisw", true,  &scale_);

    } else {

        // FP Mul use our op package.
        outputs[0]->setDtype(MLLM_TYPE_F32);
        auto outName = outputs[0]->name();

        uint32_t dimensionsOutput[4];

        dimensionsOutput[0] = static_cast<uint32_t>(outputs[0]->batch());
        dimensionsOutput[1] = static_cast<uint32_t>(outputs[0]->sequence());
        dimensionsOutput[2] = static_cast<uint32_t>(outputs[0]->head());
        dimensionsOutput[3] = static_cast<uint32_t>(outputs[0]->dimension());

        auto type = QNN_DATATYPE_FLOAT_32;

        if (inputs[0]->dtype() == MLLM_TYPE_F16) {
            type = QNN_DATATYPE_FLOAT_16;
            outputs[0]->setDtype(MLLM_TYPE_F16);
        }

        vector<Qnn_Tensor_t> outputTensor = {{QNN_TENSOR_VERSION_1,
                                            {.v1 = {
                                                .id = 0,
                                                .name = outName.c_str(),
                                                .type = getOutputTensorType(outputs[0]),
                                                .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                                .dataType = type,
                                                .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                                    {.scaleOffsetEncoding = {.scale = 0.0000000000000000f,
                                                                                            .offset = 0}}},
                                                .rank = 4,
                                                .dimensions = dimensionsOutput,
                                                .memType = QNN_TENSORMEMTYPE_RAW,
                                                .clientBuf = {.data = nullptr,
                                                                .dataSize = 0}}}}};
        return graphAddNode(name(), "LLaMAMul", {inputs[0]->name(), inputs[1]->name()}, outputTensor, {}, "LLaMAPackage");
    }

    // return graphAddNode(name(), "LLaMAMul", inputs, outputs, {}, "LLaMAPackage");
    
}

ErrorCode QNNMul::load(AbstructLoader &loader) {
    string scaleName = name();

    std::string wordToRemove = "gate_proj.relu-00_mul_";
    int pos = scaleName.find(wordToRemove);
    if (pos != -1) {
        scaleName.erase(pos, wordToRemove.length());
    }

    scale_.setName(scaleName + "down_proj.input_scale");
    scale_.reshape(1, 1, 1, 1);
    scale_.setDtype(MLLM_TYPE_F32);
    scale_.alloc();
    loader.load(&scale_);

    // std::cout <<  scale_.hostPtr<float>()[0] << std::endl;

    return Op::load(loader);
}


} // namespace mllm
