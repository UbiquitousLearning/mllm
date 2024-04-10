
#include "QNNDequantize.hpp"
#include "QnnTypes.h"
#include "Types.hpp"
#include "QNNCommonOp.hpp"
#include <cassert>

namespace mllm {
QNNDequantize::QNNDequantize(Backend *bn, string opName, bool isNSHD) :
    QNNCommonOp(bn, opName) {
        isNSHD_ = isNSHD;
        scale_.setBackend(bn);
}

ErrorCode QNNDequantize::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(outputs.size() == 1);
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    return Op::reshape(inputs, outputs);
}

ErrorCode QNNDequantize::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto outName = outputs[0]->name();
    uint32_t dimensionsOutput[4];

    if (isNSHD_) {
        dimensionsOutput[0] = static_cast<uint32_t>(outputs[0]->batch());
        dimensionsOutput[1] = static_cast<uint32_t>(outputs[0]->sequence());
        dimensionsOutput[2] = static_cast<uint32_t>(outputs[0]->head());
        dimensionsOutput[3] = static_cast<uint32_t>(outputs[0]->dimension());
    } else {

        dimensionsOutput[0] = static_cast<uint32_t>(outputs[0]->batch());
        dimensionsOutput[1] = static_cast<uint32_t>(outputs[0]->head());
        dimensionsOutput[2] = static_cast<uint32_t>(outputs[0]->sequence());
        dimensionsOutput[3] = static_cast<uint32_t>(outputs[0]->dimension());
    }

    float dequantScale = 0;
    dequantScale = scale_.hostPtr<float>()[0]  / 127.0;
    dequantScale = roundf(dequantScale * 10000) / 10000;
    
    vector<Qnn_Tensor_t> outputTensor = {{QNN_TENSOR_VERSION_1,
                                          {.v1 = {
                                               .id = 0,
                                               .name = outName.c_str(),
                                               .type = getOutputTensorType(outputs[0]),
                                               .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                               .dataType = QNN_DATATYPE_FLOAT_32,
                                               .quantizeParams = {QNN_DEFINITION_DEFINED,
                                                                  QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                                  {.scaleOffsetEncoding = {.scale = dequantScale, .offset = 0}}},
                                               .rank = 4,
                                               .dimensions = dimensionsOutput,
                                               .memType = QNN_TENSORMEMTYPE_RAW,
                                               {.clientBuf = {.data = nullptr,
                                                              .dataSize = 0}}}}}};
    return graphAddNode(name(), "Dequantize", {inputs[0]->name()}, outputTensor);
}

ErrorCode QNNDequantize::load(AbstructLoader &loader) {

    std::cout << "load dequantize" << std::endl;


    string scaleName = name();

    std::string wordToRemove = "dequantize";
    int pos = scaleName.find(wordToRemove);
    if (pos != -1) {
        scaleName.erase(pos, wordToRemove.length());
    }

    scale_.setName(scaleName + "output_scale");
    scale_.reshape(1, 1, 1, 1);
    scale_.setDtype(MLLM_TYPE_F32);
    scale_.alloc();
    loader.load(&scale_);

    std::cout <<  scale_.hostPtr<float>()[0] << std::endl;

    return Op::load(loader);
}
} // namespace mllm
