
#include "QNNSuperSiLU.hpp"
#include "Types.hpp"
#include "QNNCommonOp.hpp"

namespace mllm {
QNNSuperSiLU::QNNSuperSiLU(Backend *bn, string opName) :
    QNNCommonOp(bn, opName) {

    a_scale_.setBackend(bn);
    b_scale_.setBackend(bn);
    o_scale_.setBackend(bn);
}

ErrorCode QNNSuperSiLU::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    return Op::reshape(inputs, outputs);
}

ErrorCode QNNSuperSiLU::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {

    auto outName = outputs[0]->name();

    uint32_t dimensionsOutput[4];


    dimensionsOutput[0] = static_cast<uint32_t>(outputs[0]->batch());
    dimensionsOutput[1] = static_cast<uint32_t>(outputs[0]->sequence());
    dimensionsOutput[2] = static_cast<uint32_t>(outputs[0]->head());
    dimensionsOutput[3] = static_cast<uint32_t>(outputs[0]->dimension()); 

    float aScale = 0;
    aScale = a_scale_.hostPtr<float>()[0] / 127.0;
    // aScale = roundf(aScale * 100000) / 100000;

    float bScale = 0;
    bScale = b_scale_.hostPtr<float>()[0] / 127.0;
    // bScale = roundf(bScale * 100000) / 100000;

    float oScale = 0;
    oScale = o_scale_.hostPtr<float>()[0] / 127.0;
    // oScale = roundf(oScale * 100000) / 100000;

    auto paramsSuperSiLuNameA = name() + ".supersilu_params.a_scale";
    auto paramsSuperSiLuNameB = name() + ".supersilu_params.b_scale";
    auto paramsSuperSiLuNameO = name() + ".supersilu_params.o_scale";

    uint32_t paramsSuperSiLuDimension[1] = {1};

    vector<Qnn_Param_t> paramsSuperSiLu = {
            {.paramType = QNN_PARAMTYPE_TENSOR,
             .name = "a_scale",
             {.tensorParam =
                  (Qnn_Tensor_t){.version = QNN_TENSOR_VERSION_1,
                                 {.v1 = {
                                      .id = 0,
                                      .name = paramsSuperSiLuNameA.c_str(),
                                      .type = QNN_TENSOR_TYPE_STATIC,
                                      .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                      .dataType = QNN_DATATYPE_FLOAT_32,
                                      .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                         QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                         {.scaleOffsetEncoding = {.scale = 0.0000000000000000f,
                                                                                  .offset = 0}}},
                                      .rank = 1,
                                      .dimensions = paramsSuperSiLuDimension,
                                      .memType = QNN_TENSORMEMTYPE_RAW,
                                      {.clientBuf = {.data = (uint8_t *)&aScale,
                                                     .dataSize = sizeof(float)}}}}}}},
            {.paramType = QNN_PARAMTYPE_TENSOR,
             .name = "b_scale",
             {.tensorParam =
                  (Qnn_Tensor_t){.version = QNN_TENSOR_VERSION_1,
                                 {.v1 = {
                                      .id = 0,
                                      .name = paramsSuperSiLuNameB.c_str(),
                                      .type = QNN_TENSOR_TYPE_STATIC,
                                      .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                      .dataType = QNN_DATATYPE_FLOAT_32,
                                      .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                         QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                         {.scaleOffsetEncoding = {.scale = 0.0000000000000000f,
                                                                                  .offset = 0}}},
                                      .rank = 1,
                                      .dimensions = paramsSuperSiLuDimension,
                                      .memType = QNN_TENSORMEMTYPE_RAW,
                                      {.clientBuf = {.data = (uint8_t *)&bScale,
                                                     .dataSize = sizeof(float)}}}}}}},
            {.paramType = QNN_PARAMTYPE_TENSOR,
             .name = "o_scale",
             {.tensorParam =
                  (Qnn_Tensor_t){.version = QNN_TENSOR_VERSION_1,
                                 {.v1 = {
                                      .id = 0,
                                      .name = paramsSuperSiLuNameO.c_str(),
                                      .type = QNN_TENSOR_TYPE_STATIC,
                                      .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                      .dataType = QNN_DATATYPE_FLOAT_32,
                                      .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                         QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                         {.scaleOffsetEncoding = {.scale = 0.0000000000000000f,
                                                                                  .offset = 0}}},
                                      .rank = 1,
                                      .dimensions = paramsSuperSiLuDimension,
                                      .memType = QNN_TENSORMEMTYPE_RAW,
                                      {.clientBuf = {.data = (uint8_t *)&oScale,
                                                     .dataSize = sizeof(float)}}}}}}},
                                                     };
        

    vector<Qnn_Tensor_t> outputTensor = {{QNN_TENSOR_VERSION_1,
                                          {.v1 = {
                                               .id = 0,
                                               .name = outName.c_str(),
                                               .type = getOutputTensorType(outputs[0]),
                                               .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                               .dataType = QNN_DATATYPE_SFIXED_POINT_8,
                                               .quantizeParams = {QNN_DEFINITION_DEFINED,
                                                    QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                    {.scaleOffsetEncoding = {.scale  = oScale,
                                                                            .offset = 0}}},
                                               .rank = 4,
                                               .dimensions = dimensionsOutput,
                                               .memType = QNN_TENSORMEMTYPE_RAW,
                                               {.clientBuf = {.data = nullptr,
                                                              .dataSize = 0}}}}}};
    return graphAddNode(name(), "LLaMASuperSiLU", {inputs[0]->name(), inputs[1]->name()}, outputTensor, paramsSuperSiLu, "LLaMAPackage");
}

ErrorCode QNNSuperSiLU::load(AbstructLoader &loader) {
    string opName = name();
    std::string wordToRemove = ".supersilu";

    int pos = opName.find(wordToRemove);
    if (pos != -1) {
        opName.erase(pos, wordToRemove.length());
    }

    a_scale_.setName(opName + ".gate_proj.output_scale");
    a_scale_.reshape(1, 1, 1, 1);
    a_scale_.setDtype(MLLM_TYPE_F32);
    a_scale_.alloc();
    loader.load(&a_scale_);

    b_scale_.setName(opName + ".up_proj.output_scale");
    b_scale_.reshape(1, 1, 1, 1);
    b_scale_.setDtype(MLLM_TYPE_F32);
    b_scale_.alloc();
    loader.load(&b_scale_);

    o_scale_.setName(opName + ".down_proj.input_scale");
    o_scale_.reshape(1, 1, 1, 1);
    o_scale_.setDtype(MLLM_TYPE_F32);
    o_scale_.alloc();
    loader.load(&o_scale_);

    return Op::load(loader);
}

} // namespace mllm

