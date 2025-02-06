
#include "QNNLinearINT8.hpp"
#include "QnnTypes.h"
#include "Types.hpp"
#include "QNNCommonOp.hpp"
#include <cstdint>
#include <memory>

namespace mllm {
QNNLinearINT8::QNNLinearINT8(Backend *bn, string opName, int in_features, int out_features, bool bias) :
    QNNCommonOp(bn, opName), in_features_(in_features), out_features_(out_features), support_bias_(bias) {
    weight_.setBackend(bn);
    bias_.setBackend(bn);

    weightScale_.setBackend(bn);
    biasScale_.setBackend(bn);
    outputScale_.setBackend(bn);
    inputScale_.setBackend(bn);
}

ErrorCode QNNLinearINT8::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(inputs.size() == 1);
    assert(outputs.size() == 1);
    // N     |    C       |   H                   |  W
    // -----------------------------------------------
    // 1     |out_channel | in_channel            |  1
    //       |out_features| in_features           |
    // -----------------------------------------------
    // batch |in_channel  | seq_len               |  1
    //       |in_features | inputs[0]->sequence()   |
    // -----------------------------------------------
    // batch |out_channel | seq_len               |  1
    //       |out_features|  inputs[0]->sequence()  |
    assert(inputs[0]->head() == 1);
    assert(in_features_ == inputs[0]->dimension());
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), out_features_);
    return Op::reshape(inputs, outputs);
}

ErrorCode QNNLinearINT8::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    outputs[0]->setDtype(MLLM_TYPE_I8);
    // add matmul param to qnn
    vector<Qnn_Param_t> paramsMatmul = {
        {.paramType = QNN_PARAMTYPE_SCALAR,
         .name = "transpose_in0",
         .scalarParam = (Qnn_Scalar_t){QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}},
        {.paramType = QNN_PARAMTYPE_SCALAR,
         .name = "transpose_in1",
         .scalarParam = (Qnn_Scalar_t){QNN_DATATYPE_BOOL_8, {.bool8Value = 1}}}};

    uint32_t dimensions_InceptionV3_InceptionV3_Conv2d_1a_3x3_Conv2D_dilation[] = {2};
    uint32_t InceptionV3_InceptionV3_Conv2d_1a_3x3_Conv2D_dilation[] = {1, 1};
    uint32_t dimensions_InceptionV3_InceptionV3_Conv2d_1a_3x3_Conv2D_pad_amount[] = {2, 2};
    uint32_t InceptionV3_InceptionV3_Conv2d_1a_3x3_Conv2D_pad_amount[] = {0, 0, 0, 0};
    uint32_t dimensions_InceptionV3_InceptionV3_Conv2d_1a_3x3_Conv2D_stride[] = {2};
    uint32_t InceptionV3_InceptionV3_Conv2d_1a_3x3_Conv2D_stride[] = {1, 1};

    string strideName = name() + ".stride";
    string padName = name() + ".pad";
    vector<Qnn_Param_t> params_InceptionV3_InceptionV3_Conv2d_1a_3x3_Conv2D = {
        {.paramType = QNN_PARAMTYPE_TENSOR,
         .name = "stride",
         .tensorParam =
             (Qnn_Tensor_t){
                 .version = QNN_TENSOR_VERSION_1,
                 .v1 = {.id = 0,
                        .name = strideName.c_str(),
                        .type = QNN_TENSOR_TYPE_STATIC,
                        .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                        .dataType = QNN_DATATYPE_UINT_32,
                        .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                           QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                           {.scaleOffsetEncoding = {.scale = 0.0000000000000000f,
                                                                    .offset = 0}}},
                        .rank = 1,
                        .dimensions = dimensions_InceptionV3_InceptionV3_Conv2d_1a_3x3_Conv2D_stride,
                        .memType = QNN_TENSORMEMTYPE_RAW,
                        .clientBuf =
                            {.data = (uint8_t *)InceptionV3_InceptionV3_Conv2d_1a_3x3_Conv2D_stride,
                             .dataSize = 8}}}},
        {.paramType = QNN_PARAMTYPE_TENSOR,
         .name = "pad_amount",
         .tensorParam =
             (Qnn_Tensor_t){
                 .version = QNN_TENSOR_VERSION_1,
                 .v1 = {.id = 0,
                        .name = padName.c_str(),
                        .type = QNN_TENSOR_TYPE_STATIC,
                        .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                        .dataType = QNN_DATATYPE_UINT_32,
                        .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                           QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                           {.scaleOffsetEncoding = {.scale = 0.0000000000000000f,
                                                                    .offset = 0}}},
                        .rank = 2,
                        .dimensions =
                            dimensions_InceptionV3_InceptionV3_Conv2d_1a_3x3_Conv2D_pad_amount,
                        .memType = QNN_TENSORMEMTYPE_RAW,
                        .clientBuf =
                            {.data = (uint8_t *)
                                 InceptionV3_InceptionV3_Conv2d_1a_3x3_Conv2D_pad_amount,
                             .dataSize = 16}}}},

    };

    // add weight tensor to qnn
    uint32_t dimensionsWeight[4] = {1, 1, static_cast<uint32_t>(weight_.sequence()), static_cast<uint32_t>(weight_.dimension())};

    auto qnnQuantDefined = QNN_DEFINITION_UNDEFINED;
    float weightScale = 0;

    qnnQuantDefined = QNN_DEFINITION_DEFINED;
    weightScale = weightScale_.hostPtr<float>()[0];

    qnnBackend_->modelAddTensor(weight_.name(), (Qnn_Tensor_t){
                                                    .version = QNN_TENSOR_VERSION_1,
                                                    .v1 = {
                                                        .id = 0,
                                                        .name = weight_.name().c_str(),
                                                        .type = QNN_TENSOR_TYPE_STATIC,
                                                        .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                                        .dataType = QNN_DATATYPE_SFIXED_POINT_8,
                                                        .quantizeParams = {qnnQuantDefined,
                                                                           QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                                           {.scaleOffsetEncoding = {.scale = weightScale, .offset = 0}}},
                                                        .rank = 4,
                                                        .dimensions = dimensionsWeight,
                                                        .memType = QNN_TENSORMEMTYPE_RAW,
                                                        .clientBuf = {.data = weight_.hostPtr<void>(),
                                                                      .dataSize = (uint32_t)weight_.cntSize()}}});
    // free weight host memory
    weight_.free();

    // dimensions of matmul output and bias
    uint32_t dimensionsOutput[4] = {static_cast<uint32_t>(outputs[0]->batch()),
                                    static_cast<uint32_t>(outputs[0]->sequence()),
                                    static_cast<uint32_t>(outputs[0]->head()),
                                    static_cast<uint32_t>(outputs[0]->dimension())};

    auto outName = outputs[0]->name();

    // if don't support bias, just dequantize and write to tensor with name of outputs[0]
    if (!support_bias_) {
        float outputScale = 0;
        outputScale = outputScale_.hostPtr<float>()[0] / 127.0;
        outputScale = roundf(outputScale * 100000) / 100000;

        vector<Qnn_Tensor_t> matmulOut = {{QNN_TENSOR_VERSION_1,
                                           {.v1 = {
                                                .id = 0,
                                                .name = outName.c_str(),
                                                .type = getOutputTensorType(outputs[0]),
                                                .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                                .dataType = QNN_DATATYPE_SFIXED_POINT_8,
                                                .quantizeParams = {QNN_DEFINITION_DEFINED,
                                                                   QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                                   {.scaleOffsetEncoding = {.scale = outputScale, .offset = 0}}},
                                                .rank = 4,
                                                .dimensions = dimensionsOutput,
                                                .memType = QNN_TENSORMEMTYPE_RAW,
                                                .clientBuf = {.data = nullptr,
                                                              .dataSize = 0}}}}};
        return graphAddNode(name() + ".linearint8", "Conv2d", {inputs[0]->name(), weight_.name()}, matmulOut, params_InceptionV3_InceptionV3_Conv2d_1a_3x3_Conv2D);
    }

    // add bias tensor to qnn
    uint32_t dimensionsBias[1] = {(uint32_t)out_features_};
    float biasScale = 0;

    qnnQuantDefined = QNN_DEFINITION_DEFINED;
    biasScale = biasScale_.hostPtr<float>()[0];

    qnnBackend_->modelAddTensor(bias_.name(), (Qnn_Tensor_t){
                                                  .version = QNN_TENSOR_VERSION_1,
                                                  .v1 = {
                                                      .id = 0,
                                                      .name = bias_.name().c_str(),
                                                      .type = QNN_TENSOR_TYPE_STATIC,
                                                      .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                                      .dataType = QNN_DATATYPE_UFIXED_POINT_8,
                                                      .quantizeParams = {qnnQuantDefined,
                                                                         QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                                         {.scaleOffsetEncoding = {.scale = biasScale, .offset = -128}}},
                                                      .rank = 1,
                                                      .dimensions = dimensionsBias,
                                                      .memType = QNN_TENSORMEMTYPE_RAW,
                                                      .clientBuf = {.data = bias_.hostPtr<void>(),
                                                                    .dataSize = (uint32_t)bias_.cntSize()}}});
    // free bias host memory
    bias_.free();

    float outputScale = 0;
    outputScale = outputScale_.hostPtr<float>()[0] / 127.0;
    outputScale = roundf(outputScale * 100000) / 100000;

    // final output
    vector<Qnn_Tensor_t> biasOutput = {{QNN_TENSOR_VERSION_1,
                                        {.v1 = {
                                             .id = 0,
                                             .name = outName.c_str(),
                                             .type = getOutputTensorType(outputs[0]),
                                             .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                             .dataType = QNN_DATATYPE_SFIXED_POINT_8,
                                             .quantizeParams = {QNN_DEFINITION_DEFINED,
                                                                QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                                {.scaleOffsetEncoding = {.scale = outputScale, .offset = 0}}},
                                             .rank = 4,
                                             .dimensions = dimensionsOutput,
                                             .memType = QNN_TENSORMEMTYPE_RAW,
                                             .clientBuf = {.data = nullptr,
                                                           .dataSize = 0}}}}};
    return graphAddNode(name() + ".linearint8", "Conv2d", {inputs[0]->name(), weight_.name(), bias_.name()}, biasOutput, params_InceptionV3_InceptionV3_Conv2d_1a_3x3_Conv2D);
}

ErrorCode QNNLinearINT8::load(AbstructLoader &loader) {
    weight_.setName(name() + ".weight");
    weight_.reshape(1, 1, in_features_, out_features_);
    weight_.setDtype(MLLM_TYPE_I8);
    weight_.alloc();
    loader.load(&weight_);

    bias_.setName(name() + ".bias");
    bias_.reshape(1, 1, 1, out_features_);
    bias_.setDtype(MLLM_TYPE_I8);
    bias_.alloc();
    if (support_bias_) {
        loader.load(&bias_);
        // sign to unsign
        for (int i = 0; i < out_features_; i++) {
            int32_t val = bias_.dataAt<int8_t>(0, 0, 0, i);
            val += 128;
            bias_.setDataAt<uint8_t>(0, 0, 0, i, (uint8_t)val);
        }
    } else {
        memset(bias_.hostPtr<void>(), 0, bias_.cntSize());
    }

    weightScale_.setName(name() + ".weight.scale");
    weightScale_.reshape(1, 1, 1, 1);
    weightScale_.setDtype(MLLM_TYPE_F32);
    weightScale_.alloc();
    loader.load(&weightScale_);

    biasScale_.setName(name() + ".bias.scale");
    biasScale_.reshape(1, 1, 1, 1);
    biasScale_.setDtype(MLLM_TYPE_F32);
    biasScale_.alloc();
    loader.load(&biasScale_);

    outputScale_.setName(name() + ".output_scale");
    outputScale_.reshape(1, 1, 1, 1);
    outputScale_.setDtype(MLLM_TYPE_F32);
    outputScale_.alloc();
    loader.load(&outputScale_);

    inputScale_.setName(name() + ".input_scale");
    inputScale_.reshape(1, 1, 1, 1);
    inputScale_.setDtype(MLLM_TYPE_F32);
    inputScale_.alloc();
    loader.load(&inputScale_);

    return Op::load(loader);
}

ErrorCode QNNLinearINT8::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return Op::free(inputs, outputs);
}
} // namespace mllm
