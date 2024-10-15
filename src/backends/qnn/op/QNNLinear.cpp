
#include "QNNLinear.hpp"
#include "QnnTypes.h"
#include "Types.hpp"
#include "QNNCommonOp.hpp"
#include <cstdint>
#include <memory>

namespace mllm {
QNNLinear::QNNLinear(Backend *bn, string opName, int in_features, int out_features, bool bias) :
    QNNCommonOp(bn, opName), in_features_(in_features), out_features_(out_features), support_bias_(bias) {
    weight_.setBackend(bn);
    bias_.setBackend(bn);
}

ErrorCode QNNLinear::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
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

ErrorCode QNNLinear::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // add matmul param to qnn
    vector<Qnn_Param_t> paramsMatmul = {
        {.paramType = QNN_PARAMTYPE_SCALAR,
         .name = "transpose_in0",
         .scalarParam = (Qnn_Scalar_t){QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}},
        {.paramType = QNN_PARAMTYPE_SCALAR,
         .name = "transpose_in1",
         .scalarParam = (Qnn_Scalar_t){QNN_DATATYPE_BOOL_8, {.bool8Value = 1}}}};
    // add quantized input tensor to qnn
    auto inputQuantizeName = name() + inputs[0]->name() + ".quantize";
    uint32_t dimensionsInput[4] = {static_cast<uint32_t>(inputs[0]->batch()),
                                   static_cast<uint32_t>(inputs[0]->sequence()),
                                   static_cast<uint32_t>(inputs[0]->head()),
                                   static_cast<uint32_t>(inputs[0]->dimension())};

    uint32_t dimensions_InceptionV3_InceptionV3_Conv2d_1a_3x3_Conv2D_dilation[] = {2};
    uint32_t InceptionV3_InceptionV3_Conv2d_1a_3x3_Conv2D_dilation[] = {1, 1};
    uint32_t dimensions_InceptionV3_InceptionV3_Conv2d_1a_3x3_Conv2D_pad_amount[] = {2, 2};
    uint32_t InceptionV3_InceptionV3_Conv2d_1a_3x3_Conv2D_pad_amount[] = {0, 0, 0, 0};
    uint32_t dimensions_InceptionV3_InceptionV3_Conv2d_1a_3x3_Conv2D_stride[] = {2};
    uint32_t InceptionV3_InceptionV3_Conv2d_1a_3x3_Conv2D_stride[] = {1, 1};

    vector<Qnn_Param_t> params_InceptionV3_InceptionV3_Conv2d_1a_3x3_Conv2D = {
        {.paramType = QNN_PARAMTYPE_TENSOR,
         .name = "stride",
         .tensorParam =
             (Qnn_Tensor_t){
                 .version = QNN_TENSOR_VERSION_1,
                 .v1 = {.id = 0,
                        .name = "InceptionV3_InceptionV3_Conv2d_1a_3x3_Conv2D_stride",
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
                        .name = "InceptionV3_InceptionV3_Conv2d_1a_3x3_Conv2D_pad_amount",
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

    // TODO： split into another function
    // if weight is float32, use float matmul
    if (weight_.dtype() == MLLM_TYPE_F32) {
        std::cout << " test fp linear " << name() << std::endl;

        uint32_t dimensionsWeight[4] = {1, 1, static_cast<uint32_t>(weight_.sequence()), static_cast<uint32_t>(weight_.dimension())};
        qnnBackend_->modelAddTensor(weight_.name(), (Qnn_Tensor_t){
                                                        .version = QNN_TENSOR_VERSION_1,
                                                        .v1 = {
                                                            .id = 0,
                                                            .name = weight_.name().c_str(),
                                                            .type = QNN_TENSOR_TYPE_STATIC,
                                                            .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                                            .dataType = QNN_DATATYPE_FLOAT_32,
                                                            .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                                               {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                                                            .rank = 4,
                                                            .dimensions = dimensionsWeight,
                                                            .memType = QNN_TENSORMEMTYPE_RAW,
                                                            .clientBuf = {.data = weight_.hostPtr<void>(),
                                                                          .dataSize = (uint32_t)weight_.cntSize()}}});
        // final output
        uint32_t dimensionsOutput[4] = {static_cast<uint32_t>(outputs[0]->batch()),
                                        static_cast<uint32_t>(outputs[0]->sequence()),
                                        static_cast<uint32_t>(outputs[0]->head()),
                                        static_cast<uint32_t>(outputs[0]->dimension())};
        auto outString = outputs[0]->name();
        vector<Qnn_Tensor_t>
            matmulOut = {{QNN_TENSOR_VERSION_1,
                          {.v1 = {
                               .id = 0,
                               .name = outString.c_str(),
                               .type = getOutputTensorType(outputs[0]),
                               .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                               .dataType = QNN_DATATYPE_FLOAT_32,
                               .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                  QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                  {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                               .rank = 4,
                               .dimensions = dimensionsOutput,
                               .memType = QNN_TENSORMEMTYPE_RAW,
                               .clientBuf = {.data = nullptr,
                                             .dataSize = 0}}}}};

        return graphAddNode(name() + ".matmul", "Conv2d", {inputs[0]->name(), weight_.name()}, matmulOut, params_InceptionV3_InceptionV3_Conv2d_1a_3x3_Conv2D);
    } // TODO： split into another function

    vector<Qnn_Tensor_t> quantizedInput = {
        (Qnn_Tensor_t){
            .version = QNN_TENSOR_VERSION_1,
            .v1 = {
                .id = 0,
                .name = inputQuantizeName.c_str(),
                .type = QNN_TENSOR_TYPE_NATIVE,
                .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                .dataType = QNN_DATATYPE_UFIXED_POINT_8,
                .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                   QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                   {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                .rank = 4,
                .dimensions = dimensionsInput,
                .memType = QNN_TENSORMEMTYPE_RAW,
                .clientBuf = {.data = nullptr,
                              .dataSize = 0}}}};
    graphAddNode(name() + ".quantize", "Quantize", {inputs[0]->name()}, quantizedInput);
    // add weight tensor to qnn
    uint32_t dimensionsWeight[4] = {1, 1, static_cast<uint32_t>(weight_.sequence()), static_cast<uint32_t>(weight_.dimension())};
    qnnBackend_->modelAddTensor(weight_.name(), (Qnn_Tensor_t){
                                                    .version = QNN_TENSOR_VERSION_1,
                                                    .v1 = {
                                                        .id = 0,
                                                        .name = weight_.name().c_str(),
                                                        .type = QNN_TENSOR_TYPE_STATIC,
                                                        .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                                        .dataType = QNN_DATATYPE_UFIXED_POINT_8,
                                                        .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                                           QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                                           {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                                                        .rank = 4,
                                                        .dimensions = dimensionsWeight,
                                                        .memType = QNN_TENSORMEMTYPE_RAW,
                                                        .clientBuf = {.data = weight_.hostPtr<void>(),
                                                                      .dataSize = (uint32_t)weight_.cntSize()}}});
    // free weight host memory
    // weight_.free();

    // dimensions of matmul output and bias
    uint32_t dimensionsOutput[4] = {static_cast<uint32_t>(outputs[0]->batch()),
                                    static_cast<uint32_t>(outputs[0]->sequence()),
                                    static_cast<uint32_t>(outputs[0]->head()),
                                    static_cast<uint32_t>(outputs[0]->dimension())};

    auto outName = outputs[0]->name();
    auto outQuantizedName = name() + outputs[0]->name() + ".quantized";
    auto outDeqnName = name() + outputs[0]->name() + ".dequantized";
    vector<Qnn_Tensor_t> matmulOut = {{QNN_TENSOR_VERSION_1,
                                       {.v1 = {
                                            .id = 0,
                                            .name = outQuantizedName.c_str(),
                                            .type = QNN_TENSOR_TYPE_NATIVE,
                                            .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                            .dataType = QNN_DATATYPE_UFIXED_POINT_8,
                                            .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                               {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                                            .rank = 4,
                                            .dimensions = dimensionsOutput,
                                            .memType = QNN_TENSORMEMTYPE_RAW,
                                            .clientBuf = {.data = nullptr,
                                                          .dataSize = 0}}}}};
    graphAddNode(name() + ".matmul", "MatMul", {inputQuantizeName, weight_.name()}, matmulOut, paramsMatmul);

    // if don't support bias, just dequantize and write to tensor with name of outputs[0]
    if (!support_bias_) {
        // output of dequantized result of matmul
        vector<Qnn_Tensor_t> deqnOut = {{QNN_TENSOR_VERSION_1,
                                         {.v1 = {
                                              .id = 0,
                                              .name = outName.c_str(),
                                              .type = getOutputTensorType(outputs[0]),
                                              .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                              .dataType = QNN_DATATYPE_FLOAT_32,
                                              .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                                 QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                                 {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                                              .rank = 4,
                                              .dimensions = dimensionsOutput,
                                              .memType = QNN_TENSORMEMTYPE_RAW,
                                              .clientBuf = {.data = nullptr,
                                                            .dataSize = 0}}}}};
        return graphAddNode(name() + ".dequantize", "Dequantize", {outQuantizedName}, deqnOut);
    }

    // dequantize to tensor with name of outputs[0] + ".dequantize"
    // output of dequantized result of matmul
    vector<Qnn_Tensor_t> deqnOut = {{QNN_TENSOR_VERSION_1,
                                     {.v1 = {
                                          .id = 0,
                                          .name = outDeqnName.c_str(),
                                          .type = QNN_TENSOR_TYPE_NATIVE,
                                          .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType = QNN_DATATYPE_FLOAT_32,
                                          .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                             QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                             {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                                          .rank = 4,
                                          .dimensions = dimensionsOutput,
                                          .memType = QNN_TENSORMEMTYPE_RAW,
                                          .clientBuf = {.data = nullptr,
                                                        .dataSize = 0}}}}};
    graphAddNode(name() + ".dequantize", "Dequantize", {outQuantizedName}, deqnOut);
    // add bias tensor to qnn
    uint32_t dimensionsBias[4] = {1, 1, 1, (uint32_t)out_features_};
    qnnBackend_->modelAddTensor(bias_.name(), (Qnn_Tensor_t){
                                                  .version = QNN_TENSOR_VERSION_1,
                                                  .v1 = {
                                                      .id = 0,
                                                      .name = bias_.name().c_str(),
                                                      .type = QNN_TENSOR_TYPE_STATIC,
                                                      .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                                      .dataType = QNN_DATATYPE_FLOAT_32,
                                                      .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                                         QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                                         {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                                                      .rank = 4,
                                                      .dimensions = dimensionsBias,
                                                      .memType = QNN_TENSORMEMTYPE_RAW,
                                                      .clientBuf = {.data = bias_.hostPtr<void>(),
                                                                    .dataSize = (uint32_t)bias_.cntSize()}}});
    // free bias host memory
    bias_.free();

    // final output
    vector<Qnn_Tensor_t> biasOutput = {{QNN_TENSOR_VERSION_1,
                                        {.v1 = {
                                             .id = 0,
                                             .name = outName.c_str(),
                                             .type = getOutputTensorType(outputs[0]),
                                             .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                             .dataType = QNN_DATATYPE_FLOAT_32,
                                             .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                                QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                                {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                                             .rank = 4,
                                             .dimensions = dimensionsOutput,
                                             .memType = QNN_TENSORMEMTYPE_RAW,
                                             .clientBuf = {.data = nullptr,
                                                           .dataSize = 0}}}}};
    return graphAddNode(name() + ".add", "ElementWiseAdd", {outDeqnName, bias_.name()}, biasOutput);
}

ErrorCode QNNLinear::load(AbstructLoader &loader) {
    weight_.setName(name() + ".weight");
    weight_.reshape(1, 1, in_features_, out_features_);
    if (loader.getDataType(weight_.name()) != MLLM_TYPE_COUNT) {
        weight_.setDtype(loader.getDataType(weight_.name()));
        weight_.alloc();
        loader.load(&weight_);
    } else {
        weight_.setDtype(MLLM_TYPE_F32);
        weight_.alloc();
    }
    if (support_bias_) {
        bias_.setName(name() + ".bias");
        bias_.reshape(1, 1, 1, out_features_);
        if (loader.getDataType(bias_.name()) != MLLM_TYPE_COUNT) {
            bias_.setDtype(loader.getDataType(bias_.name()));
            bias_.alloc();
            loader.load(&bias_);
        } else {
            bias_.setDtype(MLLM_TYPE_F32);
            bias_.alloc();
        }
    }
    return Op::load(loader);
}

ErrorCode QNNLinear::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // weight_.free();
    // if (support_bias_) {
    //     bias_.free();
    // }
    return Op::free(inputs, outputs);
}
} // namespace mllm
