
#include "QNNLayerNorm.hpp"
#include "QnnTypes.h"
#include "Types.hpp"
#include "QNNCommonOp.hpp"
#include <cstdint>

namespace mllm {
QNNLayerNorm::QNNLayerNorm(Backend *bn, string opName, int normSize, bool bias, float epsilon) :
    QNNCommonOp(bn, opName), normSize_(normSize), bias(bias), epsilon_(epsilon) {
    weight_.setBackend(bn);
    bias_.setBackend(bn);
}

ErrorCode QNNLayerNorm::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    normSize_ = inputs[0]->dimension();
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    return Op::reshape(inputs, outputs);
}

ErrorCode QNNLayerNorm::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    uint32_t axesDim[1] = {1};
    uint32_t axes[1] = {3};
    auto axesName = name() + ".axes";
    vector<Qnn_Param_t> params = {
        {.paramType = QNN_PARAMTYPE_SCALAR,
         .name = "epsilon",
         .scalarParam = (Qnn_Scalar_t){QNN_DATATYPE_FLOAT_32, {.floatValue = epsilon_}}},
        {.paramType = QNN_PARAMTYPE_TENSOR,
         .name = "axes",
         .tensorParam = (Qnn_Tensor_t){
             .version = QNN_TENSOR_VERSION_1,
             .v1 = {
                 .id = 0,
                 .name = axesName.c_str(),
                 .type = QNN_TENSOR_TYPE_STATIC,
                 .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                 .dataType = QNN_DATATYPE_UINT_32,
                 .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                    {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                 .rank = 1,
                 .dimensions = axesDim,
                 .memType = QNN_TENSORMEMTYPE_RAW,
                 .clientBuf = {.data = axes,
                               .dataSize = 4}}}}};

    uint32_t dimWeight[1] = {(uint32_t)normSize_};
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
                                                        .rank = 1,
                                                        .dimensions = dimWeight,
                                                        .memType = QNN_TENSORMEMTYPE_RAW,
                                                        .clientBuf = {.data = weight_.hostPtr<void>(),
                                                                      .dataSize = static_cast<uint32_t>(weight_.cntSize())}}});
    if (bias) {
        uint32_t dimBias[1] = {(uint32_t)normSize_};
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
                                                          .rank = 1,
                                                          .dimensions = dimBias,
                                                          .memType = QNN_TENSORMEMTYPE_RAW,
                                                          .clientBuf = {.data = bias_.hostPtr<void>(),
                                                                        .dataSize = static_cast<uint32_t>(bias_.cntSize())}}});
    }

    uint32_t dimOut[] = {(uint32_t)outputs[0]->batch(), (uint32_t)outputs[0]->sequence(), (uint32_t)outputs[0]->head(), (uint32_t)outputs[0]->dimension()};
    auto outName = outputs[0]->name();
    vector<Qnn_Tensor_t>
        out = {
            (Qnn_Tensor_t){
                .version = QNN_TENSOR_VERSION_1,
                .v1 = {
                    .id = 0,
                    .name = outName.c_str(),
                    .type = getOutputTensorType(outputs[0]),
                    .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                    .dataType = QNN_DATATYPE_FLOAT_32,
                    .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                       QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                       {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                    .rank = 4,
                    .dimensions = dimOut,
                    .memType = QNN_TENSORMEMTYPE_RAW,
                    .clientBuf = {.data = nullptr,
                                  .dataSize = 0}}}};
    if (bias) {
        return graphAddNode(name(), "LayerNorm", {inputs[0]->name(), weight_.name(), bias_.name()}, out, params);
    } else {
        return graphAddNode(name(), "LayerNorm", {inputs[0]->name(), weight_.name()}, out, params);
    }
}

// ErrorCode QNNLayerNorm::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {

//     uint32_t dimWeight[1] = {(uint32_t)normSize_};
//     qnnBackend_->modelAddTensor(weight_.name(), (Qnn_Tensor_t){
//                                                     .version = QNN_TENSOR_VERSION_1,
//                                                     {.v1 = {
//                                                          .id = 0,
//                                                          .name = weight_.name().c_str(),
//                                                          .type = QNN_TENSOR_TYPE_STATIC,
//                                                          .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
//                                                          .dataType = QNN_DATATYPE_FLOAT_32,
//                                                          .quantizeParams = {QNN_DEFINITION_UNDEFINED,
//                                                                             QNN_QUANTIZATION_ENCODING_UNDEFINED,
//                                                                             {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
//                                                          .rank = 1,
//                                                          .dimensions = dimWeight,
//                                                          .memType = QNN_TENSORMEMTYPE_RAW,
//                                                          {.clientBuf = {.data = weight_.hostPtr<void>(),
//                                                                         .dataSize = static_cast<uint32_t>(weight_.cntSize())}}}}});

//     uint32_t dimBias[1] = {(uint32_t)normSize_};
//     qnnBackend_->modelAddTensor(bias_.name(), (Qnn_Tensor_t){
//                                                     .version = QNN_TENSOR_VERSION_1,
//                                                     {.v1 = {
//                                                         .id = 0,
//                                                         .name = bias_.name().c_str(),
//                                                         .type = QNN_TENSOR_TYPE_STATIC,
//                                                         .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
//                                                         .dataType = QNN_DATATYPE_FLOAT_32,
//                                                         .quantizeParams = {QNN_DEFINITION_UNDEFINED,
//                                                                             QNN_QUANTIZATION_ENCODING_UNDEFINED,
//                                                                             {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
//                                                         .rank = 1,
//                                                         .dimensions = dimBias,
//                                                         .memType = QNN_TENSORMEMTYPE_RAW,
//                                                         {.clientBuf = {.data = bias_.hostPtr<void>(),
//                                                                         .dataSize = static_cast<uint32_t>(bias_.cntSize())}}}}});

//     uint32_t dimOut[] = {(uint32_t)outputs[0]->batch(), (uint32_t)outputs[0]->sequence(), (uint32_t)outputs[0]->head(), (uint32_t)outputs[0]->dimension()};
//     auto outName = outputs[0]->name();
//     vector<Qnn_Tensor_t>
//         out = {
//             (Qnn_Tensor_t){
//                 .version = QNN_TENSOR_VERSION_1,
//                 {.v1 = {
//                      .id = 0,
//                      .name = outName.c_str(),
//                      .type = getOutputTensorType(outputs[0]),
//                      .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
//                      .dataType = QNN_DATATYPE_FLOAT_32,
//                      .quantizeParams = {QNN_DEFINITION_UNDEFINED,
//                                         QNN_QUANTIZATION_ENCODING_UNDEFINED,
//                                         {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
//                      .rank = 4,
//                      .dimensions = dimOut,
//                      .memType = QNN_TENSORMEMTYPE_RAW,
//                      {.clientBuf = {.data = nullptr,
//                                     .dataSize = 0}}}}}};

//     return graphAddNode(name(), "QLayerNorm", {inputs[0]->name(), weight_.name(), bias_.name()}, out, {}, "LLaMAPackage");
// }

ErrorCode QNNLayerNorm::load(AbstructLoader &loader) {
    weight_.setName(name() + ".weight");
    weight_.reshape(1, 1, 1, normSize_);
    weight_.setDtype(MLLM_TYPE_F32);
    weight_.alloc();
    loader.load(&weight_);

    if (bias) {
        bias_.setName(name() + ".bias");
        bias_.reshape(1, 1, 1, normSize_);
        bias_.setDtype(MLLM_TYPE_F32);
        bias_.alloc();
        loader.load(&bias_);
    }
    return Op::load(loader);
}
} // namespace mllm
