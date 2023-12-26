
#include "QNNLinear.hpp"
#include "Types.hpp"
#include "QNNCommonOp.hpp"

namespace mllm {
QNNLinear::QNNLinear(Backend *bn, string opName, int in_features, int out_features, bool bias) :
    QNNCommonOp(bn, opName) {
}

ErrorCode QNNLinear::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    CHECK_EQ(inputs.size(), 1);
    CHECK_EQ(outputs.size(), 1);
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
    CHECK_EQ(inputs[0]->head(), 1);
    CHECK_EQ(in_features_, inputs[0]->dimension());
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), out_features_);
    return Op::reshape(inputs, outputs);
}

ErrorCode QNNLinear::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // linear.addNode(QNN_OPCONFIG_VERSION_1,               // Op_Config_t Version
    //                "linear_broadcast_to_0_0_nchw",       // Node Name
    //                "qti.aisw",                           // Package Name
    //                "Transpose",                          // Qnn Node Type
    //                params_linear_broadcast_to_0_0_nchw,  // Node Params
    //                1,                                    // Num Node Params
    //                inputs_linear_broadcast_to_0_0_nchw,  // Input Tensor Names
    //                1,                                    // Num Input Tensor Names
    //                outputs_linear_broadcast_to_0_0_nchw, // Output Tensors
    //                1                                     // Num Output Tensors
    //                )
    graphAddNode(name(), "MatMul", inputs, outputs);
    // linear.addNode(QNN_OPCONFIG_VERSION_1,   // Op_Config_t Version
    //                "linear_reshape_0",       // Node Name
    //                "qti.aisw",               // Package Name
    //                "Reshape",                // Qnn Node Type
    //                nullptr,                  // Node Params
    //                0,                        // Num Node Params
    //                inputs_linear_reshape_0,  // Input Tensor Names
    //                1,                        // Num Input Tensor Names
    //                outputs_linear_reshape_0, // Output Tensors
    //                1                         // Num Output Tensors
    //                )
    graphAddNode(name(), "MatMul", inputs, outputs);
    std::vector<Qnn_Param_t>
        paramsMatmul = {
            {.paramType = QNN_PARAMTYPE_SCALAR,
             .name = "transpose_in0",
             {.scalarParam = (Qnn_Scalar_t){QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}},
            {.paramType = QNN_PARAMTYPE_SCALAR,
             .name = "transpose_in1",
             {.scalarParam = (Qnn_Scalar_t){QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}}};
    graphAddNode(name(), "MatMul", inputs, outputs);
    // linear.addNode(QNN_OPCONFIG_VERSION_1,   // Op_Config_t Version
    //                "linear_reshape_1",       // Node Name
    //                "qti.aisw",               // Package Name
    //                "Reshape",                // Qnn Node Type
    //                nullptr,                  // Node Params
    //                0,                        // Num Node Params
    //                inputs_linear_reshape_1,  // Input Tensor Names
    //                1,                        // Num Input Tensor Names
    //                outputs_linear_reshape_1, // Output Tensors
    //                1                         // Num Output Tensors
    // )
    graphAddNode(name(), "MatMul", inputs, outputs);
    // linear.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
    //                "linear_add_0",         // Node Name
    //                "qti.aisw",             // Package Name
    //                "ElementWiseAdd",       // Qnn Node Type
    //                nullptr,                // Node Params
    //                0,                      // Num Node Params
    //                inputs_linear_add_0,    // Input Tensor Names
    //                2,                      // Num Input Tensor Names
    //                outputs_linear_add_0,   // Output Tensors
    //                1                       // Num Output Tensors
    // )
    return graphAddNode(name(), "MatMul", inputs, outputs);
}

ErrorCode QNNLinear::load(AbstructLoader &loader) {
    // std::cout << name() << "  CPULinear load" << std::endl;
    weight_.setName(name() + ".weight");
    weight_.reshape(1, 1, out_features_, in_features_);
    weight_.setDtype(loader.getDataType(weight_.name()));
    weight_.alloc();
    loader.load(&weight_);
    qnnBackend_->modelAddTensor(weight_.name().c_str(), (Qnn_Tensor_t){
                                                            .version = QNN_TENSOR_VERSION_1,
                                                            {.v1 = {
                                                                 .id = 0,
                                                                 .name = weight_.name().c_str(),
                                                                 .type = QNN_TENSOR_TYPE_STATIC,
                                                                 .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                                                 .dataType = QNN_DATATYPE_FLOAT_32,
                                                                 .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                                                    {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                                                                 .rank = 3,
                                                                 .dimensions = {},
                                                                 .memType = QNN_TENSORMEMTYPE_RAW,
                                                                 {.clientBuf = {.data = weight_.hostPtr<void>(),
                                                                                .dataSize = (uint32_t)weight_.cntSize()}}}}});
    if (support_bias_) {
        bias_.setName(name() + ".bias");
        bias_.reshape(1, 1, 1, out_features_);
        bias_.setDtype(loader.getDataType(bias_.name()));
        bias_.alloc();
        loader.load(&bias_);
        qnnBackend_->modelAddTensor(bias_.name().c_str(), (Qnn_Tensor_t){
                                                              .version = QNN_TENSOR_VERSION_1,
                                                              {.v1 = {
                                                                   .id = 0,
                                                                   .name = bias_.name().c_str(),
                                                                   .type = QNN_TENSOR_TYPE_STATIC,
                                                                   .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                                                   .dataType = QNN_DATATYPE_FLOAT_32,
                                                                   .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                                                      QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                                                      {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                                                                   .rank = 3,
                                                                   .dimensions = {},
                                                                   .memType = QNN_TENSORMEMTYPE_RAW,
                                                                   {.clientBuf = {.data = bias_.hostPtr<void>(),
                                                                                  .dataSize = (uint32_t)bias_.cntSize()}}}}});
    }
    return Op::load(loader);
}
} // namespace mllm
