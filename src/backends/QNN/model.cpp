//==============================================================================
//
//  Copyright (c) 2019-2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

/*
 * This file is a prototype model file that is here to mimic what the Qnn Converter will
 * generate. This file along with the WrapperApis can be used to build the qnn_model.so
 */

#include "QnnModel.hpp"
#include "QnnOpDef.h"

// Flag to determine if Backend should node validation for each opNode added
#define DO_GRAPH_NODE_VALIDATIONS 1

using namespace qnn_wrapper_api;
extern "C" {
QNN_API
ModelError_t QnnModel_composeGraphs(Qnn_BackendHandle_t backendHandle,
                                    QNN_INTERFACE_VER_TYPE interface,
                                    Qnn_ContextHandle_t contextHandle,
                                    const GraphConfigInfo_t **graphsConfigInfo,
                                    const uint32_t numGraphsConfigInfo,
                                    GraphInfoPtr_t **graphsInfo,
                                    uint32_t *numGraphsInfo,
                                    bool debug,
                                    QnnLog_Callback_t logCallback,
                                    QnnLog_Level_t maxLogLevel) {
    ModelError_t err = MODEL_NO_ERROR;

    /* model/graph for convReluModel*/
    QnnModel convReluModel;
    const QnnGraph_Config_t **graphConfigs = nullptr;
    VALIDATE(getQnnGraphConfigFromInfo(
                 "convReluModel", graphsConfigInfo, numGraphsConfigInfo, graphConfigs),
             err);
    VALIDATE(convReluModel.initialize(backendHandle,
                                      interface,
                                      contextHandle,
                                      "convReluModel",
                                      debug,
                                      DO_GRAPH_NODE_VALIDATIONS,
                                      graphConfigs),
             err);
    uint32_t dimensions_input_0[] = {1, 299, 299, 3};
    VALIDATE(convReluModel.addTensor(
                 "input_0", // Node Name
                 (Qnn_Tensor_t){
                     .version = QNN_TENSOR_VERSION_1,
                     .v1 = {.id = 0,
                            .name = "input_0",
                            .type = QNN_TENSOR_TYPE_APP_WRITE,
                            .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                            .dataType = QNN_DATATYPE_FLOAT_32,
                            .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                               {.scaleOffsetEncoding = {.scale = 0.0000000000000000f,
                                                                        .offset = 0}}},
                            .rank = 4,
                            .dimensions = dimensions_input_0,
                            .memType = QNN_TENSORMEMTYPE_RAW,
                            .clientBuf = {.data = nullptr, .dataSize = 0}}}),
             err);
    uint32_t dimensions_InceptionV3_InceptionV3_Conv2d_1a_3x3_Conv2D_weight[] = {3, 3, 3, 32};
    VALIDATE(
        convReluModel.addTensor(
            "InceptionV3_InceptionV3_Conv2d_1a_3x3_Conv2D_weight", // Node Name
            (Qnn_Tensor_t){
                .version = QNN_TENSOR_VERSION_1,
                .v1 = {.id = 0,
                       .name = "InceptionV3_InceptionV3_Conv2d_1a_3x3_Conv2D_weight",
                       .type = QNN_TENSOR_TYPE_STATIC,
                       .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                       .dataType = QNN_DATATYPE_FLOAT_32,
                       .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                          QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                          {.scaleOffsetEncoding = {.scale = 0.0000000000000000f,
                                                                   .offset = 0}}},
                       .rank = 4,
                       .dimensions = dimensions_InceptionV3_InceptionV3_Conv2d_1a_3x3_Conv2D_weight,
                       .memType = QNN_TENSORMEMTYPE_RAW,
                       .clientBuf = {.data = BINVARSTART(
                                         InceptionV3_InceptionV3_Conv2d_1a_3x3_Conv2D_weight),
                                     .dataSize = BINLEN(
                                         InceptionV3_InceptionV3_Conv2d_1a_3x3_Conv2D_weight)}}}),
        err);
    uint32_t dimensions_InceptionV3_InceptionV3_Conv2d_1a_3x3_Conv2D_bias[] = {32};
    VALIDATE(
        convReluModel.addTensor(
            "InceptionV3_InceptionV3_Conv2d_1a_3x3_Conv2D_bias", // Node Name
            (Qnn_Tensor_t){
                .version = QNN_TENSOR_VERSION_1,
                .v1 = {.id = 0,
                       .name = "InceptionV3_InceptionV3_Conv2d_1a_3x3_Conv2D_bias",
                       .type = QNN_TENSOR_TYPE_STATIC,
                       .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                       .dataType = QNN_DATATYPE_FLOAT_32,
                       .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                          QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                          {.scaleOffsetEncoding = {.scale = 0.0000000000000000f,
                                                                   .offset = 0}}},
                       .rank = 1,
                       .dimensions = dimensions_InceptionV3_InceptionV3_Conv2d_1a_3x3_Conv2D_bias,
                       .memType = QNN_TENSORMEMTYPE_RAW,
                       .clientBuf =
                           {.data = BINVARSTART(InceptionV3_InceptionV3_Conv2d_1a_3x3_Conv2D_bias),
                            .dataSize = BINLEN(InceptionV3_InceptionV3_Conv2d_1a_3x3_Conv2D_bias)}}}),
        err);

    /* ADDING NODE FOR InceptionV3_InceptionV3_Conv2d_1a_3x3_Conv2D */
    uint32_t dimensions_InceptionV3_InceptionV3_Conv2d_1a_3x3_Conv2D_dilation[] = {2};
    uint32_t InceptionV3_InceptionV3_Conv2d_1a_3x3_Conv2D_dilation[] = {1, 1};
    uint32_t dimensions_InceptionV3_InceptionV3_Conv2d_1a_3x3_Conv2D_pad_amount[] = {2, 2};
    uint32_t InceptionV3_InceptionV3_Conv2d_1a_3x3_Conv2D_pad_amount[] = {0, 0, 0, 0};
    uint32_t dimensions_InceptionV3_InceptionV3_Conv2d_1a_3x3_Conv2D_stride[] = {2};
    uint32_t InceptionV3_InceptionV3_Conv2d_1a_3x3_Conv2D_stride[] = {2, 2};
    Qnn_Param_t params_InceptionV3_InceptionV3_Conv2d_1a_3x3_Conv2D[] = {
        {.paramType = QNN_PARAMTYPE_TENSOR,
         .name = "dilation",
         .tensorParam =
             (Qnn_Tensor_t){
                 .version = QNN_TENSOR_VERSION_1,
                 .v1 = {.id = 0,
                        .name = "InceptionV3_InceptionV3_Conv2d_1a_3x3_Conv2D_dilation",
                        .type = QNN_TENSOR_TYPE_STATIC,
                        .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                        .dataType = QNN_DATATYPE_UINT_32,
                        .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                           QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                           {.scaleOffsetEncoding = {.scale = 0.0000000000000000f,
                                                                    .offset = 0}}},
                        .rank = 1,
                        .dimensions =
                            dimensions_InceptionV3_InceptionV3_Conv2d_1a_3x3_Conv2D_dilation,
                        .memType = QNN_TENSORMEMTYPE_RAW,
                        .clientBuf = {.data = (uint8_t *)
                                          InceptionV3_InceptionV3_Conv2d_1a_3x3_Conv2D_dilation,
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
        {.paramType = QNN_PARAMTYPE_SCALAR,
         .name = "group",
         .scalarParam = {.dataType = QNN_DATATYPE_UINT_32, .uint32Value = 1}}};
    const char *inputs_InceptionV3_InceptionV3_Conv2d_1a_3x3_Conv2D[] = {
        "input_0",
        "InceptionV3_InceptionV3_Conv2d_1a_3x3_Conv2D_weight",
        "InceptionV3_InceptionV3_Conv2d_1a_3x3_Conv2D_bias"};
    uint32_t dimensions_InceptionV3_InceptionV3_Conv2d_1a_3x3_BatchNorm_FusedBatchNorm_0[] = {
        1, 149, 149, 32};
    Qnn_Tensor_t outputs_InceptionV3_InceptionV3_Conv2d_1a_3x3_Conv2D[] = {(Qnn_Tensor_t){
        .version = QNN_TENSOR_VERSION_1,
        .v1 = {
            .id = 0,
            .name = "InceptionV3_InceptionV3_Conv2d_1a_3x3_BatchNorm_FusedBatchNorm_0",
            .type = QNN_TENSOR_TYPE_NATIVE,
            .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType = QNN_DATATYPE_FLOAT_32,
            .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
            .rank = 4,
            .dimensions = dimensions_InceptionV3_InceptionV3_Conv2d_1a_3x3_BatchNorm_FusedBatchNorm_0,
            .memType = QNN_TENSORMEMTYPE_RAW,
            .clientBuf = {.data = nullptr, .dataSize = 0}}}};
    VALIDATE(convReluModel.addNode(
                 QNN_OPCONFIG_VERSION_1,                               // Op_Config_t Version
                 "InceptionV3_InceptionV3_Conv2d_1a_3x3_Conv2D",       // Node Name
                 "qti.aisw",                                           // Package Name
                 "Conv2d",                                             // Qnn Node Type
                 params_InceptionV3_InceptionV3_Conv2d_1a_3x3_Conv2D,  // Node Params
                 4,                                                    // Num Node Params
                 inputs_InceptionV3_InceptionV3_Conv2d_1a_3x3_Conv2D,  // Input Tensor Names
                 3,                                                    // Num Input Tensor Names
                 outputs_InceptionV3_InceptionV3_Conv2d_1a_3x3_Conv2D, // Output Tensors
                 1                                                     // Num Output Tensors
                 ),
             err);

    /* ADDING NODE FOR InceptionV3_InceptionV3_Conv2d_1a_3x3_Relu */
    const char *inputs_InceptionV3_InceptionV3_Conv2d_1a_3x3_Relu[] = {
        "InceptionV3_InceptionV3_Conv2d_1a_3x3_BatchNorm_FusedBatchNorm_0"};
    uint32_t dimensions_InceptionV3_InceptionV3_Conv2d_1a_3x3_Relu_0[] = {1, 149, 149, 32};
    Qnn_Tensor_t outputs_InceptionV3_InceptionV3_Conv2d_1a_3x3_Relu[] = {(Qnn_Tensor_t){
        .version = QNN_TENSOR_VERSION_1,
        .v1 = {
            .id = 0,
            .name = "InceptionV3_InceptionV3_Conv2d_1a_3x3_Relu_0",
            .type = QNN_TENSOR_TYPE_APP_READ,
            .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType = QNN_DATATYPE_FLOAT_32,
            .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
            .rank = 4,
            .dimensions = dimensions_InceptionV3_InceptionV3_Conv2d_1a_3x3_Relu_0,
            .memType = QNN_TENSORMEMTYPE_RAW,
            .clientBuf = {.data = nullptr, .dataSize = 0}}}};
    VALIDATE(convReluModel.addNode(
                 QNN_OPCONFIG_VERSION_1,                             // Op_Config_t Version
                 "InceptionV3_InceptionV3_Conv2d_1a_3x3_Relu",       // Node Name
                 "qti.aisw",                                         // Package Name
                 "Relu",                                             // Qnn Node Type
                 nullptr,                                            // Node Params
                 0,                                                  // Num Node Params
                 inputs_InceptionV3_InceptionV3_Conv2d_1a_3x3_Relu,  // Input Tensor Names
                 1,                                                  // Num Input Tensor Names
                 outputs_InceptionV3_InceptionV3_Conv2d_1a_3x3_Relu, // Output Tensors
                 1                                                   // Num Output Tensors
                 ),
             err);

    // Add all models to array to get graphsInfo
    QnnModel *models[] = {&convReluModel};
    uint32_t numModels = 1;

    // Populate the constructed graphs in provided output variables
    VALIDATE(getGraphInfoFromModels(*models, numModels, graphsInfo), err);
    *numGraphsInfo = numModels;

    return err;

} // PREPARE_GRAPHS

QNN_API
ModelError_t QnnModel_freeGraphsInfo(GraphInfoPtr_t **graphs, uint32_t numGraphsInfo) {
    return qnn_wrapper_api::freeGraphsInfo(graphs, numGraphsInfo);
} // FREEGRAPHINFO
}
