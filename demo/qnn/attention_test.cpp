#include <iostream>
#include <valarray>
#include <csignal>
#include "Net.hpp"
#include "Executor.hpp"
#include "NetParameter.hpp"
#include "QnnTypes.h"
#include "express/Express.hpp"
#include "tokenizers/BPE/Bpe.hpp"
#include "backends/QNN/QNNBackend.hpp"
#include "memory/SystemMemoryManager.hpp"
#include "backends/QNN/op/QNNAdd.hpp"

using namespace mllm;

void sinusoidal_position_embedding(int batch_size, int nums_head, int seq_len, int output_dim,  void* sin, void* cos) {
    
    for (int n = 0; n < batch_size; ++n) {
        for (int h = 0; h < nums_head; ++h) {
            for (int s = 0; s < seq_len; ++s) {
                for (int d = 0; d < output_dim; d += 2) {
                    int i = (int)d / 2;
                    float sin_value = std::sin(s / std::pow(10000, 2.0 * i / output_dim));
                    float cos_value = std::cos(s / std::pow(10000, 2.0 * i / output_dim));
                    
                    ((__fp16*)sin)[s*output_dim + d] = sin_value;
                    ((__fp16*)cos)[s*output_dim + d] = cos_value;

                    if (d + 1 < output_dim) {
                        ((__fp16*)sin)[s*output_dim + d + 1] = sin_value;
                        ((__fp16*)cos)[s*output_dim + d + 1] = cos_value;
                    }
                }
            }
        }
    }
}




void testLLaMAAttention(QNNBackend *qbn, uint32_t layer_num, uint32_t sequence_length) {

    uint32_t dimension = 4096;
    uint32_t head = 32;
    float* input_data = new float[sequence_length * dimension];
    uint32_t input_dimensions[] = {1, 1, sequence_length, dimension};

    std::string layer = "layer0";

    // fp32_X
    qbn->modelAddTensor("fp32_x", // Node Name
                        (Qnn_Tensor_t){
                            .version = QNN_TENSOR_VERSION_1,
                            {.v1 = {
                                 .id = 0,
                                 .name = "fp32_x",
                                 .type = QNN_TENSOR_TYPE_APP_WRITE,
                                 .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                 .dataType = QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding = {.scale = 1.0000000000000000f, .offset = 0}}},
                                 .rank = 4,
                                 .dimensions = input_dimensions,
                                 .memType = QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf = {.data = nullptr,
                                                .dataSize = 0}}}}});

    // quantize
    uint32_t quantize_x_dimensions[] = {1, 1, sequence_length, dimension};
    std::string quantize_x_outputs_name = (layer + "x");
    vector<Qnn_Tensor_t> quantize_x_outputs = {
        (Qnn_Tensor_t){
            .version = QNN_TENSOR_VERSION_1,
            {.v1 = {
                 .id = 0,
                 .name = quantize_x_outputs_name.c_str(),
                 .type = QNN_TENSOR_TYPE_NATIVE,
                 .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                 .dataType = QNN_DATATYPE_UFIXED_POINT_8,
                 .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                    {.scaleOffsetEncoding = {.scale = 1.0000000000000000f, .offset = 0}}},
                 .rank = 4,
                 .dimensions = quantize_x_dimensions,
                 .memType = QNN_TENSORMEMTYPE_RAW,
                 {.clientBuf = {.data = nullptr,
                                .dataSize = 0}}}}}};
    qbn->graphAddNode(layer + "x_quantize", "Quantize", {"fp32_x"}, quantize_x_outputs, {}, "qti.aisw");

    // QKV matmul
    // W_Q
    
    uint32_t weight_Q_dimensions[] = {1, 1, dimension, dimension};
    uint8_t* weight_Q = new uint8_t[dimension * dimension]; 
    qbn->modelAddTensor((layer + "W_Q").c_str(), // Node Name
                        (Qnn_Tensor_t){
                            .version = QNN_TENSOR_VERSION_1,
                            {.v1 = {
                                 .id = 0,
                                 .name = (layer + "W_Q").c_str(),
                                 .type = QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                 .dataType = QNN_DATATYPE_UFIXED_POINT_8,
                                 .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding = {.scale = 1.0000000000000000f, .offset = -128}}},
                                 .rank = 4,
                                 .dimensions = weight_Q_dimensions,
                                 .memType = QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf = {.data = weight_Q,
                                                .dataSize = dimension * dimension * 1}}}}});


    // W_K
    uint32_t weight_K_dimensions[] = {1, 1, dimension, dimension};
    uint8_t* weight_K = new uint8_t[dimension * dimension]; 
    qbn->modelAddTensor((layer + "W_K").c_str(), // Node Name
                        (Qnn_Tensor_t){
                            .version = QNN_TENSOR_VERSION_1,
                            {.v1 = {
                                 .id = 0,
                                 .name = (layer + "W_K").c_str(),
                                 .type = QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                 .dataType = QNN_DATATYPE_UFIXED_POINT_8,
                                 .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding = {.scale = 1.0000000000000000f, .offset = -128}}},
                                 .rank = 4,
                                 .dimensions = weight_K_dimensions,
                                 .memType = QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf = {.data = weight_K,
                                                .dataSize = dimension * dimension * 1}}}}});

    // W_V
    uint32_t weight_V_dimensions[] = {1, 1, dimension, dimension};
    uint8_t* weight_V = new uint8_t[dimension * dimension]; 
    qbn->modelAddTensor((layer + "W_V").c_str(), // Node Name
                        (Qnn_Tensor_t){
                            .version = QNN_TENSOR_VERSION_1,
                            {.v1 = {
                                 .id = 0,
                                 .name = (layer + "W_V").c_str(),
                                 .type = QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                 .dataType = QNN_DATATYPE_UFIXED_POINT_8,
                                 .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding = {.scale = 1.0000000000000000f, .offset = -128}}},
                                 .rank = 4,
                                 .dimensions = weight_V_dimensions,
                                 .memType = QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf = {.data = weight_V,
                                                .dataSize = dimension * dimension * 1}}}}});

    // Q Matmul
    uint32_t dimensions_q_Out[] = {1, 1, sequence_length, dimension};
    std::string q_matmul_name  =  layer + "XQ_matmul_output";
    vector<Qnn_Tensor_t> matmul_q_outputs = {
        (Qnn_Tensor_t){
            .version = QNN_TENSOR_VERSION_1,
            {.v1 = {
                 .id = 0,
                 .name = q_matmul_name.c_str(),
                 .type = QNN_TENSOR_TYPE_NATIVE,
                 .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                 .dataType = QNN_DATATYPE_UFIXED_POINT_8,
                 .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                    {.scaleOffsetEncoding = {.scale = 1.0000000000000000f, .offset = -128}}},
                 .rank = 4,
                 .dimensions = dimensions_q_Out,
                 .memType = QNN_TENSORMEMTYPE_RAW,
                 {.clientBuf = {.data = nullptr,
                                .dataSize = 0}}}}}};

    vector<Qnn_Param_t> paramsMatmul = {
        {.paramType = QNN_PARAMTYPE_SCALAR,
         .name = "transpose_in0",
         {.scalarParam = (Qnn_Scalar_t){QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}},
        {.paramType = QNN_PARAMTYPE_SCALAR,
         .name = "transpose_in1",
         {.scalarParam = (Qnn_Scalar_t){QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}}};
    qbn->graphAddNode("layer0XQ_matmul", "MatMul", {quantize_x_outputs_name.c_str(), (layer + "W_Q").c_str()}, matmul_q_outputs, paramsMatmul, "qti.aisw");


    // K Matmul
    uint32_t dimensions_k_Out[] = {1, 1, sequence_length, dimension};
    std::string k_matmul_name  =  layer + "XK_matmul_output";
    vector<Qnn_Tensor_t> matmul_k_outputs = {
        (Qnn_Tensor_t){
            .version = QNN_TENSOR_VERSION_1,
            {.v1 = {
                 .id = 0,
                 .name = k_matmul_name.c_str(),
                 .type = QNN_TENSOR_TYPE_NATIVE,
                 .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                 .dataType = QNN_DATATYPE_UFIXED_POINT_8,
                 .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                    {.scaleOffsetEncoding = {.scale = 1.0000000000000000f, .offset = -128}}},
                 .rank = 4,
                 .dimensions = dimensions_k_Out,
                 .memType = QNN_TENSORMEMTYPE_RAW,
                 {.clientBuf = {.data = nullptr,
                                .dataSize = 0}}}}}};
    
    vector<Qnn_Param_t> params_xk_Matmul = {
        {.paramType = QNN_PARAMTYPE_SCALAR,
         .name = "transpose_in0",
         {.scalarParam = (Qnn_Scalar_t){QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}},
        {.paramType = QNN_PARAMTYPE_SCALAR,
         .name = "transpose_in1",
         {.scalarParam = (Qnn_Scalar_t){QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}}};
    qbn->graphAddNode("layer0XK_matmul", "MatMul", {quantize_x_outputs_name.c_str(), (layer + "W_K").c_str()}, matmul_k_outputs, params_xk_Matmul, "qti.aisw");



    // V Matmul
    uint32_t dimensions_v_Out[] = {1, 1, sequence_length, dimension};
    std::string v_matmul_name  =  layer + "XV_matmul_output";
    vector<Qnn_Tensor_t> matmul_v_outputs = {
        (Qnn_Tensor_t){
            .version = QNN_TENSOR_VERSION_1,
            {.v1 = {
                 .id = 0,
                 .name = v_matmul_name.c_str(),
                 .type = QNN_TENSOR_TYPE_NATIVE,
                 .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                 .dataType = QNN_DATATYPE_UFIXED_POINT_8,
                 .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                    {.scaleOffsetEncoding = {.scale = 1.0000000000000000f, .offset = -128}}},
                 .rank = 4,
                 .dimensions = dimensions_v_Out,
                 .memType = QNN_TENSORMEMTYPE_RAW,
                 {.clientBuf = {.data = nullptr,
                                .dataSize = 0}}}}}};

    vector<Qnn_Param_t> params_xv_Matmul = {
        {.paramType = QNN_PARAMTYPE_SCALAR,
         .name = "transpose_in0",
         {.scalarParam = (Qnn_Scalar_t){QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}},
        {.paramType = QNN_PARAMTYPE_SCALAR,
         .name = "transpose_in1",
         {.scalarParam = (Qnn_Scalar_t){QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}}};
    qbn->graphAddNode("layer0XV_matmul", "MatMul", {quantize_x_outputs_name.c_str(), "layer0W_V"}, matmul_v_outputs, params_xv_Matmul, "qti.aisw");


    // // Q K V Reshape
    // // Q Reshape
    // uint32_t Q_reshape[] = {1, sequence_length,  head,  dimension/head};
    // // uint32_t Q_reshape_dimensions[] = {1, 1,  1,  4};
    // // qbn->modelAddTensor(("layer" + layer + "Q_reshape").c_str(), // Node Name
    // //                     (Qnn_Tensor_t){
    // //                         .version = QNN_TENSOR_VERSION_1,
    // //                         {.v1 = {
    // //                              .id = 0,
    // //                              .name = ("layer" + layer + "Q_reshap").c_str(),
    // //                              .type = QNN_TENSOR_TYPE_STATIC,
    // //                              .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
    // //                              .dataType = QNN_DATATYPE_UFIXED_POINT_32,
    // //                              .quantizeParams = {QNN_DEFINITION_UNDEFINED,
    // //                                                 QNN_QUANTIZATION_ENCODING_UNDEFINED,
    // //                                                 {.scaleOffsetEncoding = {.scale = 1.0000000000000000f, .offset = 0}}},
    // //                              .rank = 4,
    // //                              .dimensions = Q_reshape_dimensions,
    // //                              .memType = QNN_TENSORMEMTYPE_RAW,
    // //                              {.clientBuf = {.data = Q_reshape,
    // //                                             .dataSize = 4 * 4}}}}});
    // std::string Q_reshape_output_name = layer + "Q_reshap-output";
    // vector<Qnn_Tensor_t> Q_reshape_output = {
    //     (Qnn_Tensor_t){
    //         .version = QNN_TENSOR_VERSION_1,
    //         {.v1 = {
    //              .id = 0,
    //              .name = Q_reshape_output_name.c_str(),
    //              .type = QNN_TENSOR_TYPE_NATIVE,
    //              .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
    //              .dataType = QNN_DATATYPE_UFIXED_POINT_16,
    //              .quantizeParams = {QNN_DEFINITION_UNDEFINED,
    //                                 QNN_QUANTIZATION_ENCODING_UNDEFINED,
    //                                 {.scaleOffsetEncoding = {.scale = 1.0000000000000000f, .offset = 0}}},
    //              .rank = 4,
    //              .dimensions = Q_reshape,
    //              .memType = QNN_TENSORMEMTYPE_RAW,
    //              {.clientBuf = {.data = nullptr,
    //                             .dataSize = 0}}}}}};

    // qbn->graphAddNode(("layer"+ layer + "Q_reshape").c_str(), "Reshape", {q_matmul_name.c_str()}, Q_reshape_output, {}, "qti.aisw");

    // // K Reshape
    // uint32_t K_reshape[] = {1, sequence_length,  head,  dimension/head};
    // // qbn->modelAddTensor(("layer" + layer + "K_reshape").c_str(), // Node Name
    // //                     (Qnn_Tensor_t){
    // //                         .version = QNN_TENSOR_VERSION_1,
    // //                         {.v1 = {
    // //                              .id = 0,
    // //                              .name = ("layer" + layer + "K_reshap").c_str(),
    // //                              .type = QNN_TENSOR_TYPE_STATIC,
    // //                              .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
    // //                              .dataType = QNN_DATATYPE_UFIXED_POINT_32,
    // //                              .quantizeParams = {QNN_DEFINITION_UNDEFINED,
    // //                                                 QNN_QUANTIZATION_ENCODING_UNDEFINED,
    // //                                                 {.scaleOffsetEncoding = {.scale = 1.0000000000000000f, .offset = 0}}},
    // //                              .rank = 4,
    // //                              .dimensions = Q_reshape_dimensions,
    // //                              .memType = QNN_TENSORMEMTYPE_RAW,
    // //                              {.clientBuf = {.data = K_reshape,
    // //                                             .dataSize = 4 * 4}}}}});

    // std::string K_reshape_output_name = layer + "K_reshap-output";
    // vector<Qnn_Tensor_t> K_reshape_output = {
    //     (Qnn_Tensor_t){
    //         .version = QNN_TENSOR_VERSION_1,
    //         {.v1 = {
    //              .id = 0,
    //              .name = K_reshape_output_name.c_str(),
    //              .type = QNN_TENSOR_TYPE_NATIVE,
    //              .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
    //              .dataType = QNN_DATATYPE_UFIXED_POINT_16,
    //              .quantizeParams = {QNN_DEFINITION_UNDEFINED,
    //                                 QNN_QUANTIZATION_ENCODING_UNDEFINED,
    //                                 {.scaleOffsetEncoding = {.scale = 1.0000000000000000f, .offset = 0}}},
    //              .rank = 4,
    //              .dimensions = K_reshape,
    //              .memType = QNN_TENSORMEMTYPE_RAW,
    //              {.clientBuf = {.data = nullptr,
    //                             .dataSize = 0}}}}}};

    // qbn->graphAddNode(("layer"+ layer + "K_reshape").c_str(), "Reshape", {k_matmul_name.c_str()}, K_reshape_output, {}, "qti.aisw");

    // // V Reshape
    // uint32_t V_reshape[] = {1, sequence_length,  head,  dimension/head};
    // // qbn->modelAddTensor(("layer" + layer + "V_reshape").c_str(), // Node Name
    // //                     (Qnn_Tensor_t){
    // //                         .version = QNN_TENSOR_VERSION_1,
    // //                         {.v1 = {
    // //                              .id = 0,
    // //                              .name = ("layer" + layer + "V_reshap").c_str(),
    // //                              .type = QNN_TENSOR_TYPE_STATIC,
    // //                              .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
    // //                              .dataType = QNN_DATATYPE_UFIXED_POINT_32,
    // //                              .quantizeParams = {QNN_DEFINITION_UNDEFINED,
    // //                                                 QNN_QUANTIZATION_ENCODING_UNDEFINED,
    // //                                                 {.scaleOffsetEncoding = {.scale = 1.0000000000000000f, .offset = 0}}},
    // //                              .rank = 4,
    // //                              .dimensions = Q_reshape_dimensions,
    // //                              .memType = QNN_TENSORMEMTYPE_RAW,
    // //                              {.clientBuf = {.data = V_reshape,
    // //                                             .dataSize = 4 * 4}}}}});

    // std::string V_reshape_output_name = layer + "V_reshap-output";
    // vector<Qnn_Tensor_t> V_reshape_output = {
    //     (Qnn_Tensor_t){
    //         .version = QNN_TENSOR_VERSION_1,
    //         {.v1 = {
    //              .id = 0,
    //              .name = V_reshape_output_name.c_str(),
    //              .type = QNN_TENSOR_TYPE_NATIVE,
    //              .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
    //              .dataType = QNN_DATATYPE_UFIXED_POINT_16,
    //              .quantizeParams = {QNN_DEFINITION_UNDEFINED,
    //                                 QNN_QUANTIZATION_ENCODING_UNDEFINED,
    //                                 {.scaleOffsetEncoding = {.scale = 1.0000000000000000f, .offset = 0}}},
    //              .rank = 4,
    //              .dimensions = V_reshape,
    //              .memType = QNN_TENSORMEMTYPE_RAW,
    //              {.clientBuf = {.data = nullptr,
    //                             .dataSize = 0}}}}}};

    // qbn->graphAddNode(("layer"+ layer + "V_reshape").c_str(), "Reshape", {v_matmul_name.c_str()}, V_reshape_output, {}, "qti.aisw");

    // //  Q K  V dequantize
    // // Q dequantize
    // std::string Q_dequantize_output_name = ("layer"+ layer + "Q_dequantize-output");
    // uint32_t Q_dequantize[] = {1, sequence_length,  head,  dimension/head};
    // vector<Qnn_Tensor_t> Q_dequantize_output = {
    //     (Qnn_Tensor_t){
    //         .version = QNN_TENSOR_VERSION_1,
    //         {.v1 = {
    //              .id = 0,
    //              .name = Q_dequantize_output_name.c_str(),
    //              .type = QNN_TENSOR_TYPE_NATIVE,
    //              .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
    //              .dataType = QNN_DATATYPE_FLOAT_32,
    //              .quantizeParams = {QNN_DEFINITION_UNDEFINED,
    //                                 QNN_QUANTIZATION_ENCODING_UNDEFINED,
    //                                 {.scaleOffsetEncoding = {.scale = 1.0000000000000000f, .offset = 0}}},
    //              .rank = 4,
    //              .dimensions = Q_dequantize,
    //              .memType = QNN_TENSORMEMTYPE_RAW,
    //              {.clientBuf = {.data = nullptr,
    //                             .dataSize = 0}}}}}};

    // qbn->graphAddNode(("layer"+ layer + "Q-Dequantize").c_str(), "Dequantize", {Q_reshape_output_name.c_str()}, Q_dequantize_output, {}, "qti.aisw");


    // // K dequantize
    // uint32_t K_dequantize[] = {1, sequence_length,  head,  dimension/head};
    // std::string K_dequantize_output_name = ("layer"+ layer + "K_dequantize-output");
    // vector<Qnn_Tensor_t> K_dequantize_output = {
    //     (Qnn_Tensor_t){
    //         .version = QNN_TENSOR_VERSION_1,
    //         {.v1 = {
    //              .id = 0,
    //              .name = K_dequantize_output_name.c_str(),
    //              .type = QNN_TENSOR_TYPE_NATIVE,
    //              .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
    //              .dataType = QNN_DATATYPE_FLOAT_32,
    //              .quantizeParams = {QNN_DEFINITION_UNDEFINED,
    //                                 QNN_QUANTIZATION_ENCODING_UNDEFINED,
    //                                 {.scaleOffsetEncoding = {.scale = 1.0000000000000000f, .offset = 0}}},
    //              .rank = 4,
    //              .dimensions = K_dequantize,
    //              .memType = QNN_TENSORMEMTYPE_RAW,
    //              {.clientBuf = {.data = nullptr,
    //                             .dataSize = 0}}}}}};

    // qbn->graphAddNode(("layer"+ layer + "K-Dequantize").c_str(), "Dequantize", {K_reshape_output_name.c_str()}, K_dequantize_output, {}, "qti.aisw");


    // // V dequantize
    // uint32_t V_dequantize[] = {1, sequence_length,  head,  dimension/head};
    // std::string V_dequantize_output_name = ("layer"+ layer + "V_dequantize-output");
    // vector<Qnn_Tensor_t> V_dequantize_output = {
    //     (Qnn_Tensor_t){
    //         .version = QNN_TENSOR_VERSION_1,
    //         {.v1 = {
    //              .id = 0,
    //              .name = V_dequantize_output_name.c_str(),
    //              .type = QNN_TENSOR_TYPE_NATIVE,
    //              .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
    //              .dataType = QNN_DATATYPE_FLOAT_32,
    //              .quantizeParams = {QNN_DEFINITION_UNDEFINED,
    //                                 QNN_QUANTIZATION_ENCODING_UNDEFINED,
    //                                 {.scaleOffsetEncoding = {.scale = 1.0000000000000000f, .offset = 0}}},
    //              .rank = 4,
    //              .dimensions = V_dequantize,
    //              .memType = QNN_TENSORMEMTYPE_RAW,
    //              {.clientBuf = {.data = nullptr,
    //                             .dataSize = 0}}}}}};

    // qbn->graphAddNode(("layer"+ layer + "V-Dequantize").c_str(), "Dequantize", {V_reshape_output_name.c_str()}, V_dequantize_output, {}, "qti.aisw");


    // // Q K V Cast
    // // Q Cast
    // uint32_t Q_cast[] = {1, sequence_length,  head,  dimension/head};
    // std::string Q_cast_output_name = ("layer"+ layer + "Q_cast-output");
    // vector<Qnn_Tensor_t> Q_cast_output = {
    //     (Qnn_Tensor_t){
    //         .version = QNN_TENSOR_VERSION_1,
    //         {.v1 = {
    //              .id = 0,
    //              .name = Q_cast_output_name.c_str(),
    //              .type = QNN_TENSOR_TYPE_NATIVE,
    //              .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
    //              .dataType = QNN_DATATYPE_FLOAT_16,
    //              .quantizeParams = {QNN_DEFINITION_UNDEFINED,
    //                                 QNN_QUANTIZATION_ENCODING_UNDEFINED,
    //                                 {.scaleOffsetEncoding = {.scale = 1.0000000000000000f, .offset = 0}}},
    //              .rank = 4,
    //              .dimensions = Q_cast,
    //              .memType = QNN_TENSORMEMTYPE_RAW,
    //              {.clientBuf = {.data = nullptr,
    //                             .dataSize = 0}}}}}};

    // // vector<Qnn_Param_t> params_cast = {
    // //     {.paramType = QNN_PARAMTYPE_SCALAR,
    // //      .name = "pose_type",
    // //      {.scalarParam = (Qnn_Scalar_t){QNN_DATATYPE_UINT_32, {.uint32Value = QNN_DATATYPE_FLOAT_16}}}}};

    // qbn->graphAddNode(("layer"+ layer + "Q-Cast").c_str(), "Cast", {Q_dequantize_output_name.c_str()}, Q_cast_output, {}, "qti.aisw");

    // // K Cast
    // uint32_t K_cast[] = {1, sequence_length,  head,  dimension/head};
    // std::string K_cast_output_name = ("layer"+ layer + "K_cast-output");
    // vector<Qnn_Tensor_t> K_cast_output = {
    //     (Qnn_Tensor_t){
    //         .version = QNN_TENSOR_VERSION_1,
    //         {.v1 = {
    //              .id = 0,
    //              .name = K_cast_output_name.c_str(),
    //              .type = QNN_TENSOR_TYPE_NATIVE,
    //              .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
    //              .dataType = QNN_DATATYPE_FLOAT_16,
    //              .quantizeParams = {QNN_DEFINITION_UNDEFINED,
    //                                 QNN_QUANTIZATION_ENCODING_UNDEFINED,
    //                                 {.scaleOffsetEncoding = {.scale = 1.0000000000000000f, .offset = 0}}},
    //              .rank = 4,
    //              .dimensions = K_cast,
    //              .memType = QNN_TENSORMEMTYPE_RAW,
    //              {.clientBuf = {.data = nullptr,
    //                             .dataSize = 0}}}}}};

    // qbn->graphAddNode(("layer"+ layer + "K-Cast").c_str(), "Cast", {K_dequantize_output_name.c_str()}, K_cast_output, {}, "qti.aisw");

    // // V Cast
    // uint32_t V_cast[] = {1, sequence_length,  head,  dimension/head};
    // std::string V_cast_output_name = ("layer"+ layer + "V_cast-output");
    // vector<Qnn_Tensor_t> V_cast_output = {
    //     (Qnn_Tensor_t){
    //         .version = QNN_TENSOR_VERSION_1,
    //         {.v1 = {
    //              .id = 0,
    //              .name = V_cast_output_name.c_str(),
    //              .type = QNN_TENSOR_TYPE_NATIVE,
    //              .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
    //              .dataType = QNN_DATATYPE_FLOAT_16,
    //              .quantizeParams = {QNN_DEFINITION_UNDEFINED,
    //                                 QNN_QUANTIZATION_ENCODING_UNDEFINED,
    //                                 {.scaleOffsetEncoding = {.scale = 1.0000000000000000f, .offset = 0}}},
    //              .rank = 4,
    //              .dimensions = V_cast,
    //              .memType = QNN_TENSORMEMTYPE_RAW,
    //              {.clientBuf = {.data = nullptr,
    //                             .dataSize = 0}}}}}};

    // qbn->graphAddNode(("layer"+ layer + "V-Cast").c_str(), "Cast", {V_dequantize_output_name.c_str()}, V_cast_output, {}, "qti.aisw");



    // // Q K Rope
    // int pos_max_ = 16384;
    // __fp16* sin_  = new __fp16[1*1*16384*dimension/head];
    // __fp16* cos_  = new __fp16[1*1*16384*dimension/head];
    // sinusoidal_position_embedding(1, 1, pos_max_, dimension/head, (void*)sin_, (void*)cos_);


    // uint32_t sin_dimensions[] = {16384, dimension/head};
    // uint32_t cos_dimensions[] = {16384, dimension/head};
    // qbn->modelAddTensor("sin", // Node Name
    //                     (Qnn_Tensor_t){
    //                         .version = QNN_TENSOR_VERSION_1,
    //                         {.v1 = {
    //                              .id = 0,
    //                              .name = "sin",
    //                              .type = QNN_TENSOR_TYPE_STATIC,
    //                              .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
    //                              .dataType = QNN_DATATYPE_FLOAT_16,
    //                              .quantizeParams = {QNN_DEFINITION_UNDEFINED,
    //                                                 QNN_QUANTIZATION_ENCODING_UNDEFINED,
    //                                                 {.scaleOffsetEncoding = {.scale = 1.0000000000000000f, .offset = 0}}},
    //                              .rank = 2,
    //                              .dimensions = sin_dimensions,
    //                              .memType = QNN_TENSORMEMTYPE_RAW,
    //                              {.clientBuf = {.data = sin_,
    //                                             .dataSize = 1*1*16384*128*2}}}}});

    // qbn->modelAddTensor("cos", // Node Name
    //                     (Qnn_Tensor_t){
    //                         .version = QNN_TENSOR_VERSION_1,
    //                         {.v1 = {
    //                              .id = 0,
    //                              .name = "cos",
    //                              .type = QNN_TENSOR_TYPE_STATIC,
    //                              .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
    //                              .dataType = QNN_DATATYPE_FLOAT_16,
    //                              .quantizeParams = {QNN_DEFINITION_UNDEFINED,
    //                                                 QNN_QUANTIZATION_ENCODING_UNDEFINED,
    //                                                 {.scaleOffsetEncoding = {.scale = 1.0000000000000000f, .offset = 0}}},
    //                              .rank = 2,
    //                              .dimensions = cos_dimensions,
    //                              .memType = QNN_TENSORMEMTYPE_RAW,
    //                              {.clientBuf = {.data = cos_,
    //                                             .dataSize = 1*1*16384*128*2}}}}});

    // uint32_t pose_type = 2;

    // vector<Qnn_Param_t> params_rope = {
    //     {.paramType = QNN_PARAMTYPE_SCALAR,
    //      .name = "pose_type",
    //      {.scalarParam = (Qnn_Scalar_t){QNN_DATATYPE_UINT_32, {.uint32Value = pose_type}}}}};


    // uint32_t q_rope_dimensions[] = {1, sequence_length,  head,  dimension/head};
    // std::string q_rope_outputs_name = layer + "Q-rope-output";
    // vector<Qnn_Tensor_t> q_rope_outputs = {
    //     (Qnn_Tensor_t){
    //         .version = QNN_TENSOR_VERSION_1,
    //         {.v1 = {
    //              .id = 0,
    //              .name = q_rope_outputs_name.c_str(),
    //              .type = QNN_TENSOR_TYPE_NATIVE,
    //              .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
    //              .dataType = QNN_DATATYPE_FLOAT_16,
    //              .quantizeParams = {QNN_DEFINITION_UNDEFINED,
    //                                 QNN_QUANTIZATION_ENCODING_UNDEFINED,
    //                                 {.scaleOffsetEncoding = {.scale = 1.0000000000000000f, .offset = 0}}},
    //              .rank = 4,
    //              .dimensions = q_rope_dimensions,
    //              .memType = QNN_TENSORMEMTYPE_RAW,
    //              {.clientBuf = {.data = nullptr,
    //                             .dataSize = 0}}}}}};
                                
    // qbn->graphAddNode("Q-rope", "RoPE", {Q_cast_output_name.c_str(), "sin", "cos"}, q_rope_outputs, params_rope, "LLaMAPackage");

    // uint32_t k_rope_dimensions[] = {1, sequence_length,  head,  dimension/head};
    // std::string k_rope_outputs_name = layer + "K-rope-output";
    // vector<Qnn_Tensor_t> k_rope_outputs = {
    //     (Qnn_Tensor_t){
    //         .version = QNN_TENSOR_VERSION_1,
    //         {.v1 = {
    //              .id = 0,
    //              .name = k_rope_outputs_name.c_str(),
    //              .type = QNN_TENSOR_TYPE_NATIVE,
    //              .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
    //              .dataType = QNN_DATATYPE_FLOAT_16,
    //              .quantizeParams = {QNN_DEFINITION_UNDEFINED,
    //                                 QNN_QUANTIZATION_ENCODING_UNDEFINED,
    //                                 {.scaleOffsetEncoding = {.scale = 1.0000000000000000f, .offset = 0}}},
    //              .rank = 4,
    //              .dimensions = k_rope_dimensions,
    //              .memType = QNN_TENSORMEMTYPE_RAW,
    //              {.clientBuf = {.data = nullptr,
    //                             .dataSize = 0}}}}}};
                                
    // qbn->graphAddNode("V-rope", "RoPE", {K_cast_output_name.c_str(), "sin", "cos"}, k_rope_outputs, params_rope, "LLaMAPackage");

    // // Q KT Matmul
    // uint32_t QKT_dimensions[] = {1, head, sequence_length, sequence_length};
    // vector<Qnn_Param_t> paramsQKTMatmul = {
    //     {.paramType = QNN_PARAMTYPE_SCALAR,
    //      .name = "transpose_in0",
    //      {.scalarParam = (Qnn_Scalar_t){QNN_DATATYPE_BOOL_8, {.bool8Value = 1}}}},
    //     {.paramType = QNN_PARAMTYPE_SCALAR,
    //      .name = "transpose_in1",
    //      {.scalarParam = (Qnn_Scalar_t){QNN_DATATYPE_BOOL_8, {.bool8Value = 1}}}}};

    // std::string  QKT_matmul_output_name = ("layer"+ layer + "QKT_HeadMatmul_output");
    // vector<Qnn_Tensor_t> QKT_matmul_output = {
    //     (Qnn_Tensor_t){
    //         .version = QNN_TENSOR_VERSION_1,
    //         {.v1 = {
    //              .id = 0,
    //              .name = QKT_matmul_output_name.c_str(),
    //              .type = QNN_TENSOR_TYPE_NATIVE,
    //              .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
    //              .dataType = QNN_DATATYPE_FLOAT_16,
    //              .quantizeParams = {QNN_DEFINITION_UNDEFINED,
    //                                 QNN_QUANTIZATION_ENCODING_UNDEFINED,
    //                                 {.scaleOffsetEncoding = {.scale = 1.0000000000000000f, .offset = 0}}},
    //              .rank = 4,
    //              .dimensions = QKT_dimensions,
    //              .memType = QNN_TENSORMEMTYPE_RAW,
    //              {.clientBuf = {.data = nullptr,
    //                             .dataSize = 0}}}}}};
    // qbn->graphAddNode(("layer"+ layer + "QKT-HeadMatmul").c_str(), "HeadMatmul", {Q_reshape_output_name.c_str(), K_reshape_output_name.c_str()}, QKT_matmul_output, paramsQKTMatmul, "LLaMAPackage");


    // // QKT Scale
    // uint32_t scale_dimensions[] = {1, head, sequence_length, sequence_length};
    // // add scale and bias tensor
    // uint32_t scalarDimensions[1] = {1};
    // __fp16 biasData[] = {0.0f};
    // __fp16 scaleData[] = {static_cast<__fp16>( 1.0f/dimension ) };
    // qbn->modelAddTensor(("layer" + layer + ".scale").c_str(), (Qnn_Tensor_t){
    //                                                              .version = QNN_TENSOR_VERSION_1,
    //                                                              {.v1 = {
    //                                                                   .id = 0,
    //                                                                   .name = ("layer" + layer + ".scale").c_str(),
    //                                                                   .type = QNN_TENSOR_TYPE_STATIC,
    //                                                                   .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
    //                                                                   .dataType = QNN_DATATYPE_FLOAT_16,
    //                                                                   .quantizeParams = {QNN_DEFINITION_UNDEFINED,
    //                                                                                      QNN_QUANTIZATION_ENCODING_UNDEFINED,
    //                                                                                      {.scaleOffsetEncoding = {.scale = 1.0000000000000000f, .offset = 0}}},
    //                                                                   .rank = 1,
    //                                                                   .dimensions = scalarDimensions,
    //                                                                   .memType = QNN_TENSORMEMTYPE_RAW,
    //                                                                   {.clientBuf = {.data = scaleData,
    //                                                                                  .dataSize = 2}}}}});
    // qbn->modelAddTensor(("layer" + layer + ".bias").c_str(), (Qnn_Tensor_t){
    //                                                             .version = QNN_TENSOR_VERSION_1,
    //                                                             {.v1 = {
    //                                                                  .id = 0,
    //                                                                  .name = ("layer" + layer + ".bias").c_str(),
    //                                                                  .type = QNN_TENSOR_TYPE_STATIC,
    //                                                                  .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
    //                                                                  .dataType = QNN_DATATYPE_FLOAT_16,
    //                                                                  .quantizeParams = {QNN_DEFINITION_UNDEFINED,
    //                                                                                     QNN_QUANTIZATION_ENCODING_UNDEFINED,
    //                                                                                     {.scaleOffsetEncoding = {.scale = 1.0000000000000000f, .offset = 0}}},
    //                                                                  .rank = 1,
    //                                                                  .dimensions = scalarDimensions,
    //                                                                  .memType = QNN_TENSORMEMTYPE_RAW,
    //                                                                  {.clientBuf = {.data = biasData,
    //                                                                                 .dataSize = 2}}}}});
    // // convert output to qnn tensor
    // std::string qk_scale_mul_outputs_name = ("layer" + layer + ".scale_mul_output");
    // vector<Qnn_Tensor_t> qk_scale_mul_outputs = {
    //     {.version = QNN_TENSOR_VERSION_1,
    //      {.v1 = {
    //           .id = 0,
    //           .name = qk_scale_mul_outputs_name.c_str(),
    //           .type = QNN_TENSOR_TYPE_APP_READ,
    //           .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
    //           .dataType = QNN_DATATYPE_FLOAT_16,
    //           .quantizeParams = {QNN_DEFINITION_UNDEFINED,
    //                              QNN_QUANTIZATION_ENCODING_UNDEFINED,
    //                              {.scaleOffsetEncoding = {.scale = 1.0000000000000000f, .offset = 0}}},
    //           .rank = 4,
    //           .dimensions = scale_dimensions,
    //           .memType = QNN_TENSORMEMTYPE_RAW,
    //           {.clientBuf = {.data = nullptr,
    //                          .dataSize = 0}}}}}};

    // qbn->graphAddNode(("layer"+ layer + "QKT-scale").c_str(), "ElementWiseMultiply", {QKT_matmul_output_name.c_str(), ("layer" + layer + ".scale").c_str()}, qk_scale_mul_outputs, {}, "qti.aisw");

    // std::string qk_scale_add_outputs_name = ("layer" + layer + ".scale_add_output");
    // vector<Qnn_Tensor_t> qk_scale_add_outputs = {
    //     {.version = QNN_TENSOR_VERSION_1,
    //      {.v1 = {
    //           .id = 0,
    //           .name = qk_scale_add_outputs_name.c_str(),
    //           .type = QNN_TENSOR_TYPE_APP_READ,
    //           .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
    //           .dataType = QNN_DATATYPE_FLOAT_16,
    //           .quantizeParams = {QNN_DEFINITION_UNDEFINED,
    //                              QNN_QUANTIZATION_ENCODING_UNDEFINED,
    //                              {.scaleOffsetEncoding = {.scale = 1.0000000000000000f, .offset = 0}}},
    //           .rank = 4,
    //           .dimensions = scale_dimensions,
    //           .memType = QNN_TENSORMEMTYPE_RAW,
    //           {.clientBuf = {.data = nullptr,
    //                          .dataSize = 0}}}}}};


    // qbn->graphAddNode(("layer"+ layer + "QKT-scale").c_str(), "ElementWiseAdd", {qk_scale_mul_outputs_name.c_str(),  ("layer" + layer + ".bias").c_str()}, qk_scale_add_outputs, {}, "qti.aisw");


    // // QKT CausalMask
    // uint32_t ck_qk_dimensions[] = {1, head, sequence_length, sequence_length};
    // std::string ck_qk_outputs_name = ("layer" + layer + ".CausalMask_output");
    // vector<Qnn_Tensor_t> ck_qk_outputs = {
    //     (Qnn_Tensor_t){
    //         .version = QNN_TENSOR_VERSION_1,
    //         {.v1 = {
    //              .id = 0,
    //              .name = ck_qk_outputs_name.c_str(),
    //              .type = QNN_TENSOR_TYPE_NATIVE,
    //              .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
    //              .dataType = QNN_DATATYPE_FLOAT_16,
    //              .quantizeParams = {QNN_DEFINITION_UNDEFINED,
    //                                 QNN_QUANTIZATION_ENCODING_UNDEFINED,
    //                                 {.scaleOffsetEncoding = {.scale = 1.0000000000000000f, .offset = 0}}},
    //              .rank = 4,
    //              .dimensions = ck_qk_dimensions,
    //              .memType = QNN_TENSORMEMTYPE_RAW,
    //              {.clientBuf = {.data = nullptr,
    //                             .dataSize = 0}}}}}};

    // qbn->graphAddNode(("layer"+ layer + "QKT-CausalMask").c_str(), "CausalMask", {qk_scale_add_outputs_name.c_str()}, ck_qk_outputs, {}, "LLaMAPackage");


    // // QK Softmax
    // uint32_t sm_qk_dimensions[] = {1, head, sequence_length, sequence_length};
    // std::string  sm_qk_outputs_names = ("layer" + layer + ".softmax_output");
    // vector<Qnn_Tensor_t> sm_qk_outputs = {
    //     (Qnn_Tensor_t){
    //         .version = QNN_TENSOR_VERSION_1,
    //         {.v1 = {
    //              .id = 0,
    //              .name = sm_qk_outputs_names.c_str(),
    //              .type = QNN_TENSOR_TYPE_NATIVE,
    //              .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
    //              .dataType = QNN_DATATYPE_FLOAT_16,
    //              .quantizeParams = {QNN_DEFINITION_UNDEFINED,
    //                                 QNN_QUANTIZATION_ENCODING_UNDEFINED,
    //                                 {.scaleOffsetEncoding = {.scale = 1.0000000000000000f, .offset = 0}}},
    //              .rank = 4,
    //              .dimensions = sm_qk_dimensions,
    //              .memType = QNN_TENSORMEMTYPE_RAW,
    //              {.clientBuf = {.data = nullptr,
    //                             .dataSize = 0}}}}}};


    // vector<Qnn_Param_t> soft_params = {
    //     {.paramType = QNN_PARAMTYPE_SCALAR,
    //      .name = "axis",
    //      {.scalarParam = (Qnn_Scalar_t){QNN_DATATYPE_UINT_32, {.uint32Value = 3}}}},
    //     {.paramType = QNN_PARAMTYPE_SCALAR,
    //      .name = "beta",
    //      {.scalarParam = (Qnn_Scalar_t){QNN_DATATYPE_FLOAT_32, {.floatValue = 1.000000000000f}}}}};

    // qbn->graphAddNode(("layer" + layer + ".softmax").c_str(), "Softmax", {ck_qk_outputs_name.c_str()}, sm_qk_outputs, soft_params, "qti.aisw");



    // // QKT V Matmul
    // uint32_t QKTV_dimensions[] = {1, sequence_length, head, dimension/head};
    // std::string QKTV_matmul_output_name = ("layer"+ layer + "QKTV_HeadMatmul_output");
    // vector<Qnn_Tensor_t> QKTV_matmul_output = {
    //     (Qnn_Tensor_t){
    //         .version = QNN_TENSOR_VERSION_1,
    //         {.v1 = {
    //              .id = 0,
    //              .name = QKTV_matmul_output_name.c_str(),
    //              .type = QNN_TENSOR_TYPE_NATIVE,
    //              .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
    //              .dataType = QNN_DATATYPE_FLOAT_16,
    //              .quantizeParams = {QNN_DEFINITION_UNDEFINED,
    //                                 QNN_QUANTIZATION_ENCODING_UNDEFINED,
    //                                 {.scaleOffsetEncoding = {.scale = 1.0000000000000000f, .offset = 0}}},
    //              .rank = 4,
    //              .dimensions = QKTV_dimensions,
    //              .memType = QNN_TENSORMEMTYPE_RAW,
    //              {.clientBuf = {.data = nullptr,
    //                             .dataSize = 0}}}}}};
    // vector<Qnn_Param_t> paramsQKTVMatmul = {
    //     {.paramType = QNN_PARAMTYPE_SCALAR,
    //      .name = "transpose_in0",
    //      {.scalarParam = (Qnn_Scalar_t){QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}},
    //     {.paramType = QNN_PARAMTYPE_SCALAR,
    //      .name = "transpose_in1",
    //      {.scalarParam = (Qnn_Scalar_t){QNN_DATATYPE_BOOL_8, {.bool8Value = 1}}}}};
    // qbn->graphAddNode(("layer"+ layer + "QKTV-HeadMatmul").c_str(), "HeadMatmul", {sm_qk_outputs_names.c_str(), V_cast_output_name.c_str()}, QKTV_matmul_output, paramsQKTVMatmul, "LLaMAPackage");

    // // O Reshape
    // uint32_t O_reshape[] = {1, 1, sequence_length,  dimension};
    // // qbn->modelAddTensor(("layer" + layer + "O_reshape").c_str(), // Node Name
    // //                     (Qnn_Tensor_t){
    // //                         .version = QNN_TENSOR_VERSION_1,
    // //                         {.v1 = {
    // //                              .id = 0,
    // //                              .name = ("layer" + layer + "O_reshap").c_str(),
    // //                              .type = QNN_TENSOR_TYPE_STATIC,
    // //                              .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
    // //                              .dataType = QNN_DATATYPE_FLOAT_16,
    // //                              .quantizeParams = {QNN_DEFINITION_UNDEFINED,
    // //                                                 QNN_QUANTIZATION_ENCODING_UNDEFINED,
    // //                                                 {.scaleOffsetEncoding = {.scale = 1.0000000000000000f, .offset = 0}}},
    // //                              .rank = 4,
    // //                              .dimensions = Q_reshape_dimensions,
    // //                              .memType = QNN_TENSORMEMTYPE_RAW,
    // //                              {.clientBuf = {.data = O_reshape,
    // //                                             .dataSize = 4 * 4}}}}});

    // std::string O_reshape_output_name = ("layer"+ layer + "O_reshap-output");
    // vector<Qnn_Tensor_t> O_reshape_output = {
    //     (Qnn_Tensor_t){
    //         .version = QNN_TENSOR_VERSION_1,
    //         {.v1 = {
    //              .id = 0,
    //              .name = O_reshape_output_name.c_str(),
    //              .type = QNN_TENSOR_TYPE_NATIVE,
    //              .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
    //              .dataType = QNN_DATATYPE_FLOAT_16,
    //              .quantizeParams = {QNN_DEFINITION_UNDEFINED,
    //                                 QNN_QUANTIZATION_ENCODING_UNDEFINED,
    //                                 {.scaleOffsetEncoding = {.scale = 1.0000000000000000f, .offset = 0}}},
    //              .rank = 4,
    //              .dimensions = O_reshape,
    //              .memType = QNN_TENSORMEMTYPE_RAW,
    //              {.clientBuf = {.data = nullptr,
    //                             .dataSize = 0}}}}}};

    // qbn->graphAddNode(("layer"+ layer + "O-reshape").c_str(), "Reshape", {QKTV_matmul_output_name.c_str()}, O_reshape_output, {}, "qti.aisw");


    // // W_O
    // uint32_t weight_O_dimensions[] = {1, 1, dimension, dimension};
    // __fp16* weight_O = new __fp16[dimension * dimension]; 
    // qbn->modelAddTensor(("layer" + layer + "W_O").c_str(), // Node Name
    //                     (Qnn_Tensor_t){
    //                         .version = QNN_TENSOR_VERSION_1,
    //                         {.v1 = {
    //                              .id = 0,
    //                              .name = ("layer" + layer + "W_O").c_str(),
    //                              .type = QNN_TENSOR_TYPE_STATIC,
    //                              .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
    //                              .dataType = QNN_DATATYPE_FLOAT_16,
    //                              .quantizeParams = {QNN_DEFINITION_UNDEFINED,
    //                                                 QNN_QUANTIZATION_ENCODING_UNDEFINED,
    //                                                 {.scaleOffsetEncoding = {.scale = 1.0000000000000000f, .offset = 0}}},
    //                              .rank = 4,
    //                              .dimensions = weight_O_dimensions,
    //                              .memType = QNN_TENSORMEMTYPE_RAW,
    //                              {.clientBuf = {.data = weight_O,
    //                                             .dataSize = dimension * dimension * 2}}}}});

    // uint32_t O_matmul_dimensions[] = {1, 1, sequence_length, dimension};
    // std::string matmul_o_outputs_names = ("layer"+ layer + "XO-matmul-output");
    // vector<Qnn_Tensor_t> matmul_o_outputs = {
    //     (Qnn_Tensor_t){
    //         .version = QNN_TENSOR_VERSION_1,
    //         {.v1 = {
    //              .id = 0,
    //              .name = matmul_o_outputs_names.c_str(),
    //              .type = QNN_TENSOR_TYPE_APP_READ,
    //              .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
    //              .dataType = QNN_DATATYPE_FLOAT_16,
    //              .quantizeParams = {QNN_DEFINITION_UNDEFINED,
    //                                 QNN_QUANTIZATION_ENCODING_UNDEFINED,
    //                                 {.scaleOffsetEncoding = {.scale = 1.0000000000000000f, .offset = 0}}},
    //              .rank = 4,
    //              .dimensions = O_matmul_dimensions,
    //              .memType = QNN_TENSORMEMTYPE_RAW,
    //              {.clientBuf = {.data = nullptr,
    //                             .dataSize = 0}}}}}};
    // vector<Qnn_Param_t> params_O_Matmul = {
    //     {.paramType = QNN_PARAMTYPE_SCALAR,
    //      .name = "transpose_in0",
    //      {.scalarParam = (Qnn_Scalar_t){QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}},
    //     {.paramType = QNN_PARAMTYPE_SCALAR,
    //      .name = "transpose_in1",
    //      {.scalarParam = (Qnn_Scalar_t){QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}}};
    // qbn->graphAddNode(("layer"+ layer + "XO-matmul").c_str(), "MatMul", {O_reshape_output_name.c_str(), ("layer" + layer + "W_O").c_str()}, matmul_o_outputs, params_O_Matmul, "qti.aisw");


    __fp16* output_data = new __fp16[1 * 1 * sequence_length * dimension]; 


    // graph compile
    std::cout << "graph compile" << std::endl;
    qbn->graphFinilize();

    // build input and outputs
    std::map<std::string, std::vector<uint8_t*>> inputBufferMap;
    std::vector<uint8_t*> inputBuffers;
    inputBuffers.push_back((uint8_t*)input_data);
    inputBufferMap.insert(std::make_pair("graph", inputBuffers));

    std::map<std::string, std::vector<uint8_t*>> outputBufferMap;
    std::vector<uint8_t*> outputBuffers;
    outputBuffers.push_back((uint8_t*)output_data);
    outputBufferMap.insert(std::make_pair("graph", outputBuffers));

    // graph run
    std::cout << "graph run" << std::endl;
    qbn->graphExecute(inputBufferMap, outputBufferMap);


}             



int main() {
    BackendConfig bnc;

    shared_ptr<MemoryManager> mm = nullptr;
    switch (bnc.memory) {
    case BackendConfig::Memory_High:
        mm = std::make_shared<SystemMemoryManager>();
        break;
    default:
        mm = std::make_shared<SystemMemoryManager>();
        break;
    }

    QNNBackend *qbn = new QNNBackend(mm);
    testLLaMAAttention(qbn, 1, 1);
    

    delete qbn;
}