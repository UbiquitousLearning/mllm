#include <iostream>

#include "backends/QNN/QNNBackend.hpp"
#include "memory/SystemMemoryManager.hpp"
#include "backends/QNN/op/QNNAdd.hpp"

using namespace mllm;

void testMatMul(QNNBackend *qbn) {
    // graph add node
    uint32_t dimensions0[] = {1, 2, 2, 2};
    uint32_t dimensions1[] = {1, 1, 4, 2};
    qbn->modelAddTensor("x", // Node Name
                        (Qnn_Tensor_t){
                            .version = QNN_TENSOR_VERSION_1,
                            {.v1 = {
                                 .id = 0,
                                 .name = "x",
                                 .type = QNN_TENSOR_TYPE_APP_WRITE,
                                 .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                 .dataType = QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                                 .rank = 4,
                                 .dimensions = dimensions0,
                                 .memType = QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf = {.data = nullptr,
                                                .dataSize = 0}}}}});

    float data[] = {1, 2, 3, 4, 5, 6, 7, 8};
    qbn->modelAddTensor("y", // Node Name
                        (Qnn_Tensor_t){
                            .version = QNN_TENSOR_VERSION_1,
                            {.v1 = {
                                 .id = 0,
                                 .name = "y",
                                 .type = QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                 .dataType = QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                                 .rank = 4,
                                 .dimensions = dimensions1,
                                 .memType = QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf = {.data = data,
                                                .dataSize = 32}}}}});

    uint32_t dimensionsOut[] = {1, 2, 2, 4};
    vector<Qnn_Tensor_t> outputs = {
        (Qnn_Tensor_t){
            .version = QNN_TENSOR_VERSION_1,
            {.v1 = {
                 .id = 0,
                 .name = "add-output",
                 .type = QNN_TENSOR_TYPE_APP_READ,
                 .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                 .dataType = QNN_DATATYPE_FLOAT_32,
                 .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                    {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                 .rank = 4,
                 .dimensions = dimensionsOut,
                 .memType = QNN_TENSORMEMTYPE_RAW,
                 {.clientBuf = {.data = nullptr,
                                .dataSize = 0}}}}}};
    vector<Qnn_Param_t> paramsMatmul = {
        {.paramType = QNN_PARAMTYPE_SCALAR,
         .name = "transpose_in0",
         {.scalarParam = (Qnn_Scalar_t){QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}},
        {.paramType = QNN_PARAMTYPE_SCALAR,
         .name = "transpose_in1",
         {.scalarParam = (Qnn_Scalar_t){QNN_DATATYPE_BOOL_8, {.bool8Value = 1}}}}};
    qbn->graphAddNode("qnn-add", "MatMul", {"x", "y"}, outputs, paramsMatmul, "qti.aisw");
}

void testSoftmax(QNNBackend *qbn) {
    uint32_t dimensions[] = {1, 2, 2, 2};

    vector<Qnn_Param_t> params = {
        {.paramType = QNN_PARAMTYPE_SCALAR,
         .name = "axis",
         {.scalarParam = (Qnn_Scalar_t){QNN_DATATYPE_UINT_32, {.uint32Value = static_cast<uint32_t>(3)}}}},
        {.paramType = QNN_PARAMTYPE_SCALAR,
         .name = "beta",
         {.scalarParam = (Qnn_Scalar_t){QNN_DATATYPE_FLOAT_32, {.floatValue = 1.000000000000f}}}}};
    qbn->modelAddTensor("x", // Node Name
                        (Qnn_Tensor_t){
                            .version = QNN_TENSOR_VERSION_1,
                            {.v1 = {
                                 .id = 0,
                                 .name = "x",
                                 .type = QNN_TENSOR_TYPE_APP_WRITE,
                                 .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                 .dataType = QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                                 .rank = 4,
                                 .dimensions = dimensions,
                                 .memType = QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf = {.data = nullptr,
                                                .dataSize = 0}}}}});
    vector<Qnn_Tensor_t> outputs = {
        (Qnn_Tensor_t){
            .version = QNN_TENSOR_VERSION_1,
            {.v1 = {
                 .id = 0,
                 .name = "add-output",
                 .type = QNN_TENSOR_TYPE_APP_READ,
                 .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                 .dataType = QNN_DATATYPE_FLOAT_32,
                 .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                    {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                 .rank = 4,
                 .dimensions = dimensions,
                 .memType = QNN_TENSORMEMTYPE_RAW,
                 {.clientBuf = {.data = nullptr,
                                .dataSize = 0}}}}}};
    qbn->graphAddNode("qnn-softmax", "Softmax", {"x"}, outputs, params, "qti.aisw");
}

void testAdd(QNNBackend *qbn) {
    // graph add node
    uint32_t dimensions0[] = {1, 2, 2, 2};
    uint32_t dimensions1[] = {1, 1, 1, 1};
    qbn->modelAddTensor("x", // Node Name
                        (Qnn_Tensor_t){
                            .version = QNN_TENSOR_VERSION_1,
                            {.v1 = {
                                 .id = 0,
                                 .name = "x",
                                 .type = QNN_TENSOR_TYPE_APP_WRITE,
                                 .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                 .dataType = QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                                 .rank = 4,
                                 .dimensions = dimensions0,
                                 .memType = QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf = {.data = nullptr,
                                                .dataSize = 0}}}}});

    float data[] = {1};
    qbn->modelAddTensor("y", // Node Name
                        (Qnn_Tensor_t){
                            .version = QNN_TENSOR_VERSION_1,
                            {.v1 = {
                                 .id = 0,
                                 .name = "y",
                                 .type = QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                 .dataType = QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                                 .rank = 4,
                                 .dimensions = dimensions1,
                                 .memType = QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf = {.data = data,
                                                .dataSize = 4}}}}});

    uint32_t dimensionsOut[] = {1, 2, 2, 2};
    vector<Qnn_Tensor_t> outputs = {
        (Qnn_Tensor_t){
            .version = QNN_TENSOR_VERSION_1,
            {.v1 = {
                 .id = 0,
                 .name = "add-output",
                 .type = QNN_TENSOR_TYPE_APP_READ,
                 .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                 .dataType = QNN_DATATYPE_FLOAT_32,
                 .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                    {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                 .rank = 4,
                 .dimensions = dimensionsOut,
                 .memType = QNN_TENSORMEMTYPE_RAW,
                 {.clientBuf = {.data = nullptr,
                                .dataSize = 0}}}}}};

    qbn->graphAddNode("qnn-add", "ElementWiseAdd", {"x", "y"}, outputs, {}, "qti.aisw");
}

void testMul(QNNBackend *qbn) {
    // graph add node
    uint32_t dimensions0[] = {1, 2, 2, 2};
    uint32_t dimensions1[] = {1};
    qbn->modelAddTensor("x", // Node Name
                        (Qnn_Tensor_t){
                            .version = QNN_TENSOR_VERSION_1,
                            {.v1 = {
                                 .id = 0,
                                 .name = "x",
                                 .type = QNN_TENSOR_TYPE_APP_WRITE,
                                 .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                 .dataType = QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                                 .rank = 4,
                                 .dimensions = dimensions0,
                                 .memType = QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf = {.data = nullptr,
                                                .dataSize = 0}}}}});

    float data[] = {2};
    qbn->modelAddTensor("y", // Node Name
                        (Qnn_Tensor_t){
                            .version = QNN_TENSOR_VERSION_1,
                            {.v1 = {
                                 .id = 0,
                                 .name = "y",
                                 .type = QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                 .dataType = QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                                 .rank = 1,
                                 .dimensions = dimensions1,
                                 .memType = QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf = {.data = data,
                                                .dataSize = 4}}}}});

    uint32_t dimensionsOut[] = {1, 2, 2, 2};
    vector<Qnn_Tensor_t> outputs = {
        (Qnn_Tensor_t){
            .version = QNN_TENSOR_VERSION_1,
            {.v1 = {
                 .id = 0,
                 .name = "add-output",
                 .type = QNN_TENSOR_TYPE_APP_READ,
                 .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                 .dataType = QNN_DATATYPE_FLOAT_32,
                 .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                    {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                 .rank = 4,
                 .dimensions = dimensionsOut,
                 .memType = QNN_TENSORMEMTYPE_RAW,
                 {.clientBuf = {.data = nullptr,
                                .dataSize = 0}}}}}};

    qbn->graphAddNode("qnn-mul", "ElementWiseMultiply", {"x", "y"}, outputs, {}, "qti.aisw");
}