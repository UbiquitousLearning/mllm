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

#include <iostream>
#include <fstream>

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

    // build graph
    std::cout << "build graph" << std::endl;
    // graph add node
    
    uint32_t dimensions[] = {1, 180, 1, 128};
    std::ifstream file("/mllm/test/qnn/QNNRoPETest.input.output.txt"); // 打开文件
    if (!file.is_open()) { // 检查文件是否成功打开
        std::cout << "无法打开文件" << std::endl;
        return 1;
    }

    __fp16* data = new __fp16[1*180*1*128];

    for (int i = 0; i < 1*180*1*128; ++i) {
        float ff;
        if (!(file >> ff)) { // 逐个读取 float 数据到数组中
            data[i] = (__fp16)ff;
            std::cout << "读取文件失败" << std::endl;
            return 1;
        }
    }

    qbn->modelAddTensor("x", // Node Name
                        (Qnn_Tensor_t){
                            .version = QNN_TENSOR_VERSION_1,
                            {.v1 = {
                                 .id = 0,
                                 .name = "x",
                                 .type = QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                 .dataType = QNN_DATATYPE_FLOAT_16,
                                 .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                                 .rank = 4,
                                 .dimensions = dimensions,
                                 .memType = QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf = {.data = data,
                                                .dataSize = 1*180*1*128*2}}}}});

    int pos_max_ = 16384;
    int ishape = 128;
    __fp16* sin_  = new __fp16[1*1*16384*128];
    __fp16* cos_  = new __fp16[1*1*16384*128];
    sinusoidal_position_embedding(1, 1, pos_max_, ishape, (void*)sin_, (void*)cos_);


    uint32_t sin_dimensions[] = {16384, 128};
    uint32_t cos_dimensions[] = {16384, 128};
    qbn->modelAddTensor("sin", // Node Name
                        (Qnn_Tensor_t){
                            .version = QNN_TENSOR_VERSION_1,
                            {.v1 = {
                                 .id = 0,
                                 .name = "sin",
                                 .type = QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                 .dataType = QNN_DATATYPE_FLOAT_16,
                                 .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                                 .rank = 2,
                                 .dimensions = sin_dimensions,
                                 .memType = QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf = {.data = sin_,
                                                .dataSize = 1*1*16384*128*2}}}}});

    qbn->modelAddTensor("cos", // Node Name
                        (Qnn_Tensor_t){
                            .version = QNN_TENSOR_VERSION_1,
                            {.v1 = {
                                 .id = 0,
                                 .name = "cos",
                                 .type = QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                 .dataType = QNN_DATATYPE_FLOAT_16,
                                 .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                                 .rank = 2,
                                 .dimensions = cos_dimensions,
                                 .memType = QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf = {.data = cos_,
                                                .dataSize = 1*1*16384*128*2}}}}});

    uint32_t pose_type = 2;

    vector<Qnn_Param_t> params_rope = {
        {.paramType = QNN_PARAMTYPE_SCALAR,
         .name = "pose_type",
         {.scalarParam = (Qnn_Scalar_t){QNN_DATATYPE_UINT_32, {.uint32Value = pose_type}}}}};



    __fp16 output_data[1*180*1*128];
    vector<Qnn_Tensor_t> outputs = {
        (Qnn_Tensor_t){
            .version = QNN_TENSOR_VERSION_1,
            {.v1 = {
                 .id = 0,
                 .name = "rope-output",
                 .type = QNN_TENSOR_TYPE_APP_READ,
                 .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                 .dataType = QNN_DATATYPE_FLOAT_16,
                 .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                    {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                 .rank = 4,
                 .dimensions = dimensions,
                 .memType = QNN_TENSORMEMTYPE_RAW,
                 {.clientBuf = {.data = nullptr,
                                .dataSize = 0}}}}}};
                                
    qbn->graphAddNode("qnn-rope", "RoPE", {"x", "sin", "cos"}, outputs, params_rope, "LLaMAPackage");
    // graph compile
    std::cout << "graph compile" << std::endl;
    qbn->graphFinilize();
    // graph run
    std::cout << "graph run" << std::endl;
    qbn->graphExecute();

    // for(int i=0; i<8; i++) 
    //     std::cout << output_data[i] << std::endl;

    delete qbn;
}