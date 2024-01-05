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
    
    uint32_t dimensions[] = {1, 1, 10, 32000};
    std::ifstream file("/mllm/test/qnn/QNNRMSNormTest.input.weight.output.txt"); // 打开文件
    if (!file.is_open()) { // 检查文件是否成功打开
        std::cout << "无法打开文件" << std::endl;
        return 1;
    }

    float* data = new float[1*1*10*32000];

    for (int i = 0; i < 1*1*10*32000; ++i) {
        float ff;
        if (!(file >> data[i] )) { // 逐个读取 float 数据到数组中
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
                                 .type = QNN_TENSOR_TYPE_APP_WRITE,
                                 .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                 .dataType = QNN_DATATYPE_FLOAT_16,
                                 .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                                 .rank = 4,
                                 .dimensions = dimensions,
                                 .memType = QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf = {.data = nullptr,
                                                .dataSize = 0}}}}});

    __fp16* weights  = new __fp16[1*1*1*32000];
    for (int i = 0; i < 1*1*1*32000; ++i) {
        float ff;
        if (!(file >> ff )) { // 逐个读取 float 数据到数组中
            std::cout << "读取文件失败" << std::endl;
            return 1;
        }
        weights[i] = static_cast<__fp16>(ff); 
    }

    uint32_t weights_dimensions[] = {32000};

    qbn->modelAddTensor("weights", // Node Name
                        (Qnn_Tensor_t){
                            .version = QNN_TENSOR_VERSION_1,
                            {.v1 = {
                                 .id = 0,
                                 .name = "weights",
                                 .type = QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                 .dataType = QNN_DATATYPE_FLOAT_16,
                                 .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                                 .rank = 1,
                                 .dimensions = weights_dimensions,
                                 .memType = QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf = {.data = weights,
                                                .dataSize = 1*1*1*32000*2}}}}});



    __fp16* output_data = new __fp16[1*1*10*32000];
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
                                
    qbn->graphAddNode("qnn-rmsnorm", "RMSNorm", {"x", "weights"}, outputs, {}, "LLaMAPackage");
    // graph compile
    std::cout << "graph compile" << std::endl;
    qbn->graphFinilize();

    // build input and outputs
    std::map<std::string, std::vector<uint8_t*>> inputBufferMap;
    std::vector<uint8_t*> inputBuffers;
    inputBuffers.push_back((uint8_t*)data);
    inputBufferMap.insert(std::make_pair("graph", inputBuffers));

    std::map<std::string, std::vector<uint8_t*>> outputBufferMap;
    std::vector<uint8_t*> outputBuffers;
    outputBuffers.push_back((uint8_t*)output_data);
    outputBufferMap.insert(std::make_pair("graph", outputBuffers));

    // graph run
    std::cout << "graph run" << std::endl;
    qbn->graphExecute(inputBufferMap, outputBufferMap);

    // for(int i=0; i<1*180*1*128; i++) 
    //     std::cout << output_data[i] << " ";


    float* label_data = new float[1*1*10*32000];
    for (int i = 0; i < 1*1*10*32000; ++i) {
        float ff;
        if (!(file >> label_data[i] )) { // 逐个读取 float 数据到数组中
            std::cout << "读取文件失败" << std::endl;
            return 1;
        }
    }

    for (int i = 0; i < 1*1*10*32; ++i) {
        __fp16 a_ = output_data[i];
        __fp16 b_ = static_cast<__fp16>(label_data[i]);
        float eps = 0.000001;
        if ( (abs(a_ - b_) / std::max(a_, b_)) > eps  ) {
            std::cout << a_ << " " << b_ << std::endl;
        }
    }

    delete qbn;
}