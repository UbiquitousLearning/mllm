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
    uint32_t dimensions[] = {1, 2, 1, 2};
    float data[] = {3, 2, -1, 0,};
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

    __fp16 output_data[4];
    vector<Qnn_Tensor_t> outputs = {
        (Qnn_Tensor_t){
            .version = QNN_TENSOR_VERSION_1,
            {.v1 = {
                 .id = 0,
                 .name = "CausalMask-output",
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

    qbn->graphAddNode("qnn-CausalMask", "CausalMask", {"x"}, outputs, {}, "LLaMAPackage");
    
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

    for(int i=0; i<4; i++) 
        std::cout << output_data[i] << std::endl;

    delete qbn;
}