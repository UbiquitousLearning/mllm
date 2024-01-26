#include <cstdint>
#include <iostream>
#include <valarray>
#include <csignal>
#include "MockLoader.hpp"
#include "Net.hpp"
#include "Executor.hpp"
#include "NetParameter.hpp"
#include "QnnTypes.h"
#include "Types.hpp"
#include "express/Express.hpp"
#include "tokenizers/BPE/Bpe.hpp"
#include "backends/QNN/QNNBackend.hpp"
#include "memory/SystemMemoryManager.hpp"
#include "qnn_wrapper.hpp"
using namespace mllm;

// int main() {
//     BackendConfig bnc;

//     shared_ptr<MemoryManager> mm = nullptr;
//     switch (bnc.memory) {
//     case BackendConfig::Memory_High:
//         mm = std::make_shared<SystemMemoryManager>();
//         break;
//     default:
//         mm = std::make_shared<SystemMemoryManager>();
//         break;
//     }

//     QNNBackend *qbn = new QNNBackend(mm);

//     // build graph
//     std::cout << "build graph" << std::endl;
//     // graph add node
//     uint32_t dimensions[] = {1, 2, 2, 2};
//     float data[] = {-3, -2, -1, 0, 1, 2, 3, 4};
//     qbn->modelAddTensor("x", // Node Name
//                         (Qnn_Tensor_t){
//                             .version = QNN_TENSOR_VERSION_1,
//                             {.v1 = {
//                                  .id = 0,
//                                  .name = "x",
//                                  .type = QNN_TENSOR_TYPE_APP_WRITE,
//                                  .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
//                                  .dataType = QNN_DATATYPE_FLOAT_16,
//                                  .quantizeParams = {QNN_DEFINITION_UNDEFINED,
//                                                     QNN_QUANTIZATION_ENCODING_UNDEFINED,
//                                                     {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
//                                  .rank = 4,
//                                  .dimensions = dimensions,
//                                  .memType = QNN_TENSORMEMTYPE_RAW,
//                                  {.clientBuf = {.data = nullptr,
//                                                 .dataSize = 0}}}}});

//     __fp16 output_data[8];
//     vector<Qnn_Tensor_t> outputs = {
//         (Qnn_Tensor_t){
//             .version = QNN_TENSOR_VERSION_1,
//             {.v1 = {
//                  .id = 0,
//                  .name = "silu-output",
//                  .type = QNN_TENSOR_TYPE_APP_READ,
//                  .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
//                  .dataType = QNN_DATATYPE_FLOAT_16,
//                  .quantizeParams = {QNN_DEFINITION_UNDEFINED,
//                                     QNN_QUANTIZATION_ENCODING_UNDEFINED,
//                                     {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
//                  .rank = 4,
//                  .dimensions = dimensions,
//                  .memType = QNN_TENSORMEMTYPE_RAW,
//                  {.clientBuf = {.data = nullptr,
//                                 .dataSize = 0}}}}}};

//     qbn->graphAddNode("qnn-silu", "SiLU", {"x"}, outputs, {}, "LLaMAPackage");
    
//     // graph compile
//     std::cout << "graph compile" << std::endl;
//     qbn->graphFinilize();

//     // build input and outputs
//     std::map<std::string, std::vector<uint8_t*>> inputBufferMap;
//     std::vector<uint8_t*> inputBuffers;
//     inputBuffers.push_back((uint8_t*)data);
//     inputBufferMap.insert(std::make_pair("graph", inputBuffers));

//     std::map<std::string, std::vector<uint8_t*>> outputBufferMap;
//     std::vector<uint8_t*> outputBuffers;
//     outputBuffers.push_back((uint8_t*)output_data);
//     outputBufferMap.insert(std::make_pair("graph", outputBuffers));

//     // graph run
//     std::cout << "graph run" << std::endl;
//     qbn->graphExecute(inputBufferMap, outputBufferMap);

//     for(int i=0; i<8; i++) 
//         std::cout << output_data[i] << std::endl;

//     delete qbn;
// }



NetTensor * KVCache(Context *ctx,  uint32_t hidden_dim, uint32_t ffn_hidden_dim) {

    auto *i = _Input(ctx);
    auto *z = _KVCache(ctx, {i}, true,  "i_cache");
    z = _SiLU(ctx, {z}, "silu");

    return z;
}


template <typename Dtype>
void fullTensor(shared_ptr<Tensor> input_tensor, Net net, vector<int> shape, Dtype value) {
    input_tensor->setBackend(net.backends()[BackendType::MLLM_QNN].get());
    input_tensor->reshape(shape);
    input_tensor->alloc();
    input_tensor->fullData<Dtype>(value);
}

template <typename Dtype>
void seqTensor(shared_ptr<Tensor> input_tensor, Net &net, vector<int> shape, Dtype begin_value) {
    input_tensor->setBackend(net.backends()[BackendType::MLLM_CPU].get());
    input_tensor->reshape(shape);
    input_tensor->alloc();
    input_tensor->fullData<float>(1);

    int batch = shape[0];
    int n1 = shape[1];
    int n2 = shape[2];
    int n3 = shape[3];
    for (int n = 0; n < batch; n++) {
        for (int c = 0; c < n1; c++) {
            for (int h = 0; h < n2; h++) {
                for (int w = 0; w < n3; w++) {
                    input_tensor->setDataAt<Dtype>(n, c, h, w, begin_value + w%15);
                }
            }
        }
    }
}

int main() {

    int vocab_size = 32000;
    int hidden_dim = 4096;
    int ffn_hidden_dim = 11008;
    int mutil_head_size = 32;

    std::unique_ptr<Context> c_ptr(new Context());
    auto *c = c_ptr.get();

    KVCache(c, hidden_dim, ffn_hidden_dim);

    BackendConfig bn;
    Net net(c->sub_param_, bn);
    net.convert(c->sub_param_, MLLM_QNN);
    std::cout << "convert done" << std::endl;

    MockLoader loader("");
    Executor ex(&loader);
    shared_ptr<Tensor> input = std::make_shared<Tensor>();

    // 1 batch seqence length embedding
    // seqTensor(input, net, {1, 1, 1, hidden_dim}, -7.23f);
    fullTensor(input, net, {1, 1, 1, hidden_dim}, 2.f);

    ex.execute(&net, input);
    // ex.perf();
    auto result = ex.result();
    result[0]->printData<float>();
}