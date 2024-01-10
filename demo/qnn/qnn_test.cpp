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

void BuildModel(Context *ctx) {
    auto *i = _Input(ctx);
    auto *q = _Linear(ctx, {i}, 4, 4, false, "layers." + std::to_string(0) + ".attention.wq");
    auto *k = _Linear(ctx, {q}, 4, 4, false, "layers." + std::to_string(0) + ".wq");
    _Matmul(ctx, {q, k}, false, true);
}

template <typename Dtype>
void fullTensor(shared_ptr<Tensor> input_tensor, Net net, vector<int> shape, Dtype value) {
    input_tensor->setBackend(net.backends()[BackendType::MLLM_QNN].get());
    input_tensor->reshape(shape);
    input_tensor->alloc();
    input_tensor->fullData<Dtype>(value);
}

int main() {
    // BackendConfig bnc;

    // shared_ptr<MemoryManager> mm = nullptr;
    // switch (bnc.memory) {
    // case BackendConfig::Memory_High:
    //     mm = std::make_shared<SystemMemoryManager>();
    //     break;
    // default:
    //     mm = std::make_shared<SystemMemoryManager>();
    //     break;
    // }

    // QNNBackend *qbn = new QNNBackend(mm);

    // // build graph
    // std::cout << "build graph" << std::endl;
    // testSilu(qbn);
    // // graph compile
    // std::cout << "graph compile" << std::endl;
    // qbn->graphFinilize();
    // // graph run
    // std::cout << "graph run" << std::endl;
    // qbn->graphExecute();

    // delete qbn;

    std::unique_ptr<Context> c_ptr(new Context());
    auto *c = c_ptr.get();

    BuildModel(c);

    BackendConfig bn;
    Net net(c->sub_param_, bn);
    net.convert(c->sub_param_, MLLM_QNN);
    std::cout << "convert done" << std::endl;

    MockLoader loader("");
    Executor ex(&loader);
    shared_ptr<Tensor> input = std::make_shared<Tensor>();
    fullTensor(input, net, {1, 1, 2, 4}, 2.f);

    ex.execute(&net, input);
    auto result = ex.result();
    result[0]->printData<float>();
}