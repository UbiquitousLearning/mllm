#include <iostream>
#include <valarray>
#include <csignal>
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

void Attention(Context *ctx) {
    auto *i = _Input(ctx);
    i = _Add(ctx, {i, i});
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
    testSilu(qbn);
    // graph compile
    std::cout << "graph compile" << std::endl;
    qbn->graphFinilize();
    // graph run
    std::cout << "graph run" << std::endl;
    qbn->graphExecute();

    delete qbn;

    std::unique_ptr<Context> c_ptr(new Context());
    auto *c = c_ptr.get();

    BackendConfig bn;
    Net net(c->sub_param_, bn);
    net.convert(c->sub_param_, MLLM_QNN);

    Executor ex;
    shared_ptr<Tensor> input = std::make_shared<Tensor>();
    ex.execute(&net, input);
}