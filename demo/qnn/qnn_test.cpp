#include <iostream>
#include <valarray>
#include <csignal>
#include "Net.hpp"
#include "Executor.hpp"
#include "NetParameter.hpp"
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

    // qbn->init(); // TODO: use part of the init function to create a graph handle
    // backend init and create a graph handle
    std::cout << "backend init" << std::endl;
    qbn->graphInitialize();
    // create an Add op for test
    std::cout << "create an Add op for test" << std::endl;
    Op add = QNNAdd(qbn, "add");
    // build graph
    std::cout << "build graph" << std::endl;
    qbn->graphAddNode(add);
    // graph compile
    std::cout << "graph compile" << std::endl;
    qbn->graphFinilize();
    // graph run
    std::cout << "graph run" << std::endl;
    qbn->graphExecute();

    qbn->release();
    delete  qbn;

}