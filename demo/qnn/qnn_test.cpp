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
    qbn->init();

    qbn->release();
    delete  qbn;

}