#include <cstdint>
#include <iostream>
#include <valarray>
#include <csignal>
#include "MockLoader.hpp"
#include "Types.hpp"
#include "backends/QNN/QNNOptNet.hpp"
#include "cmdline.h"
#include "Net.hpp"
#include "Executor.hpp"
#include "express/Express.hpp"
#include "tokenizers/BPE/Bpe.hpp"
#include "backends/QNN/QNNNet.hpp"
#include "backends/QNN/QNNExecutor.hpp"
#include "TestNet.hpp"

using namespace mllm;



template <typename Dtype>
void fullTensor(shared_ptr<Tensor> input_tensor, Net net, vector<int> shape, Dtype value) {
    input_tensor->setBackend(net.backends()[BackendType::MLLM_CPU].get());
    input_tensor->setCtype(ChlType::BSHD);
    input_tensor->setDtype(MLLM_TYPE_I8);
    input_tensor->reshape(shape[0], shape[1], shape[2], shape[3]);
    input_tensor->alloc();
    input_tensor->fullData<Dtype>(value);
}

int main(int argc, char **argv) {
    cmdline::parser cmdParser;
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "./vocab/vocab_opt_6.7b.mllm");
    cmdParser.add<string>("model", 'm', "specify mllm model path", false, "./models/opt-1.3b-sq_nohead.mllm");
    cmdParser.add<int>("limits", 'l', "max KV cache size", false, 400);
    // cmdParser.add<int>("thread", 't', "num of threads", false, 4);
    cmdParser.add<int>("seq", 's', "num of threads", false, 1);
    cmdParser.add<int>("head", 'h', "num of heads", false, 32);
    cmdParser.add<int>("type", 't', "type of test", false, 1);
    cmdParser.parse_check(argc, argv);

    string vocab_path = cmdParser.get<string>("vocab");
    string model_path = cmdParser.get<string>("model");
    int tokens_limit = cmdParser.get<int>("limits");
    // int thread_num = cmdParser.get<int>("thread");
    int seqLength = cmdParser.get<int>("seq");
    int head_num = cmdParser.get<int>("head");
    int type = cmdParser.get<int>("type");

    std::unique_ptr<Context> c_ptr(new Context());
    auto *c = c_ptr.get();

    auto *i = _Input(c);
    i = _MatmulINT8({i, i}, false, true, "model.decoder.layers.0.fc2");

    BackendConfig bn;
    Net net(bn);
    net.convert(c->sub_param_, BackendType::MLLM_CPU);

    // ParamLoader param_loader(model_path);
    MockLoader param_loader(model_path);
    Executor ex(&param_loader);
    ex.setup(&net);

    shared_ptr<Tensor> input = std::make_shared<Tensor>();
    fullTensor(input, net, {1, 1, 1, 32}, (uint8_t)2);
    uint8_t *data = input->hostPtr<uint8_t>();
    for (int i = 0; i < 16; i++) {
        std::cout << (int)data[i] << " ";
    }

    ex.run(&net, {input});

    auto result = ex.result();
    result[0]->printData<float>();

    ex.perf();


    return 0;
}
