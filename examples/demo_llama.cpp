//
// Created by Rongjie Yi on 2024/1/26 0026.
//

#include <iostream>
#include <valarray>
#include <csignal>
#include "cmdline.h"
#include "express/Layer.hpp"
#include "tokenizers/BPE/Bpe.hpp"
#include "Module.hpp"

using namespace mllm;

Tensor Input(string name, int batch, int head, int seq, int dim, BackendType type = MLLM_CPU) {
    Tensor tensor1(batch, head, seq, dim, Module::backends[type], true);
    tensor1.setName(name);
    tensor1.status() = TENSOR_STATIC_INIT;
    tensor1.setTtype(INPUT_TENSOR);
    tensor1.fullData<float>(1.0);
    return tensor1;
}


class SampleModule final: public Module {
    SiLU silu = SiLU( "silu1");
    Softmax softmax = Softmax(DIMENSION, "softmax1");

    vector<Tensor> Forward(vector<Tensor> inputs) override {
        auto tensor1 = inputs[0]*5;
        auto tensor2 = tensor1 + inputs[0];
        tensor2 = tensor2.view(-1, 5, -1, 1);
        tensor2 = silu(tensor2);
        tensor2 = softmax(tensor2);
        return {tensor2};
    }
};
class subMod final: public Module {
    SampleModule mode = SampleModule();

    vector<Tensor> Forward(vector<Tensor> inputs) override {
        return  mode(inputs);
    }
};
int main(int argc, char **argv) {
    cmdline::parser cmdParser;
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "../vocab/llama_vocab.mllm");
    cmdParser.add<string>("model", 'm', "specify mllm model path", false, "../models/llama-2-7b-chat-q4_k.mllm");
    cmdParser.add<int>("limits", 'l', "max KV cache size", false, 400);
    cmdParser.add<int>("thread", 't', "num of threads", false, 4);
    cmdParser.parse_check(argc, argv);

    string vocab_path = cmdParser.get<string>("vocab");
    string model_path = cmdParser.get<string>("model");
    int tokens_limit = cmdParser.get<int>("limits");
    int thread_num = cmdParser.get<int>("thread");

    Module::initBackend(MLLM_CPU);


    auto tokenizer = BPETokenizer(vocab_path);


    auto tensor1 = Input("input", 1, 1, 1, 5, MLLM_CPU);


    auto model = subMod();
    subMod::initLoader(model_path);
    auto result = model({tensor1});
    result[0].printData<float>();

    return 0;

}