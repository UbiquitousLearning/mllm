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
    SiLU silu = SiLU( "silu"+std::to_string(Module::listIdx));
    Softmax softmax = Softmax(DIMENSION, "softmax"+std::to_string(Module::listIdx));

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
    // SampleModule mode = SampleModule();
    vector<SampleModule> modules = List<SampleModule>(1);

    vector<Tensor> Forward(vector<Tensor> inputs) override {
        return  modules[0](inputs);
    }
};





class LLaMAConfig {
public:
    int vocab_size = 32000;
    int in_hidden_dim = 4096;
    int head_size = 32;
    int hidden_size = in_hidden_dim/head_size;
    int mlp_hidden = 11008;
};

class LLaMAAttention final: public Module, public LLaMAConfig {
    Linear q_proj = Linear(in_hidden_dim, head_size*hidden_size, false, "q"+std::to_string(Module::listIdx));
    Linear k_proj = Linear(in_hidden_dim, head_size*hidden_size, false,"k"+std::to_string(Module::listIdx));
    Linear v_proj = Linear(in_hidden_dim, head_size*hidden_size,false, "v"+std::to_string(Module::listIdx));
    Linear o_proj = Linear(head_size*hidden_size, in_hidden_dim, false,"o"+std::to_string(Module::listIdx));
    RoPE q_rope = RoPE( LLAMAROPE, "q_rope"+std::to_string(Module::listIdx));
    RoPE k_rope = RoPE( LLAMAROPE, "k_rope"+std::to_string(Module::listIdx));
    KVCache k_cache = KVCache(400, "k_cache"+std::to_string(Module::listIdx));
    KVCache v_cache = KVCache(400, "v_cache"+std::to_string(Module::listIdx));
    Matmul qk_mm = Matmul(false, true, "qk_mm"+std::to_string(Module::listIdx));
    Matmul qkv_mm = Matmul(false, false, "qkv_mm"+std::to_string(Module::listIdx));
    Causalmask mask = Causalmask("mask"+std::to_string(Module::listIdx));
    Softmax softmax = Softmax(DIMENSION, "softmax"+std::to_string(Module::listIdx));

    vector<Tensor> Forward(vector<Tensor> inputs) override {
        auto q = q_proj(inputs[0]);
        auto k = k_proj(inputs[1]);
        auto v = v_proj(inputs[2]);
        q = q.view(-1, head_size, -1, hidden_size);
        k = k.view(-1, head_size, -1, hidden_size);
        v = v.view(-1, head_size, -1, hidden_size);
        q = q_rope(q);
        k = k_rope(k);
        k = k_cache(k);
        v = v_cache(v);
        auto qk = qk_mm(q, k);
        qk = qk / std::sqrt(hidden_size);
        qk = softmax(qk);
        auto o = qkv_mm(qk, v);
        o = o.view(-1, 1, -1, hidden_size * head_size);
        o = o_proj(o);
        return {o};
    }
};

class LLaMAMLP final: public Module, public LLaMAConfig {
    Linear w1 = Linear(in_hidden_dim, mlp_hidden, false, "w1"+std::to_string(Module::listIdx));
    SiLU silu = SiLU( "silu"+std::to_string(Module::listIdx));
    Linear w3 = Linear(in_hidden_dim, mlp_hidden, false, "w3"+std::to_string(Module::listIdx));
    Linear w2 = Linear(mlp_hidden, in_hidden_dim, false, "w2"+std::to_string(Module::listIdx));

    vector<Tensor> Forward(vector<Tensor> inputs) override {
        auto x = w1(inputs[0]);
        x = silu(x);
        auto y = w3(inputs[0]);
        x = x * y;
        x = w2(x);
        return {x};
    }
};

class LLaMABlock final: public Module, public LLaMAConfig {
    LLaMAAttention attention = LLaMAAttention();
    LLaMAMLP mlp = LLaMAMLP();
    RMSNorm norm1 = RMSNorm(in_hidden_dim, 1e-6,"norm1"+std::to_string(Module::listIdx));
    RMSNorm norm2 = RMSNorm(in_hidden_dim, 1e-6,"norm2"+std::to_string(Module::listIdx));

    vector<Tensor> Forward(vector<Tensor> inputs) override {
        auto x_ = norm1(inputs[0]);
        auto x = attention({x_, x_, x_});
        x_ = x[0] + inputs[0];
        x_ = norm2(x_);
        x = mlp({x_});
        x_ = x[0] + x_;
        return {x_};
    }
};

class LLaMAModel final: public Module, public LLaMAConfig {
    Embedding embedding = Embedding(vocab_size, in_hidden_dim, "embedding");
    vector<LLaMABlock> blocks = List<LLaMABlock>(32);
    Linear norm = Linear(in_hidden_dim, vocab_size, false, "norm");

    vector<Tensor> Forward(vector<Tensor> inputs) override {
        auto x = embedding(inputs[0]);
        for (auto &block : blocks) {
            x = block({x})[0];
        }
        x = norm(x);
        return {x};
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


    // auto tensor1 = Input("input", 1, 1, 1, 5, MLLM_CPU);

    Tensor tensor1(1, 1, 1, 5, Module::backends[MLLM_CPU], true);
    tensor1.setName("input");
    tensor1.status() = TENSOR_STATIC_INIT;
    tensor1.setTtype(INPUT_TENSOR);
    tensor1.fullData<float>(1.0);


    auto llama = LLaMAModel();

    auto model = subMod();
    subMod::initLoader(model_path);
    auto result = model({tensor1});
    result[0].printData<float>();

    return 0;

}