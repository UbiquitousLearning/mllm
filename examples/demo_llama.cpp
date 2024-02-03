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

Tensor Tokens2Input( vector<token_id_t> tokens_id, string name= "input", BackendType type = MLLM_CPU) {
    Tensor tensor1(1, 1, tokens_id.size(), 1, Module::backends[MLLM_CPU], true);
    tensor1.setName(name);
    tensor1.status() = TENSOR_STATIC_INIT;
    tensor1.setTtype(INPUT_TENSOR);
    // tensor1.fullData<float>(1.0);
    for (int idx = 0; idx < tokens_id.size(); ++idx) {
        tensor1.setDataAt<float>(0, 0, idx, 0, tokens_id[idx]);
    }
    return tensor1;
}







class LLaMAConfig {
public:
    int vocab_size = 32000;
    int in_hidden_dim = 4096;
    int head_size = 32;
    int hidden_size = in_hidden_dim/head_size;
    int mlp_hidden = 11008;
    int block_num = 32;
    int cache_limit = 200;

    std::string base_name = "layers."+std::to_string(Module::listIdx)+ ".";
    std::string attn_base_name = base_name+ "attention.";
    std::string ffn_base_name = base_name+ "feed_forward.";
};

class LLaMAAttention final: public Module, public LLaMAConfig {
    Linear q_proj = Linear(in_hidden_dim, head_size*hidden_size, false, attn_base_name+"wq");
    Linear k_proj = Linear(in_hidden_dim, head_size*hidden_size, false,attn_base_name+"wk");
    Linear v_proj = Linear(in_hidden_dim, head_size*hidden_size,false, attn_base_name+"wv");
    Linear o_proj = Linear(head_size*hidden_size, in_hidden_dim, false,attn_base_name+"wo");
    RoPE q_rope = RoPE( LLAMAROPE, attn_base_name+"q_rope");
    RoPE k_rope = RoPE( LLAMAROPE, attn_base_name+"k_rope");
    KVCache k_cache = KVCache(cache_limit, attn_base_name+"k_cache");
    KVCache v_cache = KVCache(cache_limit, attn_base_name+"v_cache");
    Matmul qk_mm = Matmul(false, true, attn_base_name+"qk_mm");
    Matmul qkv_mm = Matmul(false, false, attn_base_name+"qkv_mm");
    Causalmask mask = Causalmask(attn_base_name+"mask");
    Softmax softmax = Softmax(DIMENSION, attn_base_name+"softmax");

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
        qk = mask(qk);
        qk = softmax(qk);
        auto o = qkv_mm(qk, v);
        o = o.view(-1, 1, -1, hidden_size * head_size);
        o = o_proj(o);
        return {o};
    }
};

class LLaMAMLP final: public Module, public LLaMAConfig {
    Linear w1 = Linear(in_hidden_dim, mlp_hidden, false, ffn_base_name+"w1");
    SiLU silu = SiLU( ffn_base_name+"silu");
    Linear w3 = Linear(in_hidden_dim, mlp_hidden, false, ffn_base_name+"w3");
    Linear w2 = Linear(mlp_hidden, in_hidden_dim, false, ffn_base_name+"w2");

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
    RMSNorm norm1 = RMSNorm(in_hidden_dim, 1e-6, base_name+"attention_norm");
    RMSNorm norm2 = RMSNorm(in_hidden_dim, 1e-6, base_name+"ffn_norm");

    vector<Tensor> Forward(vector<Tensor> inputs) override {
        auto x = norm1(inputs[0]);
        x = attention({x, x, x})[0];
        auto tmp = x + inputs[0];
        x = norm2(tmp);
        x = mlp({x})[0];
        x = x + tmp;
        return {x};
    }
};

class LLaMAModel final: public Module, public LLaMAConfig {
    Embedding embedding = Embedding(vocab_size, in_hidden_dim, "tok_embeddings");
    vector<LLaMABlock> blocks = List<LLaMABlock>(block_num);
    RMSNorm norm = RMSNorm(in_hidden_dim, 1e-6, "norm");
    Linear mlp_head = Linear(in_hidden_dim, vocab_size, false, "output");

    vector<Tensor> Forward(vector<Tensor> inputs) override {
        auto x = embedding(inputs[0]);
        for (auto &block : blocks) {
            x = block({x})[0];
        }
        x = norm(x);
        x = mlp_head(x);
        return {x};
    }
};


unsigned int argmax(const std::vector<float> &scores) {
    if (scores.empty()) {
        throw std::invalid_argument("Input vector is empty");
    }
    unsigned int maxIndex = 0;
    float maxValue = scores[0];
    for (size_t i = 1; i < scores.size(); ++i) {
        if (scores[i] > maxValue) {
            maxIndex = i;
            maxValue = scores[i];
        }
    }
    return maxIndex;
}
unsigned int postProcessing(Tensor& result, Tensor &out_result) {
    assert(result.batch() == 1);
    assert(result.head() == 1);
    out_result.reshape(1, 1, 1, 1);
    out_result.alloc();
    vector<float> scores;
    for (int i = 0; i < result.dimension(); ++i) {
        auto value = result.dataAt<float>(0, 0, result.sequence() - 1, i);
        scores.push_back(value);
    }
    auto token_idx = argmax(scores);
    out_result.setDataAt<float>(0, 0, 0, 0, token_idx);
    return token_idx;
}

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


    auto model = LLaMAModel();
    LLaMAModel::initLoader(model_path);



    vector<string> in_strs = {
        " Hello, who are you?",
        " What can you do?",
        "Please introduce Beijing University of Posts and Telecommunications."};

    for (int str_i = 0; str_i < in_strs.size(); ++str_i) {
        auto in_str = in_strs[str_i];

        if (in_str[0] != ' ') {
            in_str = ' ' + in_str;
        }
        auto tokens_id = vector<token_id_t>();
        tokenizer.tokenize(in_str, tokens_id, true);
        if (str_i > 0) {
            tokens_id[0] = 13;
        }
        auto tensor1 = Tokens2Input(tokens_id);

        std::cout << "[Q] " << in_str << std::endl;
        std::cout << "[A] " << std::flush;
        for (int step = 0; step < 100; step++) {
            auto result = model({tensor1});
            auto token_idx = postProcessing(result[0], tensor1);
            if (token_idx == 2) { // "</s>"
                break;
            }
            auto out_token = tokenizer.detokenize({token_idx});
            std::cout << out_token << std::flush;
        }
        printf("\n");
    }

    return 0;

}