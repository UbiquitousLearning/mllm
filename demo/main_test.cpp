#include <iostream>
#include <valarray>
#include <csignal>
#include "Net.hpp"
#include "Executor.hpp"
#include "NetParameter.hpp"
#include "express/Express.hpp"
#include "tokenizers/BPE/Bpe.hpp"
#include "backends/cpu/CPUBackend.hpp"
using namespace mllm;
// For Visualization and Debug
void display(NetParameter *net) {
    std::cout << "===NetParameter===" << std::endl;
    for (auto *op : net->net_ops) {
        std::cout << "===NetOP===" << std::endl;
        std::cout << "op->name:" << op->name << std::endl;
        std::cout << "op->type:" << op->type << std::endl;
        std::cout << "op input" << op->in.size() << std::endl;
        for (auto *input : op->in) {
            std::cout << "==Input==\ninput.name:" << input->name << std::endl;
            if (input->in != nullptr) {
                std::cout << "input op:" << input->in->name << std::endl;
            }
            std::cout << "input in subgraph:" << (input->subgraph == net) << std::endl;
            std::cout << std::endl;
        }
        std::cout << "op output" << op->out.size() << std::endl;
        for (auto *output : op->out) {
            std::cout << "output.name:" << output->name << std::endl;
            std::cout << "output op:" << output->out.size() << std::endl;
            if (!output->out.empty()) {
                std::cout << "output op:" << output->out[0]->name << std::endl;
            }
        }
        std::cout << std::endl;
    }
}
void display(Context *c) {
    for (auto sub : c->sub_param_) {
        display(&sub);
    }
}

void fullTensor(shared_ptr<Tensor> input_tensor, Net net, vector<int> shape, float value) {
    input_tensor->setBackend(net.backends()[BackendType::MLLM_CPU]);
    input_tensor->reshape(shape);
    input_tensor->alloc();
    input_tensor->fullData<float>(value);
}
void token2Tensor(shared_ptr<Tensor> input_tensor, Net net, vector<token_id_t> tokens) {
    input_tensor->setBackend(net.backends()[BackendType::MLLM_CPU]);
    input_tensor->reshape({1, 1, static_cast<int>(tokens.size()), 1});
    input_tensor->alloc();
    input_tensor->fullData<float>(1);
    for (int idx = 0; idx < tokens.size(); ++idx) {
        input_tensor->setDataAt<float>(0, 0, idx, 0, tokens[idx]);
    }
}
unsigned int argmax(const std::vector<float>& scores) {
    if(scores.empty()) {
        throw std::invalid_argument("Input vector is empty");
    }
    unsigned int maxIndex = 0;
    float maxValue = scores[0];
    for(size_t i = 1; i < scores.size(); ++i) {
        if(scores[i] > maxValue) {
            maxIndex = i;
            maxValue = scores[i];
        }
    }
    return maxIndex;
}
unsigned int postProcessing(shared_ptr<Tensor> result, shared_ptr<Tensor>& out_result){
    CHECK_EQ(result->shape(0), 1);
    CHECK_EQ(result->shape(1), 1);
    out_result->reshape({1, 1, 1, 1});
    out_result->alloc();
    vector<float> scores;
    for (int i = 0; i < result->shape(3); ++i) {
        auto value = result->dataAt<float>(0, 0, result->shape(2)-1, i);
        scores.push_back(value);
    }
    auto token_idx =  argmax(scores);
    out_result->setDataAt<float>(0, 0, 0, 0, token_idx);
    return token_idx;
}
NetTensor *Attention(Context *ctx, NetTensor * x, int embedding_size, int hidden_size, int head_size, string name){
    auto *q =_Linear(ctx, {x}, embedding_size, hidden_size * head_size, false, name + ".wq");
    auto *k =_Linear(ctx, {x}, embedding_size, hidden_size * head_size, false, name + ".wk");
    auto *v =_Linear(ctx, {x}, embedding_size, hidden_size * head_size, false, name + ".wv");
    q = _View(ctx, {q}, {-1, head_size, -1, -1}, {0, 3, 2, 3}, name + ".q_view");
    k = _View(ctx, {k}, {-1, head_size, -1, -1}, {0, 3, 2, 3}, name + ".k_view");
    v = _View(ctx, {v}, {-1, head_size, -1, -1}, {0, 3, 2, 3}, name + ".v_view");
    q = _RoPE(ctx, {q}, name + ".q_rope");
    k = _RoPE(ctx, {k}, name + ".k_rope");
    k = _KVCache(ctx, {k}, true, name + ".k_cache");
    v = _KVCache(ctx, {v}, false, name + ".v_cache");
    auto *qk = _Matmul(ctx, {q, k}, false, true, name + ".qk");
    qk = _Scale(ctx, {qk}, 1.0F / std::sqrt(hidden_size), 0.0F, false, name + ".scale");
    qk = _Causalmask(ctx, {qk}, name + ".mask");
    qk = _Softmax(ctx, {qk}, 3, name + ".softmax");
    auto *o = _Matmul(ctx, {qk, v}, false, true, name + ".qkv");
    o = _View(ctx, {o}, {-1, -1, -1, -1}, {0, -1, 2, 1+3}, name + ".qkv_view");
    o = _Linear(ctx, {o}, hidden_size * head_size, embedding_size, false, name + ".wo");
    return o;
}
NetTensor *FFN(Context *ctx, NetTensor * i, int hidden_dim, int ffn_hidden_dim, string name){
    auto *x = _Linear(ctx, {i}, hidden_dim, ffn_hidden_dim, false, name+".w1");
    x = _SiLU(ctx, {x});
    auto *y = _Linear(ctx, {i}, hidden_dim, ffn_hidden_dim, false, name+".w3");
    x = _Mul(ctx, {x, y});
    x = _Linear(ctx, {x}, ffn_hidden_dim, hidden_dim, false, name+".w2");
    return x;
}
void llama2(Context* c, int vocab_size= 32000, int hidden_dim= 4096, int ffn_hidden_dim = 11008, int mutil_head_size = 32){
    auto *i = _Input(c);
    i = _Embedding(c, {i}, vocab_size, hidden_dim, (string)"tok_embeddings");
    // loop
    for(int layer=0; layer<32; ++layer) {
        auto *x = _RMSNorm(c, {i}, (string)"layers."+std::to_string(layer)+".attention_norm");
        //x = _Attention(c, {x}, hidden_dim, hidden_dim / mutil_head_size, mutil_head_size, (string)"layers."+std::to_string(layer)+".attention");
        x = Attention(c, x, hidden_dim, hidden_dim / mutil_head_size, mutil_head_size, (string)"layers."+std::to_string(layer)+".attention");
        i = _Add(c, {x, i});
        x = _RMSNorm(c, {i}, (string)"layers."+std::to_string(layer)+".ffn_norm");
        x = FFN(c, x, hidden_dim, ffn_hidden_dim, (string)"layers."+std::to_string(layer) +".feed_forward");
        i = _Add(c, {x, i});
        _SubgraphBegin(c);
    }
    // end loop
    i = _RMSNorm(c, {i}, (string)"norm");
    i = _Linear(c, {i}, hidden_dim, vocab_size, false, "output");
}
int main() {
    auto tokenizer = BPETokenizer("../tools/convertor/vocab.mllm");
    auto tokens_id = vector<token_id_t>();
    // tokenizer.tokenize(string(" this is 🦙.cpp"), tokens_id, true);
    // tokenizer.tokenize(string(" 你所热爱的，就是你的生活"), tokens_id, true);
    string in_str = " I believe the meaning of life is";
//    string in_str = " Building a website can be done in 10 simple steps:\\nStep 1:";
    tokenizer.tokenize(in_str, tokens_id, true);
//    for (auto idx : tokens_id) {
//        std::cout << idx << ",";
//    }
//    std::cout << std::endl;
    int vocab_size = 32000;
    int hidden_dim = 4096;
    int ffn_hidden_dim = 11008;
    int mutil_head_size = 32;
    Context *c = new Context();
    llama2(c, vocab_size, hidden_dim, ffn_hidden_dim, mutil_head_size);

    BackendConfig bn;
    Net net(c->sub_param_, bn);
    net.convert();
    // net.Run();
//    ParamLoader param_loader("../models/llama-2-7b-fp32.mllm");
//    ParamLoader param_loader("../models/llama-2-7b-q4_0.mllm");
//    ParamLoader param_loader("../models/llama-2-7b-q4_k-64.mllm");
    ParamLoader param_loader("../models/llama-2-7b-q4_k.mllm");
    Executor ex(&net, &param_loader);
    // Executor ex(&net);
    shared_ptr<Tensor> input = std::make_shared<Tensor>();
    // fullTensor(input, net, {1, 1, 10, 1}, 1);
    //tokens_id = {tokens_id[0]};
    token2Tensor(input, net, tokens_id);

    std::cout << in_str << std::flush;
    for(int step = 0; step<64; step++) {
        ex.execute(input);
        auto result = ex.result();
        auto token_idx = postProcessing(result[0], input);
        auto out_token = tokenizer.detokenize({token_idx});
        std::cout << out_token << std::flush;
    }
    printf("\n");
    ex.perf();

//    shared_ptr<Tensor> input_2 = std::make_shared<Tensor>();
//    token2Tensor(input_2, net, {token_idx});out_result
//    ex.execute(input);
//    result = ex.result();
//    token_idx = postProcessing(result[0], input);
//    out_token = tokenizer.detokenize({token_idx});
//    std::cout<<"OUT TOKEN: "<<token_idx<<"|    "<< out_token << std::endl;

    return 0;
}
