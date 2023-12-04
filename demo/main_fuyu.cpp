#include <vector>
#include <iostream>
#include <valarray>
#include <csignal>
#include "Net.hpp"
#include "Executor.hpp"
#include "NetParameter.hpp"
#include "express/Express.hpp"
#include "tokenizers/BPE/Bpe.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "imageHelper/stb_image.h"

using namespace std;

void token2Tensor(shared_ptr<Tensor> input_tensor, Net &net, vector<token_id_t> tokens) {
    input_tensor->setBackend(net.backends()[BackendType::MLLM_CPU].get());
    input_tensor->reshape(1, 1, static_cast<int>(tokens.size()), 1);
    input_tensor->alloc();
    input_tensor->fullData<float>(1);
    for (int idx = 0; idx < tokens.size(); ++idx) {
        input_tensor->setDataAt<float>(0, 0, idx, 0, tokens[idx]);
    }
}

NetTensor *Attention(Context *ctx, NetTensor * x, int embedding_size, int hidden_size, int head_size, string name){
    x =_Linear(ctx, {x}, embedding_size, hidden_size * head_size * 3, false, name + ".query_key_value");
    auto skv = _Split(ctx, {x}, 3, Chl::DIMENSION, name + ".split");
    auto *q = _View(ctx, {skv[0]}, {-1, head_size, -1, -1}, {BATCH, DIMENSION, SEQUENCE, DIMENSION}, name + ".q_view");
    auto *k = _View(ctx, {skv[1]}, {-1, head_size, -1, -1}, {BATCH, DIMENSION, SEQUENCE, DIMENSION}, name + ".k_view");
    auto *v = _View(ctx, {skv[2]}, {-1, head_size, -1, -1}, {BATCH, DIMENSION, SEQUENCE, DIMENSION}, name + ".v_view");
    q = _LayerNorm(ctx, {q}, true, name + ".q_layernorm");
    k = _LayerNorm(ctx, {k}, true, name + ".k_layernorm");
    q = _RoPE(ctx, {q}, name + ".q_rope");
    k = _RoPE(ctx, {k}, name + ".k_rope");
    k = _KVCache(ctx, {k}, true, name + ".k_cache");
    v = _KVCache(ctx, {v}, true, name + ".v_cache");
    auto *qk = _Matmul(ctx, {q, k}, false, true, name + ".qk");
    qk = _Scale(ctx, {qk}, 1.0F / std::sqrt(hidden_size), 0.0F, false, name + ".scale");
    qk = _Causalmask(ctx, {qk}, name + ".mask");
    qk = _Softmax(ctx, {qk}, SEQUENCE, name + ".softmax");
    auto *o = _Matmul(ctx, {qk, v}, false, false, name + ".qkv");
    o = _View(ctx, {o}, {-1, -1, -1, -1}, {BATCH, -1, SEQUENCE, HEAD+DIMENSION}, name + ".qkv_view");
    o = _Linear(ctx, {o}, hidden_size * head_size, embedding_size, false, name + ".dense");
    return o;
}
NetTensor *FFN(Context *ctx, NetTensor * i, int hidden_dim, int ffn_hidden_dim, string name){
    auto *x = _Linear(ctx, {i}, hidden_dim, ffn_hidden_dim, false, name+".dense_h_to_4h");
    x = _Linear(ctx, {x}, ffn_hidden_dim, hidden_dim, false, name+".dense_4h_to_h");
    return x;
}
void Persimmon(Context* c, int vocab_size= 32000, int hidden_dim= 4096, int ffn_hidden_dim = 4096*4, int mutil_head_size = 32){
    auto *i = _Input(c);
    i = _Embedding(c, {i}, vocab_size, hidden_dim, (string)"tok_embeddings");
    // loop
    for(int layer=0; layer<1; ++layer) {
//        auto *x = _LayerNorm(c, {i}, true, (string)"layers."+std::to_string(layer)+".post_attention_layernorm");
        auto *x = Attention(c, i, hidden_dim, hidden_dim / mutil_head_size, mutil_head_size, (string)"layers."+std::to_string(layer)+".self_attention");
        i = _Add(c, {x, i});
        x = _LayerNorm(c, {i}, true, (string)"layers."+std::to_string(layer)+".post_attention_layernorm");
        x = FFN(c, x, hidden_dim, ffn_hidden_dim, (string)"layers."+std::to_string(layer) +".feed_forward");
        i = _Add(c, {x, i});
        _SubgraphBegin(c);
    }
    // end loop
    i = _RMSNorm(c, {i}, (string)"norm");
    i = _Linear(c, {i}, hidden_dim, vocab_size, false, "output");
}
int main() {
    int width, height, channel;
    unsigned char *data = stbi_load("test.jpg", &width, &height, &channel, 0);
    if (data == nullptr) {
        cout << "load image failed" << endl;
        return -1;
    }
    cout << "width: " << width << " height: " << height << " channel: " << channel << endl;
    vector<float> data_f32(width * height * channel);
    for (int i = 0; i < width * height * channel; i++) {
        data_f32[i] = data[i] / 255.0;
    }
    stbi_image_free(data);

    auto tokenizer = BPETokenizer("../tools/convertor/vocab.mllm");
    auto tokens_id = vector<token_id_t>();
    string in_str = " I believe the meaning of life is";
    tokenizer.tokenize(in_str, tokens_id, true);


    int vocab_size = 32000;
    int hidden_dim = 4096;
    int ffn_hidden_dim = 4096*4;
    int mutil_head_size = 32;

    std::unique_ptr<Context> c_ptr(new Context());
    auto *c = c_ptr.get();
    Persimmon(c, vocab_size, hidden_dim, ffn_hidden_dim, mutil_head_size);

    BackendConfig bn;
    Net net(bn);
    net.convert(c->sub_param_);


    Executor ex(nullptr);
    shared_ptr<Tensor> input = std::make_shared<Tensor>();
    token2Tensor(input, net, tokens_id);

    std::cout << in_str << std::flush;
    ex.execute(&net, input);
    auto result = ex.result();

    // free memory
    for (auto *op : c->net_ops) {
        delete op;
    }
    for (auto *tensor : c->net_tensors) {
        delete tensor;
    }
    return 0;

    return 0;
}