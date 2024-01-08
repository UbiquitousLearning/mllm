//
// Created by 咸的鱼 on 2023/12/21.
//

#ifndef MODELING_FUYU_HPP
#define MODELING_FUYU_HPP
#include <express/Express.hpp>
#include "math.h"
#include "helper.hpp"

inline NetTensor *Attention_Fuyu(Context *ctx, NetTensor * x, int embedding_size, int hidden_size, int head_size, string name){
    x =_Linear( {x}, embedding_size, hidden_size * head_size * 3, true, name + ".query_key_value");
    auto skv = _Split( {x}, 3, Chl::D_HD, head_size, name + ".split");
    auto *q = skv[0];
    auto *k = skv[1];
    auto *v = skv[2];
    q = _LayerNorm( {q}, hidden_size, true, 1e-6, name + ".q_layernorm");
    k = _LayerNorm( {k}, hidden_size, true, 1e-6, name + ".k_layernorm");
    q = _RoPE( {q}, 3, name + ".q_rope");
    k = _RoPE( {k}, 3, name + ".k_rope");
    k = _KVCache( {k}, name + ".k_cache");
    v = _KVCache( {v}, name + ".v_cache");
    auto *qk = _Matmul( {q, k}, false, true, name + ".qk");
    qk = _Scale( {qk}, 1.0F / std::sqrt(head_size), 0.0F, false, name + ".scale");
    qk = _Causalmask( {qk}, name + ".mask");
    qk = _Softmax( {qk}, DIMENSION, name + ".softmax");
    auto *o = _Matmul( {qk, v}, false, false, name + ".qkv");
    o = o->view(-1, 1, -1, hidden_size * head_size);
    o = _Linear( {o}, hidden_size * head_size, embedding_size, true, name + ".dense");
    return o;
}

inline NetTensor *MLP_Fuyu(Context *ctx, NetTensor * i, int hidden_dim, int ffn_hidden_dim, string name){
    auto *x = _Linear( {i}, hidden_dim, ffn_hidden_dim, true, name+".dense_h_to_4h");
    x = _ReLUSquaredActivation( {x}, name+".relu2");
    x = _Linear( {x}, ffn_hidden_dim, hidden_dim, true, name+".dense_4h_to_h");
    return x;
}

inline NetTensor *Persimmon(Context* c, NetTensor * i, int hidden_dim= 4096, int ffn_hidden_dim = 4096*4, int mutil_head_size = 64, string name = "language_model.model"){
    // loop
    for(int layer=0; layer<36; ++layer) {
        auto *x = _LayerNorm( {i},  hidden_dim, true, 1e-6, name + (string)".layers."+std::to_string(layer)+".input_layernorm");
        x = Attention_Fuyu(c, x, hidden_dim, hidden_dim / mutil_head_size, mutil_head_size, name + (string)".layers."+std::to_string(layer)+".self_attn");
        i = _Add( {x, i}, name + (string)".layers."+std::to_string(layer)+".add_attn");
        x = _LayerNorm( {i},  hidden_dim, true, 1e-6, name + (string)".layers."+std::to_string(layer)+".post_attention_layernorm");
        x = MLP_Fuyu(c, x, hidden_dim, ffn_hidden_dim, name + (string)".layers."+std::to_string(layer) +".mlp");
        i = _Add( {x, i}, name + (string)".layers."+std::to_string(layer)+".add_mlp");
        _SubgraphBegin(c);
    }
    // end loop
    i = _LayerNorm( {i},  hidden_dim, true, 1e-6, name + (string)".final_layernorm");
    return i;
}

inline void Fuyu(Context* c, int vocab_size= 262144, int patch_size = 30, int cnl_size = 3,int hidden_dim= 4096, int ffn_hidden_dim = 4096*4, int mutil_head_size = 32){
    auto *i = _Input(c, {}, "input_ids");
    i = _Embedding( {i}, vocab_size, hidden_dim, (string)"language_model.model.embed_tokens");
    auto *p = _Input(c, {}, "image_patches");
    p = _Linear( {p}, patch_size*patch_size*cnl_size, hidden_dim, true, "vision_embed_tokens");
    auto *id = _Input(c, {}, "image_patches_indices");
    i = _Gather( {i, p, id}, "gather");
    i = Persimmon(c, i, hidden_dim, ffn_hidden_dim, mutil_head_size, "language_model.model");
    i = _Linear( {i}, hidden_dim, vocab_size, false, "language_model.lm_head");
}
unsigned int postProcessing_Fuyu(shared_ptr<Tensor> result, shared_ptr<Tensor>& out_result){
    CHECK_EQ(result->batch(), 1);
    CHECK_EQ(result->head(), 1);
    out_result->reshape(1, 1, 1, 1);
    out_result->alloc();
    vector<float> scores;
    for (int i = 0; i < result->dimension(); ++i) {
        auto value = result->dataAt<float>(0, 0, result->sequence()-1, i);
        scores.push_back(value);
    }
    auto token_idx =  argmax(scores);
    out_result->setDataAt<float>(0, 0, 0, 0, token_idx);
    return token_idx;
}
#endif //MODELING_FUYU_HPP
