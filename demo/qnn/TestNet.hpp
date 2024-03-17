#include "Net.hpp"
#include "express/Express.hpp"
NetTensor *AttentionTest(Context *c, NetTensor *x, int embedding_size, int hidden_size, int head_size, int cache_max, string name) {
    // x = _Quantize({x}, true, (string)name + ".x.quantize");
    auto *q = _LinearINT8({x}, embedding_size, hidden_size * head_size, false, name + ".q_proj");
    return q;
    auto *k = _LinearINT8({x}, embedding_size, hidden_size * head_size, false, name + ".k_proj");
    auto *v = _LinearINT8({x}, embedding_size, hidden_size * head_size, false, name + ".v_proj");
    q = q->view(-1, head_size, -1, hidden_size);
    k = k->view(-1, head_size, -1, hidden_size);
    v = v->view(-1, head_size, -1, hidden_size);
    auto *m = _MergeOutput({q, k, v}, name + ".qkv_merge");
    return m;
}
NetTensor *AttentionWithMatmulTest(Context *c, NetTensor *x, int embedding_size, int hidden_size, int head_size, int cache_max, string name) {
    // x = _Quantize({x}, true, (string)name + ".x.quantize");
    auto *q = _LinearINT8({x}, embedding_size, hidden_size * head_size, false, name + ".q_proj");
    return q;
    auto *k = _LinearINT8({x}, embedding_size, hidden_size * head_size, false, name + ".k_proj");
    auto *v = _LinearINT8({x}, embedding_size, hidden_size * head_size, false, name + ".v_proj");
    q = q->view(-1, head_size, -1, hidden_size);
    k = k->view(-1, head_size, -1, hidden_size);
    v = v->view(-1, head_size, -1, hidden_size);
    // ----------------
    auto *qk = _Matmul({q, k}, false, true, name + ".qk");
    qk = _Softmax({qk}, DIMENSION, name + ".softmax");
    auto *o = _Matmul({qk, v}, false, false, name + ".qkv");
    o = o->view(-1, 1, -1, hidden_size * head_size);
    o = _LinearINT8({o}, hidden_size * head_size, embedding_size, false, name + ".out_proj");
    return o;
}
NetTensor *FFNTest(Context *c, NetTensor *i, int hidden_dim, int ffn_hidden_dim, string name) { 
    auto *x = i;
    x = _LinearINT8({x}, hidden_dim, ffn_hidden_dim, false, name + ".fc1");
    x = _GELU({x}, name + ".gelu");
    x = _LinearINT8({x}, ffn_hidden_dim, hidden_dim, false, name + ".fc2");
    return x;
}

void linearTest2048(Context *c, int vocab_size = 32000, int hidden_dim = 4096, int ffn_hidden_dim = 11008, int mutil_head_size = 32, int cache_max = 200) {
    auto *i = _Input(c);
    i = _Embedding({i}, vocab_size, hidden_dim, (string) "model.decoder.embed_tokens");
    _SubgraphBegin(c);
    i = _Quantize({i}, true, "x.quantize");
    _LinearINT8({i}, 2048, 2048, false, "model.decoder.layers.0.fc1");
}
void linearTest11008(Context *c, int vocab_size = 32000, int hidden_dim = 4096, int ffn_hidden_dim = 11008, int mutil_head_size = 32, int cache_max = 200) {
    auto *i = _Input(c);
    i = _Embedding({i}, vocab_size, hidden_dim, (string) "model.decoder.embed_tokens");
    _SubgraphBegin(c);
    i = _Quantize({i}, true, "x.quantize");
    _LinearINT8({i}, 2048, 11008, false, "model.decoder.layers.0.fc1");
}
void linearTest4096(Context *c, int vocab_size = 32000, int hidden_dim = 4096, int ffn_hidden_dim = 11008, int mutil_head_size = 32, int cache_max = 200) {
    auto *i = _Input(c);
    i = _Embedding({i}, vocab_size, hidden_dim, (string) "model.decoder.embed_tokens");
    _SubgraphBegin(c);
    i = _Quantize({i}, true, "x.quantize");
    _LinearINT8({i}, hidden_dim, hidden_dim, false, "model.decoder.layers.0.fc1");
}
void linearTest409616384(Context *c, int vocab_size = 32000, int hidden_dim = 4096, int ffn_hidden_dim = 11008, int mutil_head_size = 32, int cache_max = 200) {
    auto *i = _Input(c);
    i = _Embedding({i}, vocab_size, hidden_dim, (string) "model.decoder.embed_tokens");
    _SubgraphBegin(c);
    i = _Quantize({i}, true, "x.quantize");
    _LinearINT8({i}, hidden_dim, ffn_hidden_dim, false, "model.decoder.layers.0.fc1");
}
void linearTest409611008(Context *c, int vocab_size = 32000, int hidden_dim = 4096, int ffn_hidden_dim = 11008, int mutil_head_size = 32, int cache_max = 200) {
    auto *i = _Input(c);
    i = _Embedding({i}, vocab_size, hidden_dim, (string) "model.decoder.embed_tokens");
    _SubgraphBegin(c);
    i = _Quantize({i}, true, "x.quantize");
    _LinearINT8({i}, hidden_dim, ffn_hidden_dim, false, "model.decoder.layers.0.fc1");
}
void attentionMinor(Context *c, int vocab_size = 32000, int hidden_dim = 4096, int ffn_hidden_dim = 11008, int mutil_head_size = 32, int cache_max = 200) {
    auto *i = _Input(c);
    i = _Embedding({i}, vocab_size, hidden_dim, (string) "model.decoder.embed_tokens");
    _SubgraphBegin(c);
    // loop

    i = _Quantize({i}, true, "x.quantize");
    i = AttentionTest(c, i, hidden_dim, hidden_dim / mutil_head_size, mutil_head_size, cache_max, (string) "model.decoder.layers." + std::to_string(0) + ".self_attn");
    return;
}
void attentionPlus(Context *c, int vocab_size = 32000, int hidden_dim = 4096, int ffn_hidden_dim = 11008, int mutil_head_size = 32, int cache_max = 200) {
    auto *i = _Input(c);
    i = _Embedding({i}, vocab_size, hidden_dim, (string) "model.decoder.embed_tokens");
    _SubgraphBegin(c);
    // loop
    i = _Quantize({i}, true, "x.quantize");
    i = AttentionWithMatmulTest(c, i, hidden_dim, hidden_dim / mutil_head_size, mutil_head_size, cache_max, (string) "model.decoder.layers." + std::to_string(0) + ".self_attn");
    return;
}
void ffnTest(Context *c, int vocab_size = 32000, int hidden_dim = 4096, int ffn_hidden_dim = 11008, int mutil_head_size = 32, int cache_max = 200){
    auto *i = _Input(c);
    i = _Embedding({i}, vocab_size, hidden_dim, (string) "model.decoder.embed_tokens");
    _SubgraphBegin(c);
    // loop
    i = _Quantize({i}, true, "x.quantize");
    i = FFNTest(c, i, hidden_dim, ffn_hidden_dim, (string) "model.decoder.layers." + std::to_string(0));
    return;
}
