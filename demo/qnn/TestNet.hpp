#include "Net.hpp"
#include "Types.hpp"
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
    i = _LinearINT8({i}, 2048, 2048, false, "model.decoder.layers.0.fc1");
    _SubgraphBegin(c, MLLM_CPU);
    i = _MatmulINT8({i, i}, false, true, "model.decoder.layers.0.fc2");
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

std::vector<NetTensor *> KVCacheTestCPUNPUAttention(Context *c, NetTensor *x, NetTensor *res, int embedding_size, int hidden_size, int head_size, int cache_max, string name) {
    // x = _Quantize({x}, true, (string)name + ".x.quantize");
    x = x->view(-1, 1, -1, hidden_size * head_size);
    auto *q = _LinearINT8({x}, embedding_size, hidden_size * head_size, false, name + ".q_proj");
    auto *k = _LinearINT8({x}, embedding_size, hidden_size * head_size, false, name + ".k_proj");
    auto *v = _LinearINT8({x}, embedding_size, hidden_size * head_size, false, name + ".v_proj");
    q = q->view(-1, head_size, -1, hidden_size);
    k = k->view(-1, head_size, -1, hidden_size);
    v = v->view(-1, head_size, -1, hidden_size);
    // q = _RoPE({q}, LLAMAROPE, name + ".q_rope");
    // k = _RoPE({k}, LLAMAROPE, name + ".k_rope");
    // k = _KVCache({k}, cache_max, name + ".k_cache");
    // v = _KVCache({v}, cache_max, name + ".v_cache");

    auto *m = _MergeOutput({q, k, v, res}, name + ".qkv_merge");

    // --------------------
    _SubgraphBegin(c, MLLM_CPU);
    // --------------------

    auto s = _SplitInput({m}, true, 4, name + ".qkv_split");

    q = s[0];
    k = s[1];
    v = s[2];
    res = s[3];

    k = _KVCache({k}, 1, false, cache_max, name + ".k_cache");
    return {k};
    v = _KVCache({v}, 1, false, cache_max, name + ".v_cache");

    auto *qk = _MatmulINT8({q, k}, false, true, name + ".qk");

    qk = *qk / std::sqrt(hidden_size);
    qk = _Causalmask({qk}, name + ".mask");
    qk = _Softmax({qk}, DIMENSION, name + ".softmax");

    auto *o = _MatmulINT8({qk, v}, false, false, name + ".qkv");

    o = _Quantize({o}, true, (string)name + ".out_proj.quantize");
    m = _MergeOutput({o, res}, name + ".or_merge");

    // --------------------
    _SubgraphBegin(c);
    // --------------------
    s = _SplitInput({m}, true, 2, name + ".or_split");

    o = s[0];
    res = s[1];

    o = o->view(-1, 1, -1, hidden_size * head_size);
    res = res->view(-1, 1, -1, hidden_size * head_size);
    o = _LinearINT8({o}, hidden_size * head_size, embedding_size, false, name + ".out_proj");
    o = _Dequantize({o}, true, (string)name + ".out_proj.dequantize");
    return {o, res};
}
void KVCacheTest(Context *c, int vocab_size = 32000, int hidden_dim = 4096, int ffn_hidden_dim = 11008, int mutil_head_size = 32, int cache_max = 200) {
    auto *i = _Input(c);
    i = _Embedding({i}, vocab_size, hidden_dim, (string) "model.decoder.embed_tokens");

    for (int layer = 0; layer < 1; ++layer) {
        if (layer != 0)
            _SubgraphBegin(c, MLLM_CPU);

        auto res = i;
        res = res->view(-1, mutil_head_size, -1, hidden_dim / mutil_head_size);
        i = _LayerNorm({i}, hidden_dim, true, 1e-5, (string) "model.decoder.layers." + std::to_string(layer) + ".self_attn_layer_norm");
        i = _Quantize({i}, true, (string) "model.decoder.layers." + std::to_string(layer) + ".self_attn.q_proj.quantize");
        i = i->view(-1, mutil_head_size, -1, hidden_dim / mutil_head_size);
        auto *m = _MergeOutput({i, res}, (string) "model.decoder.layers." + std::to_string(layer) + ".ires_merge");
        _SubgraphBegin(c);

        auto s = _SplitInput({m}, true, 2, (string) "model.decoder.layers." + std::to_string(layer) + ".self_attn.ires_split");

        i = s[0];
        res = s[1];

        auto ix = KVCacheTestCPUNPUAttention(c, i, res, hidden_dim, hidden_dim / mutil_head_size, mutil_head_size, cache_max, (string) "model.decoder.layers." + std::to_string(layer) + ".self_attn");
    }
}