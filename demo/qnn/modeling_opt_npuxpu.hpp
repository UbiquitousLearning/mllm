#include <cmath>
#include <csignal>
#include "Types.hpp"
#include "express/Express.hpp"

using namespace mllm;

namespace modeling {

std::vector<NetTensor *> CPUNPUAttention(Context *c, NetTensor *x, NetTensor *res, int embedding_size, int hidden_size, int head_size, int cache_max, string name, int seq, int chunk) {

    x = x->view(1, static_cast<int>(seq/chunk/32), static_cast<int>(32), hidden_size * head_size);
    auto *q = _LinearINT8({x}, embedding_size, hidden_size * head_size, false, name + ".q_proj");
    auto *k = _LinearINT8({x}, embedding_size, hidden_size * head_size, false, name + ".k_proj");
    auto *v = _LinearINT8({x}, embedding_size, hidden_size * head_size, false, name + ".v_proj");
    q = q->view(1, head_size, seq/chunk, hidden_size);
    k = k->view(1, head_size, seq/chunk, hidden_size);
    v = v->view(1, head_size, seq/chunk, hidden_size);

    q = _Dequantize({q}, true, (string)name + ".q_proj.dequantize");
    k = _Dequantize({k}, true, (string)name + ".k_proj.dequantize");
    v = _Dequantize({v}, true, (string)name + ".v_proj.dequantize");

    v = _Transpose({v}, {0,2,3,1}, (string)name + ".v_proj.transpose");


    auto *m = _MergeOutput({q, k, v, res}, name + ".qkv_merge");

    // --------------------
    _SubgraphBegin(c, MLLM_CPU);
    // --------------------

    auto s = _SplitInput({m}, true, 4, name + ".qkv_split");

    q = s[0];
    k = s[1];
    v = s[2];
    res = s[3];

    k = _KVCacheNPU({k}, cache_max, name + ".k_cache");
    v = _KVCacheNPU({v}, cache_max, name + ".v_cache");

    auto *qk = _Matmul({q, k}, false, true, name + ".qk");
    qk = *qk / std::sqrt(hidden_size);
    qk = _Causalmask({qk}, name + ".mask");
    qk = _Softmax({qk}, DIMENSION, name + ".softmax");

    auto *o = _Matmul({qk, v}, false, false, name + ".qkv");
    return {o};

    o = _Quantize({o}, true, (string)name + ".out_proj.quantize");
    m = _MergeOutput({o, res}, name + ".or_merge");

    // --------------------
    _SubgraphBegin(c);
    // --------------------
    s = _SplitInput({m}, true, 2, name + ".or_split");

    o = s[0];
    res = s[1];
    
    o = o->view(1, static_cast<int>(seq/chunk/32), static_cast<int>(32), hidden_size * head_size);
    res = res->view(-1, 1, -1, hidden_size * head_size);
    o = _LinearINT8({o}, hidden_size * head_size, embedding_size, false, name + ".out_proj");
    o = _Dequantize({o}, true, (string)name + ".out_proj.dequantize");
    return {o, res};
}


NetTensor *FFN_NPU(Context *c, NetTensor *i, int hidden_dim, int ffn_hidden_dim, string name) {
    auto *x = i;
    x = _LinearINT8({x}, hidden_dim, ffn_hidden_dim, false, name + ".fc1");
    x = _ReLU({x}, name + ".fc2.relu");
    x = _LinearINT8({x}, ffn_hidden_dim, hidden_dim, false, name + ".fc2");
    x = _Dequantize({x}, true, (string)name + ".fc2.dequantize");
    return x;
}

void opt_npu(Context *c, int vocab_size = 32000, int hidden_dim = 4096, int ffn_hidden_dim = 11008, int mutil_head_size = 32, int cache_max = 200, int seq=256, int chunk=2) {
    auto *i = _Input(c);
    i = _Embedding({i}, vocab_size, hidden_dim, (string) "model.decoder.embed_tokens");

    for (int layer = 0; layer < 1; ++layer) {

         if (layer != 0) // for graph 0, it will be offloaded to CPU in QNNOptNet::convert
            _SubgraphBegin(c, MLLM_CPU);

        auto res = i;
        res = res->view(-1, mutil_head_size, -1, hidden_dim / mutil_head_size);

        i = _LayerNorm({i}, hidden_dim, true, 1e-5, (string) "model.decoder.layers." + std::to_string(layer) + ".self_attn_layer_norm");
        i = _Quantize({i}, true, (string) "model.decoder.layers." + std::to_string(layer) + ".self_attn.q_proj.quantize");

        i = i->view(-1, mutil_head_size, -1, hidden_dim / mutil_head_size);

        auto *m = _MergeOutput({i, res}, (string)"model.decoder.layers." + std::to_string(layer) + ".ires_merge");

        _SubgraphBegin(c);

        auto s = _SplitInput({m}, true, 2, (string)"model.decoder.layers." + std::to_string(layer) + ".self_attn.ires_split");

        i = s[0];
        res = s[1];
        
        auto ix = CPUNPUAttention(c, i, res, hidden_dim, hidden_dim / mutil_head_size, mutil_head_size, cache_max, (string) "model.decoder.layers." + std::to_string(layer) + ".self_attn", seq, chunk);
        return;

        i = ix[0];
        res = ix[1];

        i = i->view(1, 1, seq/chunk, hidden_dim);
        i = *i + res;        

        _SubgraphBegin(c, MLLM_CPU);
        res = i;
        i = _LayerNorm({i}, hidden_dim, true, 1e-5, (string) "model.decoder.layers." + std::to_string(layer) + ".final_layer_norm");
        i = _Quantize({i}, true, (string) "model.decoder.layers." + std::to_string(layer) + ".fc1.quantize");

        i = i->view(-1, mutil_head_size, -1, hidden_dim / mutil_head_size);
        m = _MergeOutput({i, res}, (string)"model.decoder.layers." + std::to_string(layer) + ".fres_merge");


        _SubgraphBegin(c);

        s = _SplitInput({m}, true, 2, (string)"model.decoder.layers." + std::to_string(layer) + ".fres_split");

        i = s[0];
        res = s[1];
        res = res->view(-1, 1, -1, hidden_dim);

        i = i->view(1, static_cast<int>(seq/chunk/32), static_cast<int>(32), hidden_dim);
        i = FFN_NPU(c, i, hidden_dim, ffn_hidden_dim, (string) "model.decoder.layers." + std::to_string(layer));

        i = i->view(1, 1, seq/chunk, hidden_dim);

        i = *i + res;
    }
    // _SubgraphBegin(c, MLLM_CPU);
    // i = _LayerNorm({i}, hidden_dim, true, 1e-5, (string) "model.decoder.final_layer_norm");
    // i = _Linear({i}, hidden_dim, vocab_size, false, "lm_head");
}

NetTensor * CPUAttention(Context *c, NetTensor *x, int embedding_size, int hidden_size, int head_size, int cache_max, string name) {
    x = x->view(-1, 1, -1, hidden_size * head_size);
    auto *q = _Linear({x}, embedding_size, hidden_size * head_size, false, name + ".q_proj");
    auto *k = _Linear({x}, embedding_size, hidden_size * head_size, false, name + ".k_proj");
    auto *v = _Linear({x}, embedding_size, hidden_size * head_size, false, name + ".v_proj");
    q = q->view(-1, head_size, -1, hidden_size);
    k = k->view(-1, head_size, -1, hidden_size);
    v = v->view(-1, head_size, -1, hidden_size);

    // NOTE: if share_input is false, the cache dtype will be I8(sq) and not be able to run on CPULinear
    k = _KVCache({k}, 1, true, cache_max, name + ".k_cache");
    v = _KVCache({v}, 1, true, cache_max, name + ".v_cache");

    auto *qk = _Matmul({q, k}, false, true, name + ".qk");

    qk = *qk / std::sqrt(hidden_size);
    qk = _Causalmask({qk}, name + ".mask");
    qk = _Softmax({qk}, DIMENSION, name + ".softmax");

    auto *o = _Matmul({qk, v}, false, false, name + ".qkv");

    o = o->view(-1, 1, -1, hidden_size * head_size);

    o = _Linear({o}, hidden_size * head_size, embedding_size, false, name + ".out_proj");
    return o;
}

NetTensor *FFN_CPU(Context *c, NetTensor *i, int hidden_dim, int ffn_hidden_dim, string name) {
    auto *x = i;
    x = _Linear({x}, hidden_dim, ffn_hidden_dim, false, name + ".fc1");
    x = _ReLU({x}, name + ".fc2.relu");
    x = _Linear({x}, ffn_hidden_dim, hidden_dim, false, name + ".fc2");
    return x;
}

void opt_cpu(Context *c, int vocab_size = 32000, int hidden_dim = 4096, int ffn_hidden_dim = 11008, int mutil_head_size = 32, int cache_max = 200) {
    auto *i = _Input(c);
    i = _Embedding({i}, vocab_size, hidden_dim, (string) "model.decoder.embed_tokens");

    for (int layer = 0; layer < 4; ++layer) {

        i = _LayerNorm({i}, hidden_dim, true, 1e-5, (string) "model.decoder.layers." + std::to_string(layer) + ".self_attn_layer_norm");

        i = i->view(-1, mutil_head_size, -1, hidden_dim / mutil_head_size);

        i = *CPUAttention(c, i, hidden_dim, hidden_dim / mutil_head_size, mutil_head_size, cache_max, (string) "model.decoder.layers." + std::to_string(layer) + ".self_attn") + i;

        i = _LayerNorm({i}, hidden_dim, true, 1e-5, (string) "model.decoder.layers." + std::to_string(layer) + ".final_layer_norm");

        i = i->view(-1, 1, -1, hidden_dim);
        i = *FFN_CPU(c, i, hidden_dim, ffn_hidden_dim, (string) "model.decoder.layers." + std::to_string(layer)) + i;
    }

    i = _LayerNorm({i}, hidden_dim, true, 1e-5, (string) "model.decoder.final_layer_norm");
    i = _Linear({i}, hidden_dim, vocab_size, false, "lm_head");
}
} // namespace modeling
