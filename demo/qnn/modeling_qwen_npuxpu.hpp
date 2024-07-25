#include <cmath>
#include <csignal>
#include "Types.hpp"
#include "express/Express.hpp"

using namespace mllm;

namespace modeling {

NetTensor *Qwen_CPUNPUAttention_t1(Context *c, NetTensor *x, NetTensor *res, int embedding_size, int hidden_size, int head_size, int cache_max, string name, int seq, int chunk) {
    x = x->view(1, static_cast<int>(seq / chunk / 32), static_cast<int>(32), hidden_size * head_size);
    auto *q = _LinearINT8({x}, embedding_size, hidden_size * head_size, false, name + ".q_proj");
    auto *k = _LinearINT8({x}, embedding_size, hidden_size * head_size, false, name + ".k_proj");
    auto *v = _LinearINT8({x}, embedding_size, hidden_size * head_size, false, name + ".v_proj");
    q = q->view(1, head_size, seq / chunk, hidden_size);
    k = k->view(1, head_size, seq / chunk, hidden_size);
    v = v->view(1, head_size, seq / chunk, hidden_size);

    q = _Dequantize({q}, true, (string)name + ".q_proj.dequantize");
    k = _Dequantize({k}, true, (string)name + ".k_proj.dequantize");
    v = _Dequantize({v}, true, (string)name + ".v_proj.dequantize");

    v = _Transpose({v}, {0, 2, 3, 1}, (string)name + ".v_proj.transpose");

    auto *m = _MergeOutput({q, k, v, res}, name + ".qkv_merge");

    // --------------------
    _SubgraphBegin(c, MLLM_CPU);
    // --------------------

    auto s = _SplitInput({m}, true, 4, name + ".qkv_split");

    q = s[0];
    k = s[1];
    v = s[2];
    res = s[3];

    q = _RoPE({q}, HFHUBROPE, name + ".q_rope");
    k = _RoPE({k}, HFHUBROPE, name + ".k_rope");

    k = _KVCacheNPU({k}, cache_max, name + ".k_cache");
    v = _KVCacheNPU({v}, cache_max, name + ".v_cache");

    auto *qk = _Matmul({q, k}, false, true, name + ".qk");
    qk = *qk / std::sqrt(hidden_size);
    qk = _Causalmask({qk}, name + ".mask");
    qk = _Softmax({qk}, DIMENSION, name + ".softmax");

    auto *o = _Matmul({qk, v}, false, false, name + ".qkv");

    return o;
}

NetTensor *Qwen_FFN_NPU(Context *c, NetTensor *i, int hidden_dim, int ffn_hidden_dim, string name) {
    auto *x = _LinearINT8({i}, hidden_dim, ffn_hidden_dim, false, name + ".gate_proj");
    auto *y = _LinearINT8({i}, hidden_dim, ffn_hidden_dim, false, name + ".up_proj");
    x = _Dequantize({x}, true, (string)name + ".gate_proj.dequantize", false);
    y = _Dequantize({y}, true, (string)name + ".up_proj.dequantize", false);
    x = _SiLU({x}, name + ".silu");
    x = *x * y;
    x = _Quantize({x}, true, (string)name + ".down_proj.quantize");
    x = _LinearINT8({x}, ffn_hidden_dim, hidden_dim, false, name + ".down_proj");
    x = _Dequantize({x}, true, (string)name + ".down_proj.dequantize");
    return x;
}

// put o in the CPU.
void qwen_npu_t1(Context *c, int vocab_size = 32000, int hidden_dim = 4096, int ffn_hidden_dim = 11008, int mutil_head_size = 32, int cache_max = 200, int seq = 256, int chunk = 2) {
    auto *i = _Input(c);
    i = _Embedding({i}, vocab_size, hidden_dim, (string) "model.embed_tokens");

    for (int layer = 0; layer < 24; ++layer) {
        if (layer != 0) // for graph 0, it will be offloaded to CPU in QNNOptNet::convert
            _SubgraphBegin(c, MLLM_CPU);

        auto res = i;
        res = res->view(-1, mutil_head_size, -1, hidden_dim / mutil_head_size);

        i = _RMSNorm({i}, hidden_dim, 1e-5, (string) "model.layers." + std::to_string(layer) + ".input_layernorm");
        i = _Quantize({i}, true, (string) "model.layers." + std::to_string(layer) + ".self_attn.q_proj.quantize");

        i = i->view(-1, mutil_head_size, -1, hidden_dim / mutil_head_size);

        auto *m = _MergeOutput({i, res}, (string) "model.layers." + std::to_string(layer) + ".ires_merge");

        _SubgraphBegin(c);

        auto s = _SplitInput({m}, true, 2, (string) "model.layers." + std::to_string(layer) + ".self_attn.ires_split");

        i = s[0];
        res = s[1];

        auto *o = Qwen_CPUNPUAttention_t1(c, i, res, hidden_dim, hidden_dim / mutil_head_size, mutil_head_size, cache_max, (string) "model.layers." + std::to_string(layer) + ".self_attn", seq, chunk);

        i = _RMSNorm({o}, hidden_dim, 1e-5, (string) "model.layers." + std::to_string(layer) + ".post_attention_layernorm");
        i = _Quantize({i}, true, (string) "model.layers." + std::to_string(layer) + ".mlp.up_proj.quantize");

        i = i->view(1, 1, seq / chunk, hidden_dim);
        i = i->view(-1, mutil_head_size, -1, hidden_dim / mutil_head_size);
        m = _MergeOutput({i, o}, (string) "model.layers." + std::to_string(layer) + ".fres_merge");

        _SubgraphBegin(c);

        s = _SplitInput({m}, true, 2, (string) "model.layers." + std::to_string(layer) + ".fres_split");

        i = s[0];
        res = s[1];
        res = res->view(-1, 1, -1, hidden_dim);

        i = i->view(1, static_cast<int>(seq / chunk / 32), static_cast<int>(32), hidden_dim);
        i = Qwen_FFN_NPU(c, i, hidden_dim, ffn_hidden_dim, (string) "model.layers.mlp." + std::to_string(layer));

        i = i->view(1, 1, seq / chunk, hidden_dim);

        i = *i + res;
    }
    // _SubgraphBegin(c, MLLM_CPU);
    // i = _LayerNorm({i}, hidden_dim, true, 1e-5, (string) "model.decoder.final_layer_norm");
    // i = _Linear({i}, hidden_dim, vocab_size, false, "lm_head");
}

std::vector<NetTensor *> Qwen_CPUNPUAttention_t2(Context *c, NetTensor *x, NetTensor *res, int embedding_size, int hidden_size, int head_size, int cache_max, string name, int seq, int chunk) {
    x = x->view(1, static_cast<int>(seq / chunk / 32), static_cast<int>(32), hidden_size * head_size);
    auto *q = _LinearINT8({x}, embedding_size, hidden_size * head_size, false, name + ".q_proj");
    auto *k = _LinearINT8({x}, embedding_size, hidden_size * head_size, false, name + ".k_proj");
    auto *v = _LinearINT8({x}, embedding_size, hidden_size * head_size, false, name + ".v_proj");
    q = q->view(1, head_size, seq / chunk, hidden_size);
    k = k->view(1, head_size, seq / chunk, hidden_size);
    v = v->view(1, head_size, seq / chunk, hidden_size);

    q = _Dequantize({q}, true, (string)name + ".q_proj.dequantize");
    k = _Dequantize({k}, true, (string)name + ".k_proj.dequantize");
    v = _Dequantize({v}, true, (string)name + ".v_proj.dequantize");

    v = _Transpose({v}, {0, 2, 3, 1}, (string)name + ".v_proj.transpose");

    auto *m = _MergeOutput({q, k, v, res}, name + ".qkv_merge");

    // --------------------
    _SubgraphBegin(c, MLLM_CPU);
    // --------------------

    auto s = _SplitInput({m}, true, 4, name + ".qkv_split");

    q = s[0];
    k = s[1];
    v = s[2];
    res = s[3];

    q = _RoPE({q}, HFHUBROPE, name + ".q_rope");
    k = _RoPE({k}, HFHUBROPE, name + ".k_rope");

    k = _KVCacheNPU({k}, cache_max, name + ".k_cache");
    v = _KVCacheNPU({v}, cache_max, name + ".v_cache");

    auto *qk = _Matmul({q, k}, false, true, name + ".qk");
    qk = *qk / std::sqrt(hidden_size);
    qk = _Causalmask({qk}, name + ".mask");
    qk = _Softmax({qk}, DIMENSION, name + ".softmax");

    auto *o = _Matmul({qk, v}, false, false, name + ".qkv");
    o = _Quantize({o}, true, (string)name + ".o_proj.quantize");
    m = _MergeOutput({o, res}, name + ".or_merge");

    // --------------------
    _SubgraphBegin(c);
    // --------------------
    s = _SplitInput({m}, true, 2, name + ".or_split");

    o = s[0];
    res = s[1];

    o = o->view(1, static_cast<int>(seq / chunk / 32), static_cast<int>(32), hidden_size * head_size);
    res = res->view(-1, 1, -1, hidden_size * head_size);
    o = _LinearINT8({o}, hidden_size * head_size, embedding_size, false, name + ".o_proj");
    o = _Dequantize({o}, true, (string)name + ".o_proj.dequantize");

    return {o, res};
}

std::vector<NetTensor *> Qwen_CPUAttention_t2(Context *c, NetTensor *x, NetTensor *res, int embedding_size, int hidden_size, int head_size, int cache_max, string name, int seq, int chunk) {
    x = x->view(1, static_cast<int>(seq / chunk / 32), static_cast<int>(32), hidden_size * head_size);
    auto *q = _Linear({x}, embedding_size, hidden_size * head_size, false, name + ".q_proj");
    auto *k = _Linear({x}, embedding_size, hidden_size * head_size, false, name + ".k_proj");
    auto *v = _Linear({x}, embedding_size, hidden_size * head_size, false, name + ".v_proj");
    q = q->view(1, head_size, seq / chunk, hidden_size);
    k = k->view(1, head_size, seq / chunk, hidden_size);
    v = v->view(1, head_size, seq / chunk, hidden_size);

    q = _RoPE({q}, HFHUBROPE, name + ".q_rope");
    k = _RoPE({k}, HFHUBROPE, name + ".k_rope");

    k = _KVCacheNPU({k}, cache_max, name + ".k_cache");
    v = _KVCacheNPU({v}, cache_max, name + ".v_cache");

    auto *qk = _Matmul({q, k}, false, true, name + ".qk");
    qk = *qk / std::sqrt(hidden_size);
    qk = _Causalmask({qk}, name + ".mask");
    qk = _Softmax({qk}, DIMENSION, name + ".softmax");

    auto *o = _Matmul({qk, v}, false, false, name + ".qkv");

    o = o->view(1, static_cast<int>(seq / chunk / 32), static_cast<int>(32), hidden_size * head_size);
    res = res->view(-1, 1, -1, hidden_size * head_size);
    o = _Linear({o}, hidden_size * head_size, embedding_size, false, name + ".o_proj");

    return {o, res};
}

NetTensor *Qwen_FFN_CPU(Context *c, NetTensor *i, int hidden_dim, int ffn_hidden_dim, string name) {
    auto *x = _Linear({i}, hidden_dim, ffn_hidden_dim, false, name + ".gate_proj");
    auto *y = _Linear({i}, hidden_dim, ffn_hidden_dim, false, name + ".up_proj");
    x = _SiLU({x}, name + ".silu");
    x = *x * y;
    x = _Linear({x}, ffn_hidden_dim, hidden_dim, false, name + ".down_proj");
    return x;
}

void qwen_cpu_t2(Context *c, int vocab_size = 32000, int hidden_dim = 4096, int ffn_hidden_dim = 11008, int mutil_head_size = 32, int cache_max = 200, int seq = 256, int chunk = 2) {
    auto *i = _Input(c);
    i = _Embedding({i}, vocab_size, hidden_dim, (string) "model.embed_tokens");


    for (int layer = 0; layer < 24; ++layer) {
        auto res = i;
        res = res->view(-1, mutil_head_size, -1, hidden_dim / mutil_head_size);

        i = _RMSNorm({i}, hidden_dim, 1e-5, (string) "model.layers." + std::to_string(layer) + ".input_layernorm");
        i = _Quantize({i}, true, (string) "model.layers." + std::to_string(layer) + ".self_attn.q_proj.quantize");

        i = i->view(-1, mutil_head_size, -1, hidden_dim / mutil_head_size);

        auto ix = Qwen_CPUAttention_t2(c, i, res, hidden_dim, hidden_dim / mutil_head_size, mutil_head_size, cache_max, (string) "model.layers." + std::to_string(layer) + ".self_attn", seq, chunk);

        i = ix[0];
        res = ix[1];

        i = i->view(1, 1, seq / chunk, hidden_dim);
        i = *i + res;

        res = i;

        i = _RMSNorm({i}, hidden_dim, 1e-5, (string) "model.layers." + std::to_string(layer) + ".post_attention_layernorm");

        i = i->view(1, static_cast<int>(seq / chunk / 32), static_cast<int>(32), hidden_dim);
        i = Qwen_FFN_CPU(c, i, hidden_dim, ffn_hidden_dim, (string) "model.layers." + std::to_string(layer) + ".mlp");

        i = i->view(1, 1, seq / chunk, hidden_dim);

        i = *i + res;
    }
    i = _RMSNorm({i}, hidden_dim, 1e-5, (string) "model.norm");
    i = _Linear({i}, hidden_dim, vocab_size, false, "lm_head");
}

// merge o and FFN.
void qwen_npu_t2(Context *c, int vocab_size = 32000, int hidden_dim = 4096, int ffn_hidden_dim = 11008, int mutil_head_size = 32, int cache_max = 200, int seq = 256, int chunk = 2) {
    auto *i = _Input(c);
    i = _Embedding({i}, vocab_size, hidden_dim, (string) "model.embed_tokens");

    // first 23 layer using NPU-CPU prefilling
    for (int layer = 0; layer < 1; ++layer) {
        if (layer != 0) // for graph 0, it will be offloaded to CPU in QNNOptNet::convert
            _SubgraphBegin(c, MLLM_CPU);

        auto res = i;
        res = res->view(-1, mutil_head_size, -1, hidden_dim / mutil_head_size);

        i = _RMSNorm({i}, hidden_dim, 1e-5, (string) "model.layers." + std::to_string(layer) + ".input_layernorm");
        i = _Quantize({i}, true, (string) "model.layers." + std::to_string(layer) + ".self_attn.q_proj.quantize");

        i = i->view(-1, mutil_head_size, -1, hidden_dim / mutil_head_size);

        auto *m = _MergeOutput({i, res}, (string) "model.layers." + std::to_string(layer) + ".ires_merge");

        _SubgraphBegin(c);

        auto s = _SplitInput({m}, true, 2, (string) "model.layers." + std::to_string(layer) + ".self_attn.ires_split");

        i = s[0];
        res = s[1];

        auto ix = Qwen_CPUNPUAttention_t2(c, i, res, hidden_dim, hidden_dim / mutil_head_size, mutil_head_size, cache_max, (string) "model.layers." + std::to_string(layer) + ".self_attn", seq, chunk);

        i = ix[0];
        res = ix[1];

        i = i->view(1, 1, seq / chunk, hidden_dim);
        i = *i + res;

        // _SubgraphBegin(c, MLLM_CPU);
        res = i;

        i = _RMSNorm({i}, hidden_dim, 1e-5, (string) "model.layers." + std::to_string(layer) + ".post_attention_layernorm");
        i = _Quantize({i}, true, (string) "model.layers." + std::to_string(layer) + ".mlp.up_proj.quantize");

        i = i->view(1, static_cast<int>(seq / chunk / 32), static_cast<int>(32), hidden_dim);
        i = Qwen_FFN_NPU(c, i, hidden_dim, ffn_hidden_dim, (string) "model.layers." + std::to_string(layer) + ".mlp");


        i = i->view(1, 1, seq / chunk, hidden_dim);

        i = *i + res;
    }
    // for the 24th layer, only calculate to the KVCache
    // {
    //     int layer = 23;
    //     // --------------------
    //     _SubgraphBegin(c, MLLM_CPU);
    //     // --------------------

    //     auto res = i;
    //     res = res->view(-1, mutil_head_size, -1, hidden_dim / mutil_head_size);

    //     i = _RMSNorm({i}, hidden_dim, 1e-5, (string) "model.layers." + std::to_string(layer) + ".input_layernorm");
    //     i = _Quantize({i}, true, (string) "model.layers." + std::to_string(layer) + ".self_attn.q_proj.quantize");

    //     i = i->view(-1, mutil_head_size, -1, hidden_dim / mutil_head_size);

    //     auto *m = _MergeOutput({i, res}, (string) "model.layers." + std::to_string(layer) + ".ires_merge");

    //     _SubgraphBegin(c);

    //     auto s = _SplitInput({m}, true, 2, (string) "model.layers." + std::to_string(layer) + ".self_attn.ires_split");

    //     i = s[0];
    //     res = s[1];

    //     string name = "model.layers." + std::to_string(layer) + ".self_attn";
    //     res = res->view(1, static_cast<int>(seq / chunk / 32), static_cast<int>(32), hidden_dim);
    //     auto *q = _LinearINT8({res}, hidden_dim, hidden_dim, false, name + ".q_proj");
    //     auto *k = _LinearINT8({res}, hidden_dim, hidden_dim, false, name + ".k_proj");
    //     auto *v = _LinearINT8({res}, hidden_dim, hidden_dim, false, name + ".v_proj");
    //     q = q->view(1, mutil_head_size, seq / chunk, hidden_dim / mutil_head_size);
    //     k = k->view(1, mutil_head_size, seq / chunk, hidden_dim / mutil_head_size);
    //     v = v->view(1, mutil_head_size, seq / chunk, hidden_dim / mutil_head_size);

    //     q = _Dequantize({q}, true, (string)name + ".q_proj.dequantize");
    //     k = _Dequantize({k}, true, (string)name + ".k_proj.dequantize");
    //     v = _Dequantize({v}, true, (string)name + ".v_proj.dequantize");

    //     v = _Transpose({v}, {0, 2, 3, 1}, (string)name + ".v_proj.transpose");

    //     m = _MergeOutput({q, k, v, res}, name + ".qkv_merge");
    //     // --------------------
    //     _SubgraphBegin(c, MLLM_CPU);
    //     // --------------------
    //     s = _SplitInput({m}, true, 4, name + ".qkv_split");

    //     q = s[0];
    //     k = s[1];
    //     v = s[2];
    //     res = s[3];

    //     q = _RoPE({q}, HFHUBROPE, name + ".q_rope");
    //     k = _RoPE({k}, HFHUBROPE, name + ".k_rope");

    //     k = _KVCacheNPU({k}, cache_max, name + ".k_cache");
    //     v = _KVCacheNPU({v}, cache_max, name + ".v_cache");

    //     auto *qk = _Matmul({q, k}, false, true, name + ".qk");
    //     qk = *qk / std::sqrt(hidden_dim / mutil_head_size);
    //     qk = _Causalmask({qk}, name + ".mask");
    //     qk = _Softmax({qk}, DIMENSION, name + ".softmax");

    //     auto *o = _Matmul({qk, v}, false, false, name + ".qkv");

    //     o = o->view(1, static_cast<int>(seq / chunk / 32), static_cast<int>(32), hidden_dim);
    //     res = res->view(-1, 1, -1, hidden_dim);
    //     o = _Linear({o}, hidden_dim, hidden_dim, false, name + ".o_proj");

    //     i = i->view(1, 1, seq / chunk, hidden_dim);
    //     i = *i + res;
    // }
}

void qwen_npu_cpu_inter(Context *c, int vocab_size = 32000, int hidden_dim = 4096, int ffn_hidden_dim = 11008, int mutil_head_size = 32, int cache_max = 200, int seq = 256, int chunk = 2) {
    auto *i = _Input(c);

    // the 24th layer
    const int layer = 24 - 1;
    {
        auto res = i;
        res = res->view(-1, mutil_head_size, -1, hidden_dim / mutil_head_size);

        i = _RMSNorm({i}, hidden_dim, 1e-5, (string) "model.layers." + std::to_string(layer) + ".input_layernorm");
        i = _Quantize({i}, true, (string) "model.layers." + std::to_string(layer) + ".self_attn.q_proj.quantize");

        i = i->view(-1, mutil_head_size, -1, hidden_dim / mutil_head_size);

        auto ix = Qwen_CPUAttention_t2(c, i, res, hidden_dim, hidden_dim / mutil_head_size, mutil_head_size, cache_max, (string) "model.layers." + std::to_string(layer) + ".self_attn", seq, chunk);

        i = ix[0];
        res = ix[1];

        i = i->view(1, 1, seq / chunk, hidden_dim);
        i = *i + res;

        res = i;

        i = _RMSNorm({i}, hidden_dim, 1e-5, (string) "model.layers." + std::to_string(layer) + ".post_attention_layernorm");

        i = i->view(1, static_cast<int>(seq / chunk / 32), static_cast<int>(32), hidden_dim);
        i = Qwen_FFN_CPU(c, i, hidden_dim, ffn_hidden_dim, (string) "model.layers." + std::to_string(layer) + ".mlp");

        i = i->view(1, 1, seq / chunk, hidden_dim);

        i = *i + res;
    }

    i = _RMSNorm({i}, hidden_dim, 1e-5, (string) "model.norm");
    i = _Linear({i}, hidden_dim, vocab_size, false, "lm_head");
}
} // namespace modeling
