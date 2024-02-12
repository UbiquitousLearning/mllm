#include <iostream>
#include <valarray>
#include <csignal>
#include "Timing.hpp"
#include "Types.hpp"
#include "cmdline.h"
#include "backends/QNN/QNNNet.hpp"
#include "backends/QNN/QNNExecutor.hpp"
#include "express/Express.hpp"
#include "MockLoader.hpp"

using namespace mllm;

// when set name of linear, use q8 as postfix to let mock loader load int8 data

NetTensor *Attention(Context *ctx, NetTensor *i, uint32_t hidden_dim, uint32_t ffn_hidden_dim, int layer) {
    auto *q = _Linear({i}, hidden_dim, hidden_dim, false, std::to_string(layer) + "attention.q.q8");
    auto *k = _Linear({i}, hidden_dim, hidden_dim, false, std::to_string(layer) + "attention.k.q8");
    auto *v = _Linear({i}, hidden_dim, hidden_dim, false, std::to_string(layer) + "attention.v.q8");
    // NSD
    q = q->view(-1, 32, -1,  hidden_dim / 32); // bhsd parameter order
    k = k->view(-1, 32, -1,  hidden_dim / 32);
    v = v->view(-1, 32, -1,  hidden_dim / 32);
    // NSHD
    q = _RoPE({q}, LLAMAROPE, std::to_string(layer) + "RoPE_q");
    k = _RoPE({k}, LLAMAROPE, std::to_string(layer) + "RoPE_k");
    k = _KVCache({k}, 1, std::to_string(layer) + "KVCache_k");
    v = _KVCache({v}, 1, std::to_string(layer) + "KVCache_v");
    auto *qk = _Matmul({q, k}, false, true, std::to_string(layer) + "attention.qk");
    // NHSD
    qk = _Scale({qk}, 0.5f, 0.0F, false, std::to_string(layer) + "attention.scale");
    qk = _Causalmask({qk}, std::to_string(layer) + "mask");
    qk = _Softmax({qk}, 3, std::to_string(layer) + "softmax");
    // NHSD
    auto *o = _Matmul({qk, v}, false, false, std::to_string(layer) + "qkv");
    // NSHD
    o = o->view(-1, 1, -1, hidden_dim);
    o = _Linear({o}, hidden_dim, hidden_dim, false, std::to_string(layer) + "attention.o.q8");

    return o;
}

NetTensor *FFN(Context *ctx, NetTensor *i, uint32_t hidden_dim, uint32_t ffn_hidden_dim, int layer) {
    auto *x = _Linear({i}, hidden_dim, ffn_hidden_dim, false, std::to_string(layer) + "ffn.l1.q8");
    auto *y = _Linear({i}, hidden_dim, ffn_hidden_dim, false, std::to_string(layer) + "ffn.l3.q8");

    x = _SiLU({x}, std::to_string(layer) + "ffn.silu1");
    auto *z = _Mul({x, y}, std::to_string(layer) + "ffn.mul");

    z = _Linear({z}, ffn_hidden_dim, hidden_dim, false, std::to_string(layer) + "ffn.l2.q8");

    return z;
}

NetTensor *FFNNoSiLU(Context *ctx, NetTensor *i, uint32_t hidden_dim, uint32_t ffn_hidden_dim, int layer) {
    auto *x = _Linear({i}, hidden_dim, ffn_hidden_dim, false, std::to_string(layer) + "ffn.l1.q8");
    auto *y = _Linear({i}, hidden_dim, ffn_hidden_dim, false, std::to_string(layer) + "ffn.l3.q8");

    // x = _SiLU( {x}, std::to_string(layer)+"ffn.silu1");
    auto *z = _Add({x, y}, std::to_string(layer) + "ffn.add");

    z = _Linear({z}, ffn_hidden_dim, hidden_dim, false, std::to_string(layer) + "ffn.l2.q8");

    return z;
}

void LLaMA(Context *ctx, uint32_t hidden_dim, uint32_t ffn_hidden_dim) {
    auto *i = _Input(ctx);

    i = _RoPE({i}, RoPEType::LLAMAROPE, "RoPE_0");
    i = _Softmax({i}, 3, "softmax0");
    for (int layer = 0; layer < 8; ++layer) {
        i = _RMSNorm({i}, hidden_dim, 1e-6, std::to_string(layer) + "RMSNorm");
        i = Attention(ctx, i, hidden_dim, ffn_hidden_dim, layer);
        i = FFN(ctx, i, hidden_dim, ffn_hidden_dim, layer);
    }
}

void LLaMANoSiLU(Context *ctx, uint32_t hidden_dim, uint32_t ffn_hidden_dim) {
    auto *i = _Input(ctx);

    // i = _RoPE( {i}, "RoPE_0");
    i = _Softmax({i}, 3, "softmax0");
    for (int layer = 0; layer < 8; ++layer) {
        i = _RMSNorm({i}, hidden_dim, 1e-6, std::to_string(layer) + "RMSNorm");
        i = Attention(ctx, i, hidden_dim, ffn_hidden_dim, layer);
        i = FFNNoSiLU(ctx, i, hidden_dim, ffn_hidden_dim, layer);
    }
}

NetTensor *SiLU(Context *ctx, uint32_t hidden_dim, uint32_t ffn_hidden_dim, int layer) {
    auto *i = _Input(ctx);
    auto *z = _SiLU({i}, "ffn.silu1");
    for (int l = 1; l <= layer; l++)
        z = _SiLU({z}, std::to_string(l) + ".ffn.silu1");

    return z;
}

NetTensor *RMSNorm(Context *ctx, uint32_t hidden_dim, uint32_t ffn_hidden_dim, int layer) {
    auto *i = _Input(ctx);
    auto *z = _RMSNorm({i}, hidden_dim, 1e-6, "ffn.rmsnorm1");
    for (int i = 1; i <= layer; i++)
        z = _RMSNorm({z}, hidden_dim, 1e-6, std::to_string(i) + ".ffn.rmsnorm");

    return z;
}

NetTensor *RoPE(Context *ctx, uint32_t hidden_dim, uint32_t ffn_hidden_dim, int layer) {
    auto *i = _Input(ctx);
    auto *z = _RoPE({i}, LLAMAROPE, "ffn.rope");
    for (int i = 1; i <= layer; i++)
        z = _RoPE({z}, LLAMAROPE, std::to_string(i) + ".ffn.rope");
    return z;
}

NetTensor *SeperateFFN(Context *ctx, uint32_t hidden_dim, uint32_t ffn_hidden_dim, int layer) {
    auto *i = _Input(ctx);

    auto *z = _Softmax({i}, 3, "softmax0");
    for (int l = 1; l <= layer; l++) {
        auto *x = _Linear({z}, hidden_dim, ffn_hidden_dim, false, std::to_string(l) + "ffn.l1.q8");
        x = _SiLU({x}, std::to_string(l) + "ffn.silu1");

        auto *y = _Linear({z}, hidden_dim, ffn_hidden_dim, false, std::to_string(l) + "ffn.l3.q8");
        z = _Mul({x, y}, std::to_string(l) + "ffn.mul");

        z = _Linear({z}, ffn_hidden_dim, hidden_dim, false, std::to_string(l) + "ffn.l2.q8");
    }

    return z;
}

NetTensor *SeperateFFNNoSiLU(Context *ctx, uint32_t hidden_dim, uint32_t ffn_hidden_dim, int layer) {
    auto *i = _Input(ctx);

    auto *z = _Softmax({i}, 3, "softmax0");
    for (int l = 1; l <= layer; l++) {
        auto *x = _Linear({z}, hidden_dim, ffn_hidden_dim, false, std::to_string(l) + "ffn.l1.q8");
        auto *y = _Linear({z}, hidden_dim, ffn_hidden_dim, false, std::to_string(l) + "ffn.l3.q8");

        // x = _SiLU( {x}, std::to_string(l)+"ffn.silu1");
        z = _Mul({x, y}, std::to_string(l) + "ffn.mul");

        z = _Linear({z}, ffn_hidden_dim, hidden_dim, false, std::to_string(l) + "ffn.l2.q8");
    }

    return z;
}

NetTensor *Linear(Context *ctx, uint32_t hidden_dim, uint32_t ffn_hidden_dim, int layer) {
    auto *i = _Input(ctx);


    auto *z = _Causalmask({i}, "mask");
    auto *x = _Linear({z}, hidden_dim, ffn_hidden_dim, false, std::to_string(0) + "ffn.l1.q8");
    for (int l = 1; l <= layer; l++) {
        auto *y = _Linear({z}, hidden_dim, ffn_hidden_dim, false, std::to_string(l) + "ffn.l3.q8");
        x = _Add({x, y}, std::to_string(l) + "ffn.add");

    }

    return x;
}

NetTensor *FFNLinearCompose(Context *ctx, uint32_t hidden_dim, uint32_t ffn_hidden_dim, int layer) {
    auto *i = _Input(ctx);

    auto *z = _Softmax({i}, 3, "softmax0");

    for (int l = 1; l <= layer; l++) {
        z = _Linear({z}, hidden_dim, ffn_hidden_dim, false, std::to_string(l) + "ffn.l1.q8");
        z = _Linear({z}, ffn_hidden_dim, hidden_dim, false, std::to_string(l) + "ffn.l3.q8");
    }

    return z;
}

NetTensor *FFNLinearSiLUCompose(Context *ctx, uint32_t hidden_dim, uint32_t ffn_hidden_dim, int layer) {
    auto *i = _Input(ctx);

    auto *z = _Softmax({i}, 3, "softmax0");

    for (int l = 1; l <= layer; l++) {
        z = _Linear({z}, hidden_dim, ffn_hidden_dim, false, std::to_string(l) + "ffn.l1.q8");
        z = _SiLU({z}, std::to_string(l) + "ffn.silu1");
        z = _Linear({z}, ffn_hidden_dim, hidden_dim, false, std::to_string(l) + "ffn.l3.q8");
    }

    return z;
}

void SeperateAttention(Context *ctx, uint32_t hidden_dim, uint32_t ffn_hidden_dim, int layer) {
    auto *i = _Input(ctx);
    auto *o = _Softmax({i}, 3, "softmax0");

    for (int l = 1; l <= layer; l++) {
        auto *q = _Linear({o}, hidden_dim, hidden_dim, false, std::to_string(l) + "attention.q.q8");
        auto *k = _Linear({o}, hidden_dim, hidden_dim, false, std::to_string(l) + "attention.k.q8");
        auto *v = _Linear({o}, hidden_dim, hidden_dim, false, std::to_string(l) + "attention.v.q8");
        q = q->view(-1, 32, -1,  hidden_dim / 32); // bhsd parameter order
        k = k->view(-1, 32, -1,  hidden_dim / 32);
        v = v->view(-1, 32, -1,  hidden_dim / 32);
        q = _RoPE({q}, LLAMAROPE, std::to_string(l) + "RoPE_q");
        k = _RoPE({k}, LLAMAROPE, std::to_string(l) + "RoPE_k");
        k = _KVCache({k}, 1, std::to_string(l) + "KVCache_k");
        v = _KVCache({v}, 1, std::to_string(l) + "KVCache_v");
        auto *qk = _Matmul({q, k}, false, true, std::to_string(l) + "attention.qk");
        qk = _Scale({qk}, 0.5f, 0.0F, false, std::to_string(l) + "attention.scale");
        qk = _Causalmask({qk}, std::to_string(l) + "mask");
        qk = _Softmax({qk}, 3, std::to_string(l) + "softmax");

        o = _Matmul({qk, v}, false, false, std::to_string(l) + "qkv");
        o = o->view(-1, 1, -1, hidden_dim);
        o = _Linear({o}, hidden_dim, hidden_dim, false, std::to_string(l) + "attention.o.q8");
    }

    // return o;
}

void SeperateAttentionNOCustom(Context *ctx, uint32_t hidden_dim, uint32_t ffn_hidden_dim, int layer) {
    auto *i = _Input(ctx);
    auto *o = _Softmax({i}, 3, "softmax0");

    for (int l = 1; l <= layer; l++) {
        auto *q = _Linear({o}, hidden_dim, hidden_dim, false, std::to_string(l) + "attention.q.q8");
        auto *k = _Linear({o}, hidden_dim, hidden_dim, false, std::to_string(l) + "attention.k.q8");
        auto *v = _Linear({o}, hidden_dim, hidden_dim, false, std::to_string(l) + "attention.v.q8");
        // q = _View(ctx, {q}, {-1, 2, -1, -1}, {0, 3, 2, 3});
        // k = _View(ctx, {q}, {-1, 2, -1, -1}, {0, 3, 2, 3});
        // v = _View(ctx, {q}, {-1, 2, -1, -1}, {0, 3, 2, 3});
        // q = _RoPE( {q}, std::to_string(l)+"RoPE_q");
        // k = _RoPE( {k}), std::to_string(l)+"RoPE_k";
        auto *qk = _Matmul({q, k}, false, true, std::to_string(l) + "attention.qk");
        qk = _Scale({qk}, 0.5f, 0.0F, false, std::to_string(l) + "attention.scale");
        // qk = _Causalmask( {qk}, std::to_string(l)+"mask");
        qk = _Softmax({qk}, 3, std::to_string(l) + "softmax");
        o = _Matmul({qk, v}, false, false, std::to_string(l) + "qkv");
        // o = _View(ctx, {o}, {-1, -1, -1, -1}, {0, -1, 2, 1 + 3}, "qkv_view");
    }

    // return o;
}

// NetTensor * Attention2(Context *ctx, NetTensor * i, uint32_t hidden_dim, uint32_t ffn_hidden_dim, int layer) {

//     auto *q = _Linear({i}, hidden_dim, hidden_dim, false, std::to_string(layer)+"attention.q.q8");
//     auto *k = _Linear({i}, hidden_dim, hidden_dim, false, std::to_string(layer)+"attention.k.q8");
//     auto *v = _Linear({i}, hidden_dim, hidden_dim, false, std::to_string(layer)+"attention.v.q8");
//     // q = _View(ctx, {q}, {-1, 2, -1, -1}, {0, 3, 2, 3});
//     // k = _View(ctx, {q}, {-1, 2, -1, -1}, {0, 3, 2, 3});
//     // v = _View(ctx, {q}, {-1, 2, -1, -1}, {0, 3, 2, 3});
//     q = _RoPE( {q}, std::to_string(layer)+"RoPE_q");
//     k = _RoPE( {k}), std::to_string(layer)+"RoPE_k";
//     auto *qk = _Matmul( {q, k}, false, true, std::to_string(layer)+"attention.qk");
//     qk = _Scale( {qk}, 0.5f, 0.0F, false, std::to_string(layer)+"attention.scale");
//     qk = _Causalmask( {qk}, std::to_string(layer)+"mask");
//     qk = _Softmax({qk}, 3, std::to_string(layer)+"softmax");
//     auto *o = _Matmul( {qk, v}, false, false, std::to_string(layer)+"qkv");
//     // o = _View(ctx, {o}, {-1, -1, -1, -1}, {0, -1, 2, 1 + 3}, "qkv_view");

//     return o;
// }

// NOTE:this should be tested in QNN sub dir, as we don't include qnn headers in global
// void KVCacheCopy(int hidden_dim, int ffn_hidden_dim, int layer, int seq_len) {
//     for (int i = 0; i < layer; i++) {
//         QNNMemoryManager *mm = new QNNMemoryManager();

//         void *input_ptr;
//         void *output_ptr;
//         mm->alloc(&input_ptr, seq_len * hidden_dim * sizeof(float), 32);
//         mm->alloc(&output_ptr, seq_len * hidden_dim * sizeof(float), 32);

//         uint64_t t_start = mllm_time_us();

//         memcpy(input_ptr, output_ptr, seq_len * hidden_dim * sizeof(float));

//         uint64_t t_end = mllm_time_us();
//         std::cout << "QNN KVCache shared buffer copy time" << (t_end - t_start) / 1000.0F << " ms" << std::endl;
//     }
// }

template <typename Dtype>
void fullTensor(shared_ptr<Tensor> input_tensor, Net net, vector<int> shape, Dtype value) {
    input_tensor->setBackend(net.backends()[BackendType::MLLM_QNN].get());
    input_tensor->setCtype(ChlType::BSHD);
    input_tensor->reshape(shape[0], shape[1], shape[2], shape[3]);
    input_tensor->alloc();
    input_tensor->fullData<Dtype>(value);
}

int main(int argc, char **argv) {
    // BackendConfig bnc;

    // shared_ptr<MemoryManager> mm = nullptr;
    // switch (bnc.memory) {
    // case BackendConfig::Memory_High:
    //     mm = std::make_shared<SystemMemoryManager>();
    //     break;
    // default:
    //     mm = std::make_shared<SystemMemoryManager>();
    //     break;
    // }

    // QNNBackend *qbn = new QNNBackend(mm);

    // // build graph
    // std::cout << "build graph" << std::endl;
    // testSilu(qbn);
    // // graph compile
    // std::cout << "graph compile" << std::endl;
    // qbn->graphFinilize();
    // // graph run
    // std::cout << "graph run" << std::endl;
    // qbn->graphExecute();

    // delete qbn;

    // argv 1 op name
    // argv 2 execution times

    int vocab_size = 32000;
    int hidden_dim = 4096;
    int ffn_hidden_dim = 11008;
    int mutil_head_size = 32;

    std::unique_ptr<Context> c_ptr(new Context());
    auto *c = c_ptr.get();

    int dimension = hidden_dim;
    int head_size = 1;
    int seqence_size = 1;

    if (strcmp(argv[1], "silu") == 0) {
        SiLU(c, ffn_hidden_dim, ffn_hidden_dim, atoi(argv[2]));
        dimension = ffn_hidden_dim;
    } else if (strcmp(argv[1], "rmsnorm") == 0) {
        RMSNorm(c, hidden_dim, ffn_hidden_dim, atoi(argv[2]));
        dimension = vocab_size;
    } else if (strcmp(argv[1], "rope") == 0) {
        RoPE(c, hidden_dim, ffn_hidden_dim, atoi(argv[2]));
        dimension = hidden_dim / mutil_head_size;
        seqence_size = 1000;
    } else if (strcmp(argv[1], "ffn") == 0) {
        SeperateFFN(c, hidden_dim, ffn_hidden_dim, atoi(argv[2]));
        dimension = hidden_dim;
        seqence_size = 1;
    } else if (strcmp(argv[1], "ffn_nosilu") == 0) {
        SeperateFFNNoSiLU(c, hidden_dim, ffn_hidden_dim, atoi(argv[2]));
        dimension = hidden_dim;
        seqence_size = 1;
    } else if (strcmp(argv[1], "ffn_compose") == 0) {
        FFNLinearCompose(c, hidden_dim, ffn_hidden_dim, atoi(argv[2]));
        dimension = hidden_dim;
        seqence_size = 1;
    } else if (strcmp(argv[1], "ffnsilu_compose") == 0) {
        FFNLinearSiLUCompose(c, hidden_dim, ffn_hidden_dim, atoi(argv[2]));
        dimension = hidden_dim;
        seqence_size = 1;
    } else if (strcmp(argv[1], "linear_attn") == 0) {
        Linear(c, hidden_dim, hidden_dim, atoi(argv[2]));
        dimension = hidden_dim;
        seqence_size = 1;
    } else if (strcmp(argv[1], "linear_ffn1") == 0) {
        Linear(c, hidden_dim, ffn_hidden_dim, atoi(argv[2]));
        dimension = hidden_dim;
        seqence_size = 1;
    } else if (strcmp(argv[1], "linear_ffn2") == 0) {
        Linear(c, ffn_hidden_dim, hidden_dim, atoi(argv[2]));
        dimension = ffn_hidden_dim;
        seqence_size = 1;
    } else if (strcmp(argv[1], "attn") == 0) {
        SeperateAttention(c, hidden_dim, hidden_dim, atoi(argv[2]));
        dimension = hidden_dim;
        seqence_size = 1;
    } else if (strcmp(argv[1], "attn_nocustom") == 0) {
        SeperateAttentionNOCustom(c, hidden_dim, hidden_dim, atoi(argv[2]));
        dimension = hidden_dim;
        seqence_size = 1;
    } else if (strcmp(argv[1], "llama_nosilu") == 0) {
        LLaMANoSiLU(c, hidden_dim, hidden_dim);
        dimension = hidden_dim;
        seqence_size = 1;
    } else if (strcmp(argv[1], "llama") == 0) {
        LLaMA(c, hidden_dim, ffn_hidden_dim);
        dimension = hidden_dim;
    }
    // else if (strcmp(argv[1], "kvcache_copy") == 0) {
    //     KVCacheCopy(hidden_dim, ffn_hidden_dim, atoi(argv[2]), atoi(argv[3]));
    //     return 0;
    // }

    BackendConfig bn;
    QNNNet net(bn, c);
    net.convert(c->sub_param_, MLLM_QNN);
    std::cout << "convert done" << std::endl;

    MockLoader loader("");
    QNNExecutor ex(&loader);
    shared_ptr<Tensor> input = std::make_shared<Tensor>();

    // 1 batch seqence length embedding
    fullTensor(input, net, {1, seqence_size, 1, dimension}, 2.f);

    ex.setup(&net);
    ex.run(&net, {input});
    ex.perf();
    // auto result = ex.result();
    // result[0]->printData<float>();
}