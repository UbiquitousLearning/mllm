#include <cstdint>
#include <iostream>
#include <valarray>
#include <csignal>
#include "MockLoader.hpp"
#include "Net.hpp"
#include "Executor.hpp"
#include "NetParameter.hpp"
#include "QnnTypes.h"
#include "Types.hpp"
#include "express/Express.hpp"
#include "tokenizers/BPE/Bpe.hpp"
#include "backends/QNN/QNNBackend.hpp"
#include "memory/SystemMemoryManager.hpp"
#include "qnn_wrapper.hpp"

using namespace mllm;

// when set name of linear, use q8 as postfix to let mock loader load int8 data
NetTensor * Attention(Context *ctx, NetTensor * i, uint32_t hidden_dim, uint32_t ffn_hidden_dim, int layer) {
    
    auto *q = _Linear(ctx, {i}, hidden_dim, hidden_dim, false, std::to_string(layer)+"attention.q.q8");
    auto *k = _Linear(ctx, {i}, hidden_dim, hidden_dim, false, std::to_string(layer)+"attention.k.q8");
    auto *v = _Linear(ctx, {i}, hidden_dim, hidden_dim, false, std::to_string(layer)+"attention.v.q8");
    // q = _View(ctx, {q}, {-1, 2, -1, -1}, {0, 3, 2, 3});
    // k = _View(ctx, {q}, {-1, 2, -1, -1}, {0, 3, 2, 3});
    // v = _View(ctx, {q}, {-1, 2, -1, -1}, {0, 3, 2, 3});
    // q = _RoPE(ctx, {q}, std::to_string(layer)+"RoPE_q");
    // k = _RoPE(ctx, {k}), std::to_string(layer)+"RoPE_k";
    auto *qk = _Matmul(ctx, {q, k}, false, true, std::to_string(layer)+"attention.qk");
    qk = _Scale(ctx, {qk}, 0.5f, 0.0F, false, std::to_string(layer)+"attention.scale");
    // qk = _Causalmask(ctx, {qk}, std::to_string(layer)+"mask");
    qk = _Softmax(ctx, {qk}, 3, std::to_string(layer)+"softmax");
    auto *o = _Matmul(ctx, {qk, v}, false, false, std::to_string(layer)+"qkv");
    // o = _View(ctx, {o}, {-1, -1, -1, -1}, {0, -1, 2, 1 + 3}, "qkv_view");

    return o;
}

NetTensor * FFN(Context *ctx, NetTensor * i, uint32_t hidden_dim, uint32_t ffn_hidden_dim, int layer) {

    auto *x = _Linear(ctx, {i}, hidden_dim, ffn_hidden_dim, false, std::to_string(layer)+"ffn.l1.q8");
    auto *y = _Linear(ctx, {i}, hidden_dim, ffn_hidden_dim, false, std::to_string(layer)+"ffn.l3.q8");

    x = _SiLU(ctx, {x}, std::to_string(layer)+"ffn.silu1");
    auto *z = _Add(ctx, {x, y});

    z = _Linear(ctx, {z}, ffn_hidden_dim, hidden_dim, false, std::to_string(layer)+"ffn.l2.q8");

    return z;
}

void LLaMA(Context *ctx, uint32_t hidden_dim, uint32_t ffn_hidden_dim) {
    auto *i = _Input(ctx);

    // i = _RoPE(ctx, {i}, "RoPE_0");
    i = _Softmax(ctx, {i}, 3, "softmax0");
    for(int layer=0; layer<4; ++layer) {

        auto *x = _RMSNorm(ctx, {i}, std::to_string(layer)+"RMSNorm");
        i = Attention( ctx, i, hidden_dim, ffn_hidden_dim, layer);
        i = FFN(ctx, i, hidden_dim, ffn_hidden_dim, layer);
    }

}


NetTensor * SiLU(Context *ctx,  uint32_t hidden_dim, uint32_t ffn_hidden_dim, int layer) {

    auto *i = _Input(ctx);
    auto *z = _SiLU(ctx, {i}, "ffn.silu1");
    for (int i = 1; i<=layer; i++)
        z = _SiLU(ctx, {z}, std::to_string(i)+".ffn.silu1");

    return z;
}

NetTensor * RMSNorm(Context *ctx,  uint32_t hidden_dim, uint32_t ffn_hidden_dim, int layer) {

    auto *i = _Input(ctx);
    auto *z = _RMSNorm(ctx, {i}, "ffn.rmsnorm1");
    for (int i = 1; i<=layer; i++)
        z = _RMSNorm(ctx, {z}, std::to_string(i)+".ffn.rmsnorm");

    return z;
}



template <typename Dtype>
void fullTensor(shared_ptr<Tensor> input_tensor, Net net, vector<int> shape, Dtype value) {
    input_tensor->setBackend(net.backends()[BackendType::MLLM_QNN].get());
    input_tensor->reshape(shape);
    input_tensor->alloc();
    input_tensor->fullData<Dtype>(value);
}

int main(int argc,char **argv) {
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

    int dimension = 0;

    if (strcmp(argv[1], "silu") == 0) {
        SiLU(c, hidden_dim, ffn_hidden_dim, atoi(argv[2]));
        dimension = hidden_dim;
    } else if (strcmp(argv[1], "rmsnorm") == 0) {
        RMSNorm(c, hidden_dim, ffn_hidden_dim, atoi(argv[2]));
        dimension = vocab_size;
    } else {
        LLaMA(c, hidden_dim, ffn_hidden_dim);
        dimension = hidden_dim;
    }
    

    BackendConfig bn;
    Net net(c->sub_param_, bn);
    net.convert(c->sub_param_, MLLM_QNN);
    std::cout << "convert done" << std::endl;

    MockLoader loader("");
    Executor ex(&loader);
    shared_ptr<Tensor> input = std::make_shared<Tensor>();

    // 1 batch seqence length embedding
    fullTensor(input, net, {1, 1, 1, dimension}, 2.f);

    ex.execute(&net, input);
    ex.perf();
    // auto result = ex.result();
    // result[0]->printData<float>();
}