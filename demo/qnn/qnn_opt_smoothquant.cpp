#include <iostream>
#include <valarray>
#include <csignal>
#include "MockLoader.hpp"
#include "Types.hpp"
#include "backends/QNN/QNNOptNet.hpp"
#include "cmdline.h"
#include "Net.hpp"
#include "Executor.hpp"
#include "express/Express.hpp"
#include "tokenizers/BPE/Bpe.hpp"
#include "backends/QNN/QNNNet.hpp"
#include "backends/QNN/QNNExecutor.hpp"

using namespace mllm;

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
unsigned int postProcessing(shared_ptr<Tensor> result, shared_ptr<Tensor> &out_result) {
    assert(result->batch() == 1);
    assert(result->head() == 1);
    out_result->reshape(1, 1, 1, 1);
    out_result->alloc();
    vector<float> scores;
    for (int i = 0; i < result->dimension(); ++i) {
        auto value = result->dataAt<float>(0, 0, result->sequence() - 1, i);
        scores.push_back(value);
    }
    auto token_idx = argmax(scores);
    out_result->setDataAt<float>(0, 0, 0, 0, token_idx);
    return token_idx;
}
NetTensor *Attention(Context *c, NetTensor *x, int embedding_size, int hidden_size, int head_size, int cache_max, string name) {

    x = _Quantize({x}, true, (string) name + ".x.quantize");
    auto *q = _LinearINT8({x}, embedding_size, hidden_size * head_size, false, name + ".q_proj");
    auto *k = _LinearINT8({x}, embedding_size, hidden_size * head_size, false, name + ".k_proj");
    auto *v = _LinearINT8({x}, embedding_size, hidden_size * head_size, false, name + ".v_proj");
    q = q->view(-1, head_size, -1, hidden_size);
    k = k->view(-1, head_size, -1, hidden_size);
    v = v->view(-1, head_size, -1, hidden_size);
    // q = _RoPE({q}, LLAMAROPE, name + ".q_rope");
    // k = _RoPE({k}, LLAMAROPE, name + ".k_rope");
    k = _KVCache({k}, cache_max, name + ".k_cache");
    v = _KVCache({v}, cache_max, name + ".v_cache");

    auto *m = _MergeOutput({q,k,v}, name + ".qkv_merge");

    _SubgraphBegin(c);

    auto s = _SplitInput({m}, false, name + ".qkv_split");

    q = s[0];
    k = s[1];
    v = s[2];

    auto *qk = _MatmulINT8({q, k}, false, true, name + ".qk");
    // qk = _Dequantize({qk}, false, (string) name + ".qk.dequantize");

    // qk = *qk / std::sqrt(hidden_size);
    // qk = _Causalmask({qk}, name + ".mask");
    qk = _Softmax({qk}, DIMENSION, name + ".softmax");

    // qk = _Quantize({qk}, false, (string) name + ".qk.quantize");
    auto *o = _MatmulINT8({qk, v}, false, false, name + ".qkv");

    _SubgraphBegin(c);

    o = o->view(-1, 1, -1, hidden_size * head_size);
    o = _LinearINT8({o}, hidden_size * head_size, embedding_size, false, name + ".o_proj");
    o = _Dequantize({o}, true, (string) name + ".o.dequantize");
    return o;
}
NetTensor *FFN(NetTensor *i, int hidden_dim, int ffn_hidden_dim, string name) {
    auto *x = _Quantize({i}, true, (string) name + ".x.quantize");
    x = _LinearINT8({x}, hidden_dim, ffn_hidden_dim, false, name + ".gate_proj");

    // x = _Dequantize({x}, (string) name + ".relux.dequantize");
    x = _ReLU({x}, name + ".relu");
    // x = _Quantize({x}, (string) name + ".relux.quantize");

    x = _LinearINT8({x}, ffn_hidden_dim, hidden_dim, false, name + ".down_proj");
    x = _Dequantize({x}, true, (string) name + ".x.dequantize");
    return x;
}
void opt(Context *c, int vocab_size = 32000, int hidden_dim = 4096, int ffn_hidden_dim = 11008, int mutil_head_size = 32, int cache_max = 200) {
    auto *i = _Input(c);
    // i = _Embedding({i}, vocab_size, hidden_dim, (string) "model.embed_tokens");
    // _SubgraphBegin(c);
    // loop
    
    for (int layer = 0; layer < 16; ++layer) {
        auto *x = Attention(c, i, hidden_dim, hidden_dim / mutil_head_size, mutil_head_size, cache_max, (string) "model.layers." + std::to_string(layer) + ".self_attn");
        i = _RMSNorm({x}, hidden_dim, 1e-6, (string) "model.layers." + std::to_string(layer) + ".post_attention_layernorm");
        x = FFN(i, hidden_dim, ffn_hidden_dim, (string) "model.layers." + std::to_string(layer) + ".mlp");
        i = _RMSNorm({x}, hidden_dim, 1e-6, (string) "model.layers." + std::to_string(layer) + ".input_layernorm");
        // _SubgraphBegin(c);
    }
 
    // end loop
    // i = _RMSNorm({i}, hidden_dim, 1e-6, (string) "model.norm");
    i = _Quantize({i},  true, ".model.quantize");
    i = _LinearINT8({i}, hidden_dim, vocab_size, false, "output");
    i = _Dequantize({i}, true,  ".model.dequantize");
}

template <typename Dtype>
void fullTensor(shared_ptr<Tensor> input_tensor, Net net, vector<int> shape, Dtype value) {
    input_tensor->setBackend(net.backends()[BackendType::MLLM_QNN].get());
    input_tensor->setCtype(ChlType::BSHD);
    input_tensor->reshape(shape[0], shape[1], shape[2], shape[3]);
    input_tensor->alloc();
    input_tensor->fullData<Dtype>(value);
}

int main(int argc, char **argv) {
    
    int vocab_size = 50272;
    int hidden_dim = 4096;
    int ffn_hidden_dim = 16384;
    int mutil_head_size = 32;

    std::unique_ptr<Context> c_ptr(new Context());
    auto *c = c_ptr.get();
    opt(c, vocab_size, hidden_dim, ffn_hidden_dim, mutil_head_size, 1);

    BackendConfig bn;
    QNNOptNet net(bn, c);
    net.convert(c->sub_param_, BackendType::MLLM_QNN);

    // ParamLoader param_loader(model_path);
    MockLoader param_loader("");
    QNNExecutor ex(&param_loader);

    shared_ptr<Tensor> input = std::make_shared<Tensor>();
    fullTensor(input, net, {1, 1, 32, hidden_dim}, 2.f);
    ex.setup(&net);

    for (int i=0; i<1; i++) {
        ex.run(&net, {input});
    }
    

    /*
    cmdline::parser cmdParser;
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "./vocab/llama_vocab.mllm");
    cmdParser.add<string>("model", 'm', "specify mllm model path", false, "./models/llama-7b-smoothquant.mllm");
    cmdParser.add<int>("limits", 'l', "max KV cache size", false, 400);
    cmdParser.add<int>("thread", 't', "num of threads", false, 4);
    cmdParser.parse_check(argc, argv);

    string vocab_path = cmdParser.get<string>("vocab");
    string model_path = cmdParser.get<string>("model");
    int tokens_limit = cmdParser.get<int>("limits");
    int thread_num = cmdParser.get<int>("thread");

    auto tokenizer = BPETokenizer(vocab_path);


    vector<string> in_strs = {
        " Hello, who are you?",
    };
    // " What can you do?",
    // "Please introduce Beijing University of Posts and Telecommunications."};
    shared_ptr<Tensor> input = std::make_shared<Tensor>();

    // input->reshape(1, 1, 1, hidden_dim);
    // input->setBackend(net.backends()[MLLM_QNN].get());
    // input->alloc();

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
        BPETokenizer::token2Tensor(&net, tokens_id, input);
        std::cout << "[Q] " << in_str << std::endl;
        std::cout << "[A] " << std::flush;
        for (int step = 0; step < 100; step++) {
            ex.run(&net, {input});
            auto result = ex.result();
            result[0]->printShape();
            // --------- JUST RUN FOR ONE TOKEN ---------
            break;
            auto token_idx = postProcessing(result[0], input);
            if (token_idx == 2) { // "</s>"
                break;
            }
            // auto out_token = tokenizer.detokenize({token_idx});
            // std::cout << out_token << std::flush;
        }
        printf("\n");
    }

    ex.perf();

    // free memory
    for (auto *op : c->net_ops) {
        delete op;
    }
    for (auto *tensor : c->net_tensors) {
        delete tensor;
    }
    */
    return 0;
}
