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
#include "TestNet.hpp"

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

std::vector<NetTensor *> CPUNPUAttention(Context *c, NetTensor *x, NetTensor *res, int embedding_size, int hidden_size, int head_size, int cache_max, string name, int seq) {
    // x = _Quantize({x}, true, (string)name + ".x.quantize");
    x = x->view(1, static_cast<int>(sqrt(seq)), static_cast<int>(sqrt(seq)), hidden_size * head_size);
    auto *q = _LinearINT8({x}, embedding_size, hidden_size * head_size, false, name + ".q_proj");
    auto *k = _LinearINT8({x}, embedding_size, hidden_size * head_size, false, name + ".k_proj");
    auto *v = _LinearINT8({x}, embedding_size, hidden_size * head_size, false, name + ".v_proj");
    q = q->view(1, head_size, seq, hidden_size);
    k = k->view(1, head_size, seq, hidden_size);
    v = v->view(1, head_size, seq, hidden_size);
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
    // q = _Dequantize({q}, true, (string)name + ".q_proj.dequantize");
    // k = _Dequantize({k}, true, (string)name + ".k_proj.dequantize");
    // v = _Dequantize({v}, true, (string)name + ".v_proj.dequantize");

    auto *qk = _MatmulINT8({q, k}, false, true, name + ".qk");

    
    // qk = _Dequantize({qk}, false, (string) name + ".qk.dequantize");

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
    
    o = o->view(1, static_cast<int>(sqrt(seq)), static_cast<int>(sqrt(seq)), hidden_size * head_size);
    res = res->view(-1, 1, -1, hidden_size * head_size);
    o = _LinearINT8({o}, hidden_size * head_size, embedding_size, false, name + ".out_proj");
    o = _Dequantize({o}, true, (string)name + ".out_proj.dequantize");
    return {o, res};
}

NetTensor *NPUAttention(Context *c, NetTensor *x, int embedding_size, int hidden_size, int head_size, int cache_max, string name) {
    // x = _Quantize({x}, true, (string)name + ".x.quantize");
    auto *q = _LinearINT8({x}, embedding_size, hidden_size * head_size, false, name + ".q_proj");
    auto *k = _LinearINT8({x}, embedding_size, hidden_size * head_size, false, name + ".k_proj");
    auto *v = _LinearINT8({x}, embedding_size, hidden_size * head_size, false, name + ".v_proj");
    q = q->view(-1, head_size, -1, hidden_size);
    k = k->view(-1, head_size, -1, hidden_size);
    v = v->view(-1, head_size, -1, hidden_size);
    // // q = _RoPE({q}, LLAMAROPE, name + ".q_rope");
    // k = _RoPE({k}, LLAMAROPE, name + ".k_rope");
    // k = _KVCache({k}, cache_max, name + ".k_cache");
    // v = _KVCache({v}, cache_max, name + ".v_cache");

    q = _Dequantize({q}, true, (string)name + ".q_proj.dequantize");
    k = _Dequantize({k}, true, (string)name + ".k_proj.dequantize");
    v = _Dequantize({v}, true, (string)name + ".v_proj.dequantize");

    auto *qk = _Matmul({q, k}, false, true, name + ".qk");
    // qk = _Dequantize({qk}, false, (string) name + ".qk.dequantize");

    // qk = *qk / std::sqrt(hidden_size);
    qk = _Causalmask({qk}, name + ".mask");
    qk = _Softmax({qk}, DIMENSION, name + ".softmax");

    auto *o = _Matmul({qk, v}, false, false, name + ".qkv");
    return o;

    // // // --------------------
    // // _SubgraphBegin(c);
    // // // --------------------

    // o = _Quantize({o}, true, (string)name + ".out_proj.quantize");
    // o = o->view(-1, 1, -1, hidden_size * head_size);
    // o = _LinearINT8({o}, hidden_size * head_size, embedding_size, false, name + ".out_proj");
    // o = _Dequantize({o}, true, (string)name + ".out_proj.dequantize");
    // return o;
}
NetTensor *FFN(Context *c, NetTensor *i, int hidden_dim, int ffn_hidden_dim, string name) {
    // auto *x = _Quantize({i}, true, (string)name + ".fc1.quantize");
    auto *x = i;
    x = _LinearINT8({i}, hidden_dim, ffn_hidden_dim, false, name + ".fc1");
    // x = _LinearINT8({i}, hidden_dim, ffn_hidden_dim, false, "model.decoder.layers.1.fc1");
    // x = _LinearINT8({i}, hidden_dim, ffn_hidden_dim, false, "model.decoder.layers.2.fc1");
    // x = _LinearINT8({i}, hidden_dim, ffn_hidden_dim, false, "model.decoder.layers.2.fc1");
    // // x = _Dequantize({x}, (string) name + ".relux.dequantize");
    x = _ReLU({x}, name + ".fc2.relu");
    // y = _ReLU({y}, "model.decoder.layers.5.fc2.relu");

    // x = _Quantize({x}, (string) name + ".relux.quantize");
    x = _LinearINT8({x}, ffn_hidden_dim, hidden_dim, false, name + ".fc2");
    // // // y = _LinearINT8({y}, ffn_hidden_dim, hidden_dim, false, "model.decoder.layers.5.fc2");
    x = _Dequantize({x}, true, (string)name + ".fc2.dequantize");
    // y = _Dequantize({y}, true, "model.decoder.layers.5.fc2.dequantize");

    // x = *x + y;
    return x;
}
void opt(Context *c, int vocab_size = 32000, int hidden_dim = 4096, int ffn_hidden_dim = 11008, int mutil_head_size = 32, int cache_max = 200, int seq = 1024) {
    auto *i = _Input(c);
    i = _Embedding({i}, vocab_size, hidden_dim, (string) "model.decoder.embed_tokens");
    // i = _LayerNorm({i}, hidden_dim, true, 1e-6, (string) "model.decoder.layers.0.self_attn_layer_norm");
    // _SubgraphBegin(c);
    // loop

    for (int layer = 0; layer < 1; ++layer) {

        // i = _KVCache({i}, cache_max, std::to_string(layer) + ".kvcache");
        // _SubgraphBegin(c, MLLM_CPU);

        if (layer != 0)
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
        
        auto ix = CPUNPUAttention(c, i, res, hidden_dim, hidden_dim / mutil_head_size, mutil_head_size, cache_max, (string) "model.decoder.layers." + std::to_string(layer) + ".self_attn", seq);

        i = ix[0];
        res = ix[1];

        i = i->view(1, 1, seq, hidden_dim);
        i = *i + res;        

        _SubgraphBegin(c, MLLM_CPU);
        res = i;
        // // auto *x = _LayerNorm({i}, hidden_dim, true, 1e-6, (string) "model.decoder.layers." + std::to_string(layer) + ".final_layer_norm");
        i = _LayerNorm({i}, hidden_dim, true, 1e-5, (string) "model.decoder.layers." + std::to_string(layer) + ".final_layer_norm");
        i = _Quantize({i}, true, (string) "model.decoder.layers." + std::to_string(layer) + ".fc1.quantize");

        i = i->view(-1, mutil_head_size, -1, hidden_dim / mutil_head_size);
        // res = res->view(-1, mutil_head_size, -1, hidden_dim / mutil_head_size);

        m = _MergeOutput({i, res}, (string)"model.decoder.layers." + std::to_string(layer) + ".fres_merge");


        _SubgraphBegin(c);

        s = _SplitInput({m}, true, 2, (string)"model.decoder.layers." + std::to_string(layer) + ".fres_split");

        i = s[0];
        res = s[1];
        res = res->view(-1, 1, -1, hidden_dim);

        i = i->view(1, static_cast<int>(sqrt(seq)), static_cast<int>(sqrt(seq)), hidden_dim);
        i = FFN(c, i, hidden_dim, ffn_hidden_dim, (string) "model.decoder.layers." + std::to_string(layer));

        i = i->view(1, 1, seq, hidden_dim);

        i = *i + res;
    }

    // end loop
    // _SubgraphBegin(c, MLLM_CPU);
    // i = _LayerNorm({i}, hidden_dim, true, 1e-5, (string) "model.decoder.final_layer_norm");
    // i = _Linear({i}, hidden_dim, vocab_size, false, "lm_head");
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
    cmdline::parser cmdParser;
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "./vocab/vocab_opt.mllm");
    cmdParser.add<string>("model", 'm', "specify mllm model path", false, "./models/opt-1.3b-head-static-int8.mllm");
    cmdParser.add<int>("limits", 'l', "max KV cache size", false, 400);
    // cmdParser.add<int>("thread", 't', "num of threads", false, 4);
    cmdParser.add<int>("seq", 's', "seq length", false, 1);
    cmdParser.add<int>("head", 'h', "num of heads", false, 32);
    cmdParser.add<int>("type", 't', "type of test", false, 13);
    cmdParser.parse_check(argc, argv);

    string vocab_path = cmdParser.get<string>("vocab");
    string model_path = cmdParser.get<string>("model");
    int tokens_limit = cmdParser.get<int>("limits");
    // int thread_num = cmdParser.get<int>("thread");
    int seqLength = cmdParser.get<int>("seq");
    int head_num = cmdParser.get<int>("head");
    int type = cmdParser.get<int>("type");

    auto tokenizer = BPETokenizer(vocab_path);
    std::unordered_map<string,unsigned> merge_rank;
    auto merge_file = std::ifstream("./vocab/opt-merges.txt");
    std::string line;
    unsigned rank=0;
    while (std::getline(merge_file, line)) {
        if (line.empty()) {
            continue;
        }
        if (line[0]=='#'){
            continue;
        }
        merge_rank[line]=rank;
        rank++;
    }
    tokenizer.setMergeRank(merge_rank);

    int vocab_size = 50272;
    int hidden_dim = 2048;
    int ffn_hidden_dim = 16384;
    // int mutil_head_size = 32;

    std::unique_ptr<Context> c_ptr(new Context());
    auto *c = c_ptr.get();

    // opt(c, vocab_size, hidden_dim, ffn_hidden_dim, head_num, 512);
    switch(type){
        case 1:
            linearTest2048(c, vocab_size, hidden_dim, ffn_hidden_dim, head_num, 512);
            break;
        case 2:
            linearTest11008(c, vocab_size, hidden_dim, ffn_hidden_dim, head_num, 512);
            break;
        case 3:
            attentionMinor(c, vocab_size, hidden_dim, ffn_hidden_dim, head_num, 512);
            break;
        case 4:
            attentionPlus(c, vocab_size, hidden_dim, ffn_hidden_dim, head_num, 512);
            break;
        case 5:
            ffnTest(c, vocab_size, hidden_dim, 11008, head_num, 512);
            break;
        case 6:
            linearTest4096(c, vocab_size, 4096, ffn_hidden_dim, head_num, 512);
            break;
        case 7:
            attentionMinor(c, vocab_size, 4096, ffn_hidden_dim, head_num, 512);
            break;
        case 8:
            attentionPlus(c, vocab_size, 4096, ffn_hidden_dim, head_num, 512);
            break;
        case 9:
            linearTest409616384(c, vocab_size, 4096, 16384, head_num, 512);
            break;
        case 10:
            ffnTest(c, vocab_size,4096, 16384, head_num, 512);
            break;
        case 11:
            linearTest409611008(c, vocab_size, 4096, 11008, head_num, 512);
            break;
        case 12:
            ffnTest(c, vocab_size, 4096, 11008, head_num, 512);
            break;
        case 13:
            opt(c, vocab_size, hidden_dim, ffn_hidden_dim, head_num, 512, seqLength);
            break;
    }

    BackendConfig bn;
    QNNOptNet net(bn, c);
    net.convert(c, BackendType::MLLM_QNN);

    // ParamLoader param_loader(model_path);
    ParamLoader param_loader(model_path);
    QNNExecutor ex(&param_loader);

    ex.setup(&net);

    vector<string> in_strs = {
        "Hello, who are you?",
    };
    // " What can you do?",
    // "Please introduce Beijing University of Posts and Telecommunications."};
    shared_ptr<Tensor> input = std::make_shared<Tensor>();

    for (int str_i = 0; str_i < in_strs.size(); ++str_i) {
        auto in_str = in_strs[str_i];
        in_str = mllm::Tokenizer::replaceString(in_str,' ',"Ġ");
        tokenizer.setSpecialToken("</s>","");

        auto tokens_id = vector<token_id_t>();
        tokenizer.tokenize(in_str, tokens_id, true);
        if (str_i > 0) {
            tokens_id[0] = 13;
        }
        // delete the last end token
        tokens_id.pop_back();

        tokens_id.resize(seqLength);

        BPETokenizer::token2Tensor(&net, tokens_id, input);
        // fullTensor(input, net, {1,1, seqLength, 1}, 2.f);
        // input->printData<float>();

        std::cout << "[Q] " << in_str << std::endl;
        std::cout << "[A] " << std::flush;
        for (int step = 0; step < 1; step++) {
            ex.run(c, &net, {input});
            //  ---------------------------------
            // ex.run(&net, {input});
            auto result = ex.result();
            result[0]->printShape();
            // result[0]->printData<float>();

            // for (int n = 0; n < 32 * 7 * 64; n++) {
            //     std::cout << static_cast<float>(result[0]->hostPtr<float>()[n]) << " ";
            //     if ((n+1) % 64 == 0)
            //         std::cout << std::endl;
            // }

            auto token_idx = postProcessing(result[0], input);
            if (token_idx == 2) { // "</s>"
                break;
            }
            auto out_token = tokenizer.detokenize({token_idx});
            std::cout << out_token << std::flush;
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

    return 0;
}
