#include <iostream>
#include <csignal>
#include <memory>
#include <vector>
#include "Executor.hpp"
#include "Types.hpp"
#include "backends/QNN/QNNOptNet.hpp"
#include "cmdline.h"
#include "Net.hpp"
#include "express/Express.hpp"
#include "tokenizers/BPE/Bpe.hpp"
#include "backends/QNN/QNNExecutor.hpp"
#include "modeling_opt_npuxpu.hpp"
#include "modeling_qwen_npuxpu.hpp"

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

template <typename Dtype>
void fullTensor(shared_ptr<Tensor> input_tensor, Net net, vector<int> shape, Dtype value) {
    input_tensor->setBackend(net.backends()[BackendType::MLLM_QNN].get());
    input_tensor->setCtype(ChlType::BSHD);
    input_tensor->reshape(shape[0], shape[1], shape[2], shape[3]);
    input_tensor->alloc();
    input_tensor->fullData<Dtype>(value);
}

NetTensor *Attention(NetTensor *x, int embedding_size, int hidden_size, int head_size, int cache_max, string name) {
    auto *q = _Linear({x}, embedding_size, hidden_size * head_size, true, name + ".q_proj");

    auto *k = _Linear({x}, embedding_size, hidden_size * head_size, true, name + ".k_proj");
    auto *v = _Linear({x}, embedding_size, hidden_size * head_size, true, name + ".v_proj");

    q = q->view(-1, head_size, -1, hidden_size);
    k = k->view(-1, head_size, -1, hidden_size);
    v = v->view(-1, head_size, -1, hidden_size);

    q = _RoPE({q}, HFHUBROPE, name + ".q_rope", 1000000, 32768);
    k = _RoPE({k}, HFHUBROPE, name + ".k_rope", 1000000, 32768);
    k = _KVCache({k}, cache_max, name + ".k_cache");
    v = _KVCache({v}, cache_max, name + ".v_cache");
    auto *qk = _Matmul({q, k}, false, true, name + ".qk");
    qk = *qk / std::sqrt(hidden_size);

    qk = _Causalmask({qk}, name + ".mask");

    qk = _Softmax({qk}, DIMENSION, name + ".softmax");

    auto *o = _Matmul({qk, v}, false, false, name + ".qkv");

    o = o->view(-1, 1, -1, hidden_size * head_size);
    o = _Linear({o}, hidden_size * head_size, embedding_size, false, name + ".o_proj");
    return o;
}
NetTensor *FFN(NetTensor *i, int hidden_dim, int ffn_hidden_dim, string name) {
    auto *x = _Linear({i}, hidden_dim, ffn_hidden_dim, false, name + ".gate_proj");
    x = _SiLU({x}, name + ".silu");
    auto *y = _Linear({i}, hidden_dim, ffn_hidden_dim, false, name + ".up_proj");
    x = *x * y; // x = _Mul( {x, y}, name+".dot");
    x = _Linear({x}, ffn_hidden_dim, hidden_dim, false, name + ".down_proj");
    return x;
}
void qwen_model(Context *c, int vocab_size = 32000, int hidden_dim = 4096, int ffn_hidden_dim = 11008, int mutil_head_size = 32, int cache_max = 200) {
    auto *i = _Input(c);
    i = _Embedding({i}, vocab_size, hidden_dim, (string) "model.embed_tokens");

    for (int layer = 0; layer < 24; ++layer) {
        auto res = _RMSNorm({i}, hidden_dim, 1e-6, (string) "model.layers." + std::to_string(layer) + ".input_layernorm");

        auto tmp = Attention(res, hidden_dim, hidden_dim / mutil_head_size, mutil_head_size, cache_max, (string) "model.layers." + std::to_string(layer) + ".self_attn");

        i = *tmp+i;

        res = _RMSNorm({i}, hidden_dim, 1e-6, (string) "model.layers." + std::to_string(layer) + ".post_attention_layernorm");

        i = *FFN(res, hidden_dim, ffn_hidden_dim, (string) "model.layers." + std::to_string(layer) + ".mlp") + i;
    }
    i = _RMSNorm({i}, hidden_dim, 1e-6, (string) "model.norm");
    i = _Linear({i}, hidden_dim, vocab_size, false, "lm_head");
}

int main(int argc, char **argv) {
    cmdline::parser cmdParser;
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "./vocab/vocab-qwen.mllm");

    cmdParser.add<int>("limits", 'l', "max KV cache size", false, 1024);

    cmdParser.add<int>("thread", 't', "num of threads", false, 4);
    cmdParser.add<int>("seq", 's', "seqenth length", false, 32);
    cmdParser.add<int>("chunk", 'c', "use chunk execute", false, 1);
    cmdParser.add<int>("head", 'h', "num of heads", false, 16);

    cmdParser.add<int>("ffn", 'f', "size of ffn hidden size", false, 5504);
    cmdParser.add<int>("hds", 'd', "size of hidden size", false, 2048);

    cmdParser.parse_check(argc, argv);

    const string cpu_model_path = "./models/qwen-1.8b-chat-q4k-fp32.mllm";
    const string merge_file_path = "./vocab/merges-qwen.txt";

    string vocab_path = cmdParser.get<string>("vocab");
    int tokens_limit = cmdParser.get<int>("limits");
    int thread_num = cmdParser.get<int>("thread");
    int seqLength = cmdParser.get<int>("seq");
    int executeType = cmdParser.get<int>("chunk");
    int head_num = cmdParser.get<int>("head");

    auto tokenizer = BPETokenizer(vocab_path);
    std::unordered_map<string, unsigned> merge_rank;
    auto merge_file = std::ifstream(merge_file_path);
    std::string line;
    unsigned rank = 0;
    while (std::getline(merge_file, line)) {
        if (line.empty()) {
            continue;
        }
        if (line[0] == '#') {
            continue;
        }
        merge_rank[line] = rank;
        rank++;
    }
    tokenizer.setMergeRank(merge_rank);

    int vocab_size = 151936;

    int hidden_dim = cmdParser.get<int>("hds");
    int ffn_hidden_dim = cmdParser.get<int>("ffn");

    std::unique_ptr<Context> cpu_ctx_ptr(new Context());
    auto *cpu_ctx = cpu_ctx_ptr.get();

    // cache_max should be longer than seqLength
    qwen_model(cpu_ctx, vocab_size, hidden_dim, ffn_hidden_dim, head_num, tokens_limit);

    BackendConfig bn;
    Net cpuNet(bn);
    cpuNet.convert(cpu_ctx->sub_param_, BackendType::MLLM_CPU, thread_num);

    ParamLoader cpu_decoding_param_loader(cpu_model_path);


    Executor cpuExe(&cpu_decoding_param_loader);
    cpuExe.setup(&cpuNet);

    vector<string> in_strs = {
        "Hello, who are you?",
        // "Hello, who are you?",
    };
    // " What can you do?",
    // "Please introduce Beijing University of Posts and Telecommunications."};
    shared_ptr<Tensor> input = std::make_shared<Tensor>();

    for (int str_i = 0; str_i < in_strs.size(); ++str_i) {
        auto in_str = in_strs[str_i];
        in_str = mllm::Tokenizer::replaceString(in_str, ' ', "Ġ");
        tokenizer.setSpecialToken("</s>", "");

        // auto tokens_id = vector<token_id_t>({151643, 21927, 11, 14623, 546, 9330, 30});
        auto tokens_id = vector<token_id_t>();
        tokenizer.tokenize(in_str, tokens_id, false, true, "");
        if (str_i > 0) {
            tokens_id[0] = 13;
        }

        BPETokenizer::token2Tensor(&cpuNet, tokens_id, input);


        std::cout << "[Q] " << in_str << std::endl;
        std::cout << "[A] " << std::flush;

        vector<string> answers;

        for (int step = 0; step < 100; step++) {
            cpuExe.run(&cpuNet, {input});
            auto result = cpuExe.result();

            auto token_idx = postProcessing(result[0], input);
            if (token_idx == 151645) { // "</s>"
                break;
            }
            auto out_token = tokenizer.detokenize({token_idx});
            // replace "Ġ" with " " using std
            if(out_token.find("Ġ") != std::string::npos)
                out_token = out_token.replace(out_token.find("Ġ"), string("Ġ").length(), " ");
            std::cout << out_token << std::flush;
            answers.push_back(out_token);
        }

        for (auto answer : answers) {
            std::cout << answer;
        }
        printf("\n");
    }

    cpuExe.perf();

    // free memory
    // for (auto *op : npu_ctx->net_ops) {
    //     delete op;
    // }
    // for (auto *tensor : npu_ctx->net_tensors) {
    //     delete tensor;
    // }

    return 0;
}
