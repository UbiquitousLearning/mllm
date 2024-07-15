#include <iostream>
#include <csignal>
#include <memory>
#include "Executor.hpp"
#include "Types.hpp"
#include "backends/QNN/QNNOptNet.hpp"
#include "cmdline.h"
#include "Net.hpp"
#include "tokenizers/BPE/Bpe.hpp"
#include "backends/QNN/QNNExecutor.hpp"
#include "TestNet.hpp"
#include "modeling_opt_npuxpu.hpp"

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

int main(int argc, char **argv) {
    cmdline::parser cmdParser;
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "./vocab/vocab_opt.mllm");
    cmdParser.add<string>("model", 'm', "specify mllm model path", false, "./models/opt-1.3b-head-static-int8.mllm");
    cmdParser.add<int>("limits", 'l', "max KV cache size", false, 400);

    cmdParser.add<int>("thread", 't', "num of threads", false, 4);
    cmdParser.add<int>("seq", 's', "num of threads", false, 32);
    cmdParser.add<int>("chunk", 'c', "use chunk execute", false, 1);
    cmdParser.add<int>("head", 'h', "num of heads", false, 32);

    cmdParser.add<int>("ffn", 'f', "size of ffn hidden size", false, 8192);
    cmdParser.add<int>("hds", 'd', "size of hidden size", false, 2048);


    cmdParser.parse_check(argc, argv);

    string vocab_path = cmdParser.get<string>("vocab");
    string model_path = cmdParser.get<string>("model");
    int tokens_limit = cmdParser.get<int>("limits");
    int thread_num = cmdParser.get<int>("thread");
    int seqLength = cmdParser.get<int>("seq");
    int executeType = cmdParser.get<int>("chunk");
    int head_num = cmdParser.get<int>("head");
    int chunk = 1;

    if (executeType == 1) 
        chunk = 2;

    auto tokenizer = BPETokenizer(vocab_path);
    std::unordered_map<string, unsigned> merge_rank;
    auto merge_file = std::ifstream("./vocab/opt-merges.txt");
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

    int vocab_size = 50272;

    int hidden_dim = cmdParser.get<int>("hds");;
    int ffn_hidden_dim = cmdParser.get<int>("ffn");;
    // int mutil_head_size = 32;


    std::unique_ptr<Context> npu_ctx_ptr(new Context());
    auto *npu_ctx = npu_ctx_ptr.get();
    std::unique_ptr<Context> cpu_ctx_ptr(new Context());
    auto *cpu_ctx = cpu_ctx_ptr.get();


    // cache_max should be longer than seqLength
    modeling::opt_npu(npu_ctx, vocab_size, hidden_dim, ffn_hidden_dim, head_num, tokens_limit, seqLength, chunk);
    modeling::opt_cpu(cpu_ctx, vocab_size, hidden_dim, ffn_hidden_dim, head_num, tokens_limit);

    BackendConfig bn;
    QNNOptNet npuNet(bn, npu_ctx);
    npuNet.convert(npu_ctx, BackendType::MLLM_QNN, thread_num);
    Net cpuNet(bn);
    cpuNet.convert(cpu_ctx->sub_param_, BackendType::MLLM_CPU, thread_num);

    ParamLoader npu_prefill_param_loader(model_path);
    ParamLoader cpu_decoding_param_loader("./models/opt-1.3b-q40.mllm");

    QNNExecutor *npuExePtr;
    if (executeType == 1) {
        std::cout << "use pipeline execute" << std::endl;
        npuExePtr = new QNNPipelineExecutor(&npu_prefill_param_loader);
    } else {
        std::cout << "use normal execute" << std::endl;
        npuExePtr = new QNNExecutor(&npu_prefill_param_loader);
    }
    auto &npuExe = *npuExePtr;
    npuExe.setup(&npuNet);

    Executor cpuExe(&cpu_decoding_param_loader);
    cpuExe.setup(&cpuNet);

    vector<string> in_strs = {
        "Hello, who are you?",
    };
    // " What can you do?",
    // "Please introduce Beijing University of Posts and Telecommunications."};
    shared_ptr<Tensor> input = std::make_shared<Tensor>();

    for (int str_i = 0; str_i < in_strs.size(); ++str_i) {
        auto in_str = in_strs[str_i];
        in_str = mllm::Tokenizer::replaceString(in_str, ' ', "Ġ");
        tokenizer.setSpecialToken("</s>", "");

        auto tokens_id = vector<token_id_t>();
        tokenizer.tokenize(in_str, tokens_id, true);
        if (str_i > 0) {
            tokens_id[0] = 13;
        }
        // delete the last end token
        tokens_id.pop_back();

        // resize to the expected seqLength, the seq will be then splited to chunks
        // tokens_id.resize(0);
        tokens_id.resize(seqLength);

        BPETokenizer::token2Tensor(&npuNet, tokens_id, input);

        std::cout << "[Q] " << in_str << std::endl;
        std::cout << "[A] " << std::flush;

        do {
            // 1: Prefill stage using NPU chunk execute
            npuExe.run(npu_ctx, &npuNet, {input});
            auto result = npuExe.result();

            // ----------------- TEST PRINT --------------------
            // std::cout << "result size: " << result.size() << std::endl;
            // shared_ptr<Tensor> t;
            // if(result.size() > 1) {
            //     std::cout << "getting V" << std::endl;
            //     t = result[2];
            // } else {
            //     t = result[0];
            // }
            // auto data = t->ptrAt<float>(0,0,0,0);
            // for (int i = 0; i < t->sequence(); i++) {
            //     std::cout << data[i] << " ";
            // }

            result[0]->printData<float>();
            // result[0]->printShape();
            // auto token_idx = postProcessing(result[0], input);
            // if (token_idx == 2) { // "</s>"
            //     break;
            // }
            // auto out_token = tokenizer.detokenize({token_idx});
            // std::cout << out_token << std::flush;

            // // 2: Decoding stage using CPU execute
            // for (int step = 1; step < 100; step++) {
            //     cpuExe.run(&cpuNet, {input});
            //     auto result = cpuExe.result();
            //     auto token_idx = postProcessing(result[0], input);
            //     if (token_idx == 2) { // "</s>"
            //         break;
            //     }
            //     auto out_token = tokenizer.detokenize({token_idx});
            //     std::cout << out_token << std::flush;
            // }
        } while (false);
        printf("\n");
    }

    npuExe.perf();

    // free memory
    for (auto *op : npu_ctx->net_ops) {
        delete op;
    }
    for (auto *tensor : npu_ctx->net_tensors) {
        delete tensor;
    }

    return 0;
}
