#include <iostream>
#include <csignal>
#include <memory>
#include <vector>
#include "Executor.hpp"
#include "Types.hpp"
#include "backends/QNN/QNNOptNet.hpp"
#include "cmdline.h"
#include "Net.hpp"
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


    const string npu_model_path = "./models/Qwen1.5-1.8B-Chat_158_int8_biasfp.mllm";
    const string cpu_model_path = "./models/qwen-1.8b-chat-q4k-fp32.mllm";
    const string merge_file_path = "./vocab/merges-qwen.txt";

    string vocab_path = cmdParser.get<string>("vocab");
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

    std::unique_ptr<Context> npu_ctx_ptr(new Context());
    auto *npu_ctx = npu_ctx_ptr.get();
    std::unique_ptr<Context> cpu_ctx_ptr(new Context());
    auto *cpu_ctx = cpu_ctx_ptr.get();
    std::unique_ptr<Context> inter_ctx_ptr(new Context());
    auto *inter_ctx = inter_ctx_ptr.get();

    // cache_max should be longer than seqLength
    modeling::qwen_npu_t2(npu_ctx, vocab_size, hidden_dim, ffn_hidden_dim, head_num, tokens_limit, seqLength, chunk);
    modeling::qwen_npu_cpu_inter(inter_ctx, vocab_size, hidden_dim, ffn_hidden_dim, head_num, tokens_limit, seqLength, chunk);
    modeling::qwen_cpu_t2(cpu_ctx, vocab_size, hidden_dim, ffn_hidden_dim, head_num, tokens_limit);

    BackendConfig bn;
    QNNOptNet npuNet(bn, npu_ctx);
    npuNet.convert(npu_ctx, BackendType::MLLM_QNN, thread_num);
    Net interNet(bn);
    interNet.convert(inter_ctx->sub_param_, BackendType::MLLM_CPU, thread_num);
    Net cpuNet(bn);
    cpuNet.convert(cpu_ctx->sub_param_, BackendType::MLLM_CPU, thread_num);

    ParamLoader npu_prefill_param_loader(npu_model_path);
    ParamLoader cpu_decoding_param_loader(cpu_model_path);
    ParamLoader inter_param_loader(cpu_model_path);

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
    Executor interExe(&inter_param_loader);
    interExe.setup(&interNet);
    Executor cpuExe(&cpu_decoding_param_loader);
    cpuExe.setup(&cpuNet);

    vector<string> in_strs = {
        // "<|im_start|>system\nYou are a helpful assistant.<| im_end |>\n<| im_start |>user\nGive me a short introduction to large language model.<| im_end |>\n<| im_start |> assistant\n\n",
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
        tokenizer.tokenize(in_str, tokens_id, false, true, "");
        if (str_i > 0) {
            tokens_id[0] = 13;
        }
        // delete the last end token
        // tokens_id.pop_back();

        for (int ti = 0; ti < tokens_id.size(); ti++) {
            tokens_id[ti] = 9707;
            std::cout << tokens_id[ti] << std::endl;
        }

        int real_seq_length = tokens_id.size();

        std::cout << "real_seq_length: " << real_seq_length << std::endl;

        // resize to the expected seqLength, the seq will be then splited to chunks
        // tokens_id.resize(0);
        tokens_id.resize(seqLength);

        BPETokenizer::token2Tensor(&npuNet, tokens_id, input);

        std::cout << "[Q] " << in_str << std::endl;
        std::cout << "[A] " << std::flush;

        vector<string> answers;

        do {
            // 1: Prefill stage using NPU chunk execute
            npuExe.run(npu_ctx, &npuNet, {input});
            auto result = npuExe.result();

            result[0]->printData<float>();
            // exit(0);

            // inter model for prefill-decode
            interExe.run(&interNet, {result[0]});
            result = interExe.result();

            auto token_idx = postProcessing(result[0], input);
            if (token_idx == 2) { // "</s>"
                break;
            }
            auto out_token = tokenizer.detokenize({token_idx});
            std::cout << out_token << std::flush;
            answers.push_back(out_token);

            auto prefill_cpu_backend = dynamic_cast<CPUBackend *>(npuNet.backends()[MLLM_CPU].get());
            auto inter_cpu_backend = dynamic_cast<CPUBackend *>(interNet.backends()[MLLM_CPU].get());
            prefill_cpu_backend->setSequenceLength(real_seq_length);
            prefill_cpu_backend->switchDecodeTag();
            inter_cpu_backend->setSequenceLength(real_seq_length);
            inter_cpu_backend->switchDecodeTag();

            // // 2: Decoding stage using CPU execute
            for (int step = real_seq_length; step < 100; step++) {
                cpuExe.run(&cpuNet, {input});
                auto result = cpuExe.result();
                auto token_idx = postProcessing(result[0], input);
                if (token_idx == 2) { // "</s>"
                    break;
                }
                auto out_token = tokenizer.detokenize({token_idx});
                std::cout << out_token << std::flush;
                answers.push_back(out_token);

                if (step == real_seq_length) {
                    prefill_cpu_backend->switchDecodeTag();
                    inter_cpu_backend->switchDecodeTag();
                }
            }
        } while (false);
        printf("\n");

        for (auto answer : answers){
            std::cout << answer << " ";
        }
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
