//
// Created by xwk on 24-10-23.
//
#include "models/bert/configuration_bert.hpp"
#include "models/bert/modeling_bert.hpp"
#include "models/bert/tokenization_bert.hpp"
#include "cmdline.h"
#include <vector>

/*
 * an intent to support gte-small BertModel to do text embedding
 * current implementation is just a very basic example with a simple WordPiece tokenizer and a simple BertModel
 * not support batch embedding
 * */

int main(int argc, char *argv[]) {
    cmdline::parser cmdParser;
    cmdParser.add<string>("model", 'm', "specify mllm model path", false, "../models/gte-small-fp32.mllm");
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "../vocab/gte_vocab.mllm");
    cmdParser.add<int>("thread", 't', "num of threads", false, 4);
    cmdParser.parse_check(argc, argv);

    string model_path = cmdParser.get<string>("model");
    string vocab_path = cmdParser.get<string>("vocab");
    CPUBackend::cpu_threads = cmdParser.get<int>("thread");

    BertTokenizer tokenizer(vocab_path, true);
    auto config = BertConfig();
    auto model = BertModel(config);
    model.load(model_path);

    string text = "Help me set an alarm at 21:30";
    vector<string> texts = {text, text};
    for (auto &text : texts) {
        auto inputs = tokenizer.tokenizes(text);
        auto res = model({inputs[0], inputs[1], inputs[2]})[0];
        res.printData<float>();
    }

    return 0;
}
