//
// Created by xwk on 24-10-23.
//
#include "models/bert/configuration_bert.hpp"
#include "models/bert/modeling_bert.hpp"
#include "models/bert/tokenization_bert.hpp"

string vocab_file = "./vocab/gte_vocab.mllm";
string model_file = "./models/gte-small-fp32.mllm";

/*
 * an intent to support gte-small BertModel to do text embedding
 * current implementation is just a very basic example with a simple WordPiece tokenizer and a simple BertModel
 * not support batch embedding
 * */

int main(int argc, char *argv[]) {
    BertTokenizer tokenizer(vocab_file, true);
    string text = "Help me set an alarm at 21:30";
    auto [token_ids, type_ids, position_ids] = tokenizer.process(text);
    // token_ids.printData<float>();

    auto config = BertConfig();
    auto model = BertModel(config);
    model.load(model_file);

    auto res = model({token_ids, type_ids, position_ids})[0];

    res.printData<float>();

    return 0;
}
