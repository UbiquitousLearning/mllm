//
// Created by xwk on 24-10-23.
//
#include "models/bert/configuration_bert.hpp"
#include "models/bert/modeling_bert.hpp"
#include "models/bert/tokenization_bert.hpp"

string vocab_file = "vocab/all-MiniLM-L6-v2.mllm";
string model_file = "models/gte.mllm";

int main(int argc, char *argv[]){
    BertTokenizer tokenizer(vocab_file, false);
    string text = "Hello, my dog is cute.";
    auto [token_ids, type_ids, position_ids] = tokenizer.process(text);
    token_ids.printData<float>();

    auto config = BertConfig();
    auto model = BertModel(config);
    model.load(model_file);

    auto res = model({token_ids, type_ids, position_ids});
    res[0].printData<float>();
}
