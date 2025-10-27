#ifndef TOKENIZATION_BERT_HPP
#define TOKENIZATION_BERT_HPP

#include "tokenizers/Tokenizer.hpp"
#include "tokenizers/WordPiece/WordPiece.hpp"

// unicode
#include <string>
#include <vector>

using namespace mllm;

class BertTokenizer final : public WordPieceTokenizer {
public:
    explicit BertTokenizer(const std::string &vocab_file, bool add_special_tokens = true) :
        WordPieceTokenizer(vocab_file) {
        Module::initBackend(MLLM_CPU);
        _add_special_tokens = add_special_tokens;
        this->add_special_tokens({"[PAD]", "[CLS]", "[SEP]", "[MASK]"});
    }
    std::vector<Tensor> tokenizes(const std::string &text) override {
        string new_text;
        if (_add_special_tokens) {
            new_text = "[CLS] " + text + " [SEP]";
        }
        auto tokens_id = vector<token_id_t>();
        WordPieceTokenizer::tokenize(new_text, tokens_id, false);
        auto tokens_type = vector<token_id_t>(tokens_id.size(), 0);
        auto position_ids = vector<token_id_t>(tokens_id.size());
        for (size_t i = 0; i < tokens_id.size(); i++) {
            position_ids[i] = i;
        }
        return {
            tokens2Input(tokens_id, "input_tokens"),
            tokens2Input(tokens_type, "input_tokens_type"),
            tokens2Input(position_ids, "input_position_ids")};
    }

private:
    bool _add_special_tokens;
};

#endif //! TOKENIZATION_BERT_HPP