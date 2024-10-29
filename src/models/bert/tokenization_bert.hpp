#ifndef TOKENIZATION_BERT_HPP
#define TOKENIZATION_BERT_HPP

#include "tokenizers/BPE/Bpe.hpp"
#include "tokenizers/Tokenizer.hpp"
#include "tokenizers/Unicode.hpp"
#include "tokenizers/WordPiece/WordPiece.hpp"
#include <algorithm>
#include <unordered_map>

// unicode
#include <codecvt>

using namespace mllm;


class BertTokenizer final : public WordPieceTokenizer {
public:
    explicit BertTokenizer(const std::string &vocab_file, bool add_special_tokens = true) :
        WordPieceTokenizer(vocab_file) {
        Module::initBackend(MLLM_CPU);
        _add_special_tokens = add_special_tokens;
        this->add_special_tokens({"[PAD]", "[CLS]", "[SEP]", "[MASK]"});
    }
    std::tuple<Tensor, Tensor, Tensor> process(std::string text){
        if (_add_special_tokens) {
            text = "[CLS] " + text + " [SEP]";
        }
        auto tokens_id = vector<token_id_t>();
        WordPieceTokenizer::tokenize(text, tokens_id, false);
//        printf("token: ");
//        for (auto &token_id : tokens_id) {
//            printf("%d ", token_id);
//        }
        printf("\n");
        auto tokens_type = vector<token_id_t>(tokens_id.size(), 0);
        auto position_ids = vector<token_id_t>(tokens_id.size());
        for (size_t i = 0; i < tokens_id.size(); i++) {
            position_ids[i] = i;
        }
        return {
            tokens2Input(tokens_id, "input_tokens"),
            tokens2Input(tokens_type, "input_tokens_type"),
            tokens2Input(position_ids, "input_position_ids")
        };
    }

private:
    bool _add_special_tokens;
};

#endif //! TOKENIZATION_BERT_HPP