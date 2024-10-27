//
// Created by Rongjie Yi on 24-3-8.
//

#ifndef PROCESSING_Phi3V_HPP
#define PROCESSING_Phi3V_HPP
#include "OpDefined.hpp"
#include "tokenizers/BPE/Bpe.hpp"
#include "processor/ClipPreProcess.hpp"
#include "tokenizers/Tokenizer.hpp"
#include <utility>
#include <regex>
#include <vector>

using namespace mllm;

class Phi3VProcessor final {
    Tensor img2Tensor(vector<vector<vector<float>>> img, string name = "input", BackendType type = MLLM_CPU) {
        int channel = img.size();
        int height = img[0].size();
        int width = img[0][0].size();
        Tensor tensor1(1, height, channel, width, Backend::global_backends[type], true);
        tensor1.setName(std::move(name));
        Tensor::tensor_status = TENSOR_STATIC_INIT;
        tensor1.setTtype(INPUT_TENSOR);
        for (int h = 0; h < height; ++h) {
            for (int c = 0; c < channel; ++c) {
                for (int w = 0; w < width; ++w) {
                    tensor1.setDataAt<float>(0, h, c, w, img[c][h][w]);
                }
            }
        }
        return tensor1;
    }
    Tensor imgpos2Tensor(vector<std::pair<size_t, size_t>> img_pos, string name = "input_img_pos", BackendType type = MLLM_CPU) {
        int num_imgs = img_pos.size();
        Tensor tensor1(1, 1, num_imgs, 2, type, true);
        tensor1.setName(std::move(name));
        Tensor::tensor_status = TENSOR_STATIC_INIT;
        tensor1.setTtype(INPUT_TENSOR);
        for (int i = 0; i < num_imgs; ++i) {
            tensor1.setDataAt<size_t>(0,0, i, 0, img_pos[i].first);
            tensor1.setDataAt<size_t>(0, 0,i, 1, img_pos[i].second);
        }
        return tensor1;
    }
    unsigned int argmax(const vector<float> &scores) {
        if (scores.empty()) {
            throw std::invalid_argument("Input vector is empty");
        }
        return std::max_element(scores.begin(), scores.end()) - scores.begin();
    }

    std::pair<size_t, std::string> find_special(const std::string &text, const std::vector<std::string> &special, size_t pos) {
        for (const std::string &delimiter : special) {
            size_t found = text.find(delimiter, pos);
            if ((found != std::string::npos)) {
                return {found, delimiter};
            }
        }
        return {std::string::npos, ""};
    }

    void tokenize(const std::string &text, std::vector<token_id_t> &tokens, const std::vector<std::string> &special, vector<int> &num_img_tokens, vector<std::pair<size_t, size_t>> &img_pos) {
        int cnt_imgs = 0;
        tokens.push_back(1);
        size_t startPos = 0;
        auto result = find_special(text, special, startPos);
        size_t found = result.first;
        std::string delimiter = result.second;
        while (found != std::string::npos) {
            vector<token_id_t> tokens_id = {};
            if (found > startPos) {
                tokenizer->tokenize(text.substr(startPos, found - startPos), tokens_id, true);
                tokens.insert(tokens.end(), tokens_id.begin() + 1, tokens_id.end() - 1);
            }
            std::string delimiter_;
            if (delimiter == "\n") {
                delimiter_ = "<0x0A>";
            } else {
                delimiter_ = delimiter;
            }
            const auto vocab_map = tokenizer->getVocabMap();
            auto result = vocab_map.find(delimiter_);
            if (result != tokenizer->getVocabMap().end()) {
                if(delimiter_ != "<|image|>"){
                    tokens.push_back(result->second);
                    startPos = found + delimiter.length();
                } else {
                    vector<mllm::token_id_t> img_token(num_img_tokens[cnt_imgs], 32044*(cnt_imgs+1));
                    tokens.insert(tokens.end(), img_token.begin(), img_token.end());
                    img_pos.push_back({found, found + num_img_tokens[cnt_imgs]});
                    startPos = found + num_img_tokens[cnt_imgs];
                    cnt_imgs++;
                }
            } else {
                startPos = found + delimiter.length();
            }
            
            auto result_ = find_special(text, special, startPos);
            found = result_.first;
            delimiter = result_.second;
        }
        if (startPos < text.length()) {
            vector<token_id_t> tokens_id = {};
            tokenizer->tokenize(text.substr(startPos), tokens_id, true);
            tokens.insert(tokens.end(), tokens_id.begin() + 1, tokens_id.end() - 1);
        }
    }

    BPETokenizer *tokenizer;
    ClipPreProcessor *clip_processor;

public:
    explicit Phi3VProcessor(const string &vocab_path, const string &merges_path) {
        Module::initBackend(MLLM_CPU);
        tokenizer = new BPETokenizer(vocab_path);
        std::unordered_map<string,unsigned> merge_rank;
        auto merge_file = std::ifstream(merges_path);
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
        tokenizer->setMergeRank(merge_rank);
    }
    

    vector<Tensor> process(string text, string img_path, int hw = 336,
                                  string img_name = "input_vision", string text_name = "input_text", BackendType type = MLLM_CPU) {
        auto tokens_ids = vector<vector<token_id_t>>();
        if (text[0] != ' ') {
            text = ' ' + text;
        }
        vector<mllm::token_id_t> tokens_id = {};
        if (img_path != "") {
            clip_processor = new ClipPreProcessor(tokenizer, hw, hw);
            clip_processor->PreProcessImages({std::move(img_path)}, hw, hw);
            auto images = clip_processor->pixel_values_[0];

            // shapes = [[im.size[1], im.size[0]] for im in elems]
            // num_img_tokens = [int((h//336*w//336+1)*144 + 1 + (h//336+1)*12) for h, w in shapes]

            auto img_tensor = img2Tensor(images, std::move(img_name), type);
            vector<int> num_img_tokens;
            for (auto &img : images) {
                int h = img.size();
                int w = img[0].size();
                num_img_tokens.push_back(int((h / 336 * w / 336 + 1) * 144 + 1 + (h / 336 + 1) * 12));
            }
            
            vector<std::pair<size_t, size_t>> img_pos;
            tokenize(BPETokenizer::replaceString(text, ' ', "▁"), tokens_id, {"<|image|>", "<pad>", "<|user|>", " <|end|>", "<|assistant|>", "\n"}, num_img_tokens, img_pos);
            tokens_ids.push_back(tokens_id);

            return {Tokenizer::tokens2Input(tokens_ids, std::move(text_name)), img_tensor, imgpos2Tensor(img_pos)};
        } else {
            tokenizer->tokenize(BPETokenizer::replaceString(text, ' ', "▁"), tokens_id, {"<|image|>", "<pad>", "<|user|>", " <|end|>", "<|assistant|>", "\n"});
            tokens_ids.push_back(tokens_id);
            return {Tokenizer::tokens2Input(tokens_ids, std::move(text_name))};
        }
        
        
    }

    

    std::string detokenize(const vector<token_id_t> &tokens) {
        return tokenizer->detokenize(tokens);
    }

    std::pair<std::string, unsigned> detokenize(Tensor &result) {
        assert(result.batch() == 1 && "Batch size of result is not 1. Which is not supported for now.");
        assert(result.head() == 1 && "The 3rd dim of result should be one. e.g.:[1, 1, seq, hidden]");
        vector<float> scores;
        int _dims = result.dimension();
        int _seq = result.sequence() - 1;
        for (int i = 0; i < _dims; ++i) {
            auto value = result.dataAt<float>(0, 0, _seq, i);
            scores.push_back(value);
        }
        auto token_idx = this->argmax(scores);
        auto text = tokenizer->detokenize({token_idx});
        text = std::regex_replace(text, std::regex("▁"), " ");
        return make_pair(text, token_idx);
    }

public:
    token_id_t pad_id = 32000, eos_id = 32000, bos_id = 1, user_id = 32010, assistant_id = 32001, end_id = 32007;
};
#endif // PROCESSING_Phi3V_HPP
