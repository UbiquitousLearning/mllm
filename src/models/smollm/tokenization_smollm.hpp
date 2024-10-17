/**
 * @file tokenization_smollm.hpp
 * @author Chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2024-09-25
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef DCLMTOKENIZATION_SMOLLM_HPP
#define DCLMTOKENIZATION_SMOLLM_HPP

#include "tokenizers/BPE/Bpe.hpp"
#include "tokenizers/Tokenizer.hpp"
#include "tokenizers/Unicode.hpp"
#include <algorithm>
#include <unordered_map>

// unicode
#include <codecvt>

using namespace mllm;

#define UTF8(x) any_to_utf8(x)
#define CHR(x) __chr(x)
#define ORD(x) __ord(x)

static std::string any_to_utf8(std::string s) {
    // the original input is utf-8 already
    return s;
}

static std::string __chr(int v) {
    std::wstring wString(1, v);
    std::wstring_convert<std::codecvt_utf8<wchar_t>> convert;
    std::string utf8 = convert.to_bytes(wString);
    return utf8;
}

static std::vector<int> __ord(std::string v) {
    std::vector<int> ret;
    std::wstring_convert<std::codecvt_utf8<wchar_t>> convert;
    std::wstring utf8str = convert.from_bytes(v);
    for (auto i = 0; i < utf8str.length(); ++i) ret.emplace_back(utf8str[i]);
    return ret;
}

static const std::string PAT_STR = R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?:$|[^\S])|\s+)";
static const std::string SPLIT_PAT_STR = R"(<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>)";
static const std::vector<std::string> FIXED_PAT_STRS = {
    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
};

class SmolLMTokenizer final {
public:
    explicit SmolLMTokenizer(const std::string &vocab_file, const std::string &merge_file, bool split_special_tokens = false) :
        split_special_tokens_(split_special_tokens) {
        Module::initBackend(MLLM_CPU);
        tokenizer = new BPETokenizer(vocab_file);

        // init byte encoder
        std::vector<int> bs;
        for (int i = 33 /*!*/; i < 127 /*~*/; ++i) bs.emplace_back(i);
        for (int i = 161 /*¡*/; i < 173 /*¬*/; ++i) bs.emplace_back(i);
        for (int i = 174 /*®*/; i < 256 /*ÿ*/; ++i) bs.emplace_back(i);
        std::vector<int> cs = bs; // this is deep copy
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
                bs.push_back(b);
                cs.push_back(256 + n);
                n++;
            }
        }
        assert(bs.size() == cs.size() && "In init byte encoder, the bs and cs size should be same.");
        for (auto i = 0U; i < bs.size(); ++i) {
            byte_encoder_[bs[i]] = CHR(cs[i]);
            byte_decoder_[CHR(cs[i])] = bs[i];
        }

        // init bpe ranks
        auto merge_file_stream = std::ifstream(merge_file);
        if (!merge_file_stream.good()) {
            std::cout << "merge file is broken\n";
            exit(0);
        }
        std::string line;
        unsigned rank = 0;
        while (std::getline(merge_file_stream, line)) {
            if (line.empty()) {
                continue;
            }
            if (line[0] == '#') {
                continue;
            }
            bpe_ranks_[line] = rank;
            rank++;
        }
        tokenizer->setMergeRank(bpe_ranks_);
    }

    ~SmolLMTokenizer() {
        delete tokenizer;
    }

    std::vector<std::string> stringSplit(const std::string &str, char delim) {
        std::size_t previous = 0;
        std::size_t current = str.find(delim);
        std::vector<std::string> elems;
        while (current != std::string::npos) {
            if (current > previous) {
                elems.push_back(str.substr(previous, current - previous));
            }
            previous = current + 1;
            current = str.find(delim, previous);
        }
        if (previous != str.size()) {
            elems.push_back(str.substr(previous));
        }
        return elems;
    }

    std::vector<std::string> _splitWithDelimiters(const std::string &str, const std::vector<std::string> &delimiters) {
        std::string s = str;
        std::vector<std::string> result;
        size_t pos = 0;
        auto isDelimiter = [&](size_t currentPos) {
            for (const auto &delimiter : delimiters) {
                if (currentPos + delimiter.length() <= s.length() && s.substr(currentPos, delimiter.length()) == delimiter) {
                    return true;
                }
            }
            return false;
        };

        while (pos < s.length()) {
            if (isDelimiter(pos)) {
                if (pos != 0) {
                    result.push_back(s.substr(0, pos));
                }
                size_t delimiterLength = delimiters.front().length();
                for (const auto &delimiter : delimiters) {
                    if (s.substr(pos, delimiter.length()) == delimiter) {
                        delimiterLength = delimiter.length();
                        result.push_back(delimiter);
                        break;
                    }
                }
                pos += delimiterLength;
                s = s.substr(pos);
                pos = 0;
            } else {
                ++pos;
            }
        }

        if (!s.empty()) {
            result.push_back(s);
        }

        return result;
    }

    Tensor tokenize(std::string &text) {
        std::vector<token_id_t> ret;

        if (split_special_tokens_) {
            const auto word_collection = unicode_regex_split(text, FIXED_PAT_STRS);
            for (auto &piece : word_collection) {
                // look up table
                // std::string token;
                // for (auto b : UTF8(piece)) token += byte_encoder_[b];

                // using bpe
                std::vector<token_id_t> tmp;
                tokenizer->tokenize(piece, tmp, false, true, "");
                ret.insert(ret.end(), tmp.begin(), tmp.end() - 1);
            }
        } else {
            auto parts = _splitWithDelimiters(text, special_tokens);
            // for (auto p : parts) {
            //     std::cout << "\"" << p << "\"" << std::endl;
            // }
            for (auto &p : parts) {
                if (std::find(special_tokens.begin(), special_tokens.end(), p) != special_tokens.end()) {
                    std::string token;
                    for (auto b : UTF8(p)) token += byte_encoder_[b];

                    std::vector<token_id_t> tmp;
                    tokenizer->tokenize(token, tmp, false, special_tokens, true);
                    ret.insert(ret.end(), tmp.begin(), tmp.end() - 1);
                } else {
                    const auto word_collection = unicode_regex_split(p, FIXED_PAT_STRS);
                    for (auto &piece : word_collection) {
                        // look up table
                        // std::string token;
                        // for (auto b : UTF8(piece)) token += byte_encoder_[b];

                        // using bpe
                        std::vector<token_id_t> tmp;
                        tokenizer->tokenize(piece, tmp, false, true, "");
                        assert(!tmp.empty());
                        ret.insert(ret.end(), tmp.begin(), tmp.end() - 1);
                    }
                }
            }
        }

        return Tokenizer::tokens2Input(ret);
    }

    std::pair<int, Tensor> tokenizeWithPadding(std::string &text, int seqLength, int vocab_size) {
        std::vector<token_id_t> ret;

        if (split_special_tokens_) {
            const auto word_collection = unicode_regex_split(text, FIXED_PAT_STRS);
            for (auto &piece : word_collection) {
                // look up table
                // std::string token;
                // for (auto b : UTF8(piece)) token += byte_encoder_[b];

                // using bpe
                std::vector<token_id_t> tmp;
                tokenizer->tokenize(piece, tmp, false, true, "");
                ret.insert(ret.end(), tmp.begin(), tmp.end() - 1);
            }
        } else {
            auto parts = _splitWithDelimiters(text, special_tokens);
            // for (auto p : parts) {
            //     std::cout << "\"" << p << "\"" << std::endl;
            // }
            for (auto &p : parts) {
                if (std::find(special_tokens.begin(), special_tokens.end(), p) != special_tokens.end()) {
                    std::string token;
                    for (auto b : UTF8(p)) token += byte_encoder_[b];

                    std::vector<token_id_t> tmp;
                    tokenizer->tokenize(token, tmp, false, special_tokens, true);
                    ret.insert(ret.end(), tmp.begin(), tmp.end() - 1);
                } else {
                    const auto word_collection = unicode_regex_split(p, FIXED_PAT_STRS);
                    for (auto &piece : word_collection) {
                        // look up table
                        // std::string token;
                        // for (auto b : UTF8(piece)) token += byte_encoder_[b];

                        // using bpe
                        std::vector<token_id_t> tmp;
                        tokenizer->tokenize(piece, tmp, false, true, "");
                        assert(!tmp.empty());
                        ret.insert(ret.end(), tmp.begin(), tmp.end() - 1);
                    }
                }
            }
        }

        auto realLength = ret.size();
        ret.resize(seqLength, vocab_size);
        return std::make_pair(realLength, Tokenizer::tokens2Input(ret));
    }

    std::string _byte_decode_(const std::string &text) {
        std::string ret;
        auto _ = ORD(text);
        for (auto i : _) ret += byte_decoder_[CHR(i)];
        return ret;
    }

    std::string detokenize(const std::vector<token_id_t> &tokens) {
        return _byte_decode_(tokenizer->detokenize(tokens));
    }

    std::pair<std::string, unsigned> detokenize(Tensor &result) {
        assert(result.batch() == 1);
        assert(result.head() == 1);
        vector<float> scores;
        for (int i = 0; i < result.dimension(); ++i) {
            auto value = result.dataAt<float>(0, 0, result.sequence() - 1, i);
            scores.push_back(value);
        }
        auto token_idx = this->argmax(scores);
        return {_byte_decode_(tokenizer->detokenize({token_idx})), token_idx};
    }

private:
    unsigned int argmax(const std::vector<float> &scores) {
        if (scores.empty()) {
            throw std::invalid_argument("Input vector is empty");
        }
        return std::max_element(scores.begin(), scores.end()) - scores.begin();
    }

public:
    bool split_special_tokens_ = false;
    BPETokenizer *tokenizer;
    std::unordered_map<int, std::string> byte_encoder_;
    std::unordered_map<std::string, int> byte_decoder_;
    std::unordered_map<std::string, unsigned int> bpe_ranks_;
    token_id_t eos_id_ = 0, bos_id_ = 0;
    std::vector<std::string> special_tokens = {
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<repo_name>",
        "<reponame>",
        "<file_sep>",
        "<filename>",
        "<gh_stars>",
        "<issue_start>",
        "<issue_comment>",
        "<issue_closed>",
        "<jupyter_start>",
        "<jupyter_text>",
        "<jupyter_code>",
        "<jupyter_output>",
        "<jupyter_script>",
        "<empty_output>",
    };
};

#undef UTF8
#undef CHR
#undef ORD

#endif // TOKENIZATION_SMOLLM_HPP