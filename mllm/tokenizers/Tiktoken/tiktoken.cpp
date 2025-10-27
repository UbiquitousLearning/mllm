//
// Created by xwk on 25-1-8.
// This is a simple implementation of openai's tiktoken library in C++
//
#include <limits>
#include <unordered_map>
#include <vector>
#include <string_view>
#include <string>
#include <fstream>
#include <unordered_set>
#include <iostream>
#include <cassert>
#include <utility>

#include "tiktoken.hpp"
#include "tokenizers/Unicode.hpp"

namespace mllm {
// define a constant unsigned int MAX_RANK
const rank_t MAX_RANK = std::numeric_limits<rank_t>::max();

/*
fn _byte_pair_merge(ranks: &HashMap<Vec<u8>, Rank>, piece: &[u8]) -> Vec<(usize, Rank)> {
    // This is a vector of (start, rank).
    // The rank is of the pair starting at position start.
    let mut parts = Vec::with_capacity(piece.len() + 1);

    // Note that we hash bytes when indexing into `ranks`, not token pairs. As long as we train BPE
    // the way we currently do, this is equivalent. An easy way to break this would be to decouple
    // merge priority from token index or to prevent specific token merges.
    let mut min_rank: (Rank, usize) = (Rank::MAX, usize::MAX);
    for i in 0..piece.len() - 1 {
        let rank = *ranks.get(&piece[i..i + 2]).unwrap_or(&Rank::MAX);
        if rank < min_rank.0 {
            min_rank = (rank, i);
        }
        parts.push((i, rank));
    }
    parts.push((piece.len() - 1, Rank::MAX));
    parts.push((piece.len(), Rank::MAX));

    let get_rank = {
        #[inline(always)]
        |parts: &Vec<(usize, Rank)>, i: usize| {
            if (i + 3) < parts.len() {
                // Similar to `piece[i..i + 2]` above. The +3 is because we haven't yet deleted
                // parts[i + 1], see comment in the main loop.
                *ranks
                    .get(&piece[parts[i].0..parts[i + 3].0])
                    .unwrap_or(&Rank::MAX)
            } else {
                Rank::MAX
            }
        }
    };

    // If you have n parts and m merges, this does O(mn) work.
    // We could do something with a heap and do O(m log n) work.
    // n is often very small so considerations like cache-locality outweigh the algorithmic
    // complexity downsides of the `parts` vector.
    while min_rank.0 != Rank::MAX {
        let i = min_rank.1;
        // Update parts[i] and parts[i - 1] before removing parts[i + 1], since
        // `parts.remove(i + 1)` will thrash the cache.
        if i > 0 {
            parts[i - 1].1 = get_rank(&parts, i - 1);
        }
        parts[i].1 = get_rank(&parts, i);
        parts.remove(i + 1);

        min_rank = (Rank::MAX, usize::MAX);
        for (i, &(_, rank)) in parts[..parts.len() - 1].iter().enumerate() {
            if rank < min_rank.0 {
                min_rank = (rank, i);
            }
        }
    }
    parts
}
 */

typedef struct merge_t {
    size_t start;
    rank_t rank;
} merge_t;

rank_t find_rank(const view_merge_rank_t &ranks, const std::string_view &piece) {
    auto it = ranks.find(piece);
    if (it != ranks.end()) {
        return it->second;
    } else {
        return MAX_RANK;
    }
}

std::vector<merge_t> _byte_pair_merge(const view_merge_rank_t &ranks, const std::string &piece) {
    std::vector<merge_t> parts;
    parts.reserve(piece.size() + 1);

    std::pair<int, rank_t> min_rank = {std::numeric_limits<size_t>::max(), MAX_RANK};
    for (int i = 0; i < piece.size() - 1; ++i) {
        rank_t rank = find_rank(ranks, std::string_view(piece.data() + i, 2));
        if (rank < min_rank.second) {
            min_rank = {i, rank};
        }
        parts.push_back({static_cast<size_t>(i), rank});
    }
    parts.push_back({piece.size() - 1, MAX_RANK});
    parts.push_back({piece.size(), MAX_RANK});

    auto get_rank = [&](const std::vector<merge_t> &parts, size_t i) {
        if ((i + 3) < parts.size()) {
            return find_rank(ranks, std::string_view(piece.data() + parts[i].start, parts[i + 3].start - parts[i].start));
        } else {
            return MAX_RANK;
        }
    };

    while (min_rank.second != MAX_RANK) {
        int i = min_rank.first;
        if (i > 0) {
            parts[i - 1].rank = get_rank(parts, i - 1);
        }
        parts[i].rank = get_rank(parts, i);
        parts.erase(parts.begin() + i + 1);

        min_rank = {std::numeric_limits<size_t>::max(), MAX_RANK};
        for (int i = 0; i < parts.size() - 1; ++i) {
            if (parts[i].rank < min_rank.second) {
                min_rank = {i, parts[i].rank};
            }
        }
    }
    return parts;
}

/*
 pub fn byte_pair_encode(piece: &[u8], ranks: &HashMap<Vec<u8>, Rank>) -> Vec<Rank> {
    if piece.len() == 1 {
        return vec![ranks[piece]];
    }
    _byte_pair_merge(ranks, piece)
        .windows(2)
        .map(|part| ranks[&piece[part[0].0..part[1].0]])
        .collect()
}
 */

std::vector<rank_t> byte_pair_encode(const view_merge_rank_t &ranks, const std::string &piece) {
    if (piece.size() == 1) {
        return {ranks.at(piece)};
    }
    auto parts = _byte_pair_merge(ranks, piece);
    std::vector<rank_t> result;
    result.reserve(parts.size() - 1);
    for (int i = 0; i < parts.size() - 1; ++i) {
        result.push_back(ranks.at(std::string_view(piece.data() + parts[i].start, parts[i + 1].start - parts[i].start)));
    }

    return result;
}

std::string regex_escape(const std::string &input) {
    std::string result;
    for (char c : input) {
        if (c == '\\' || c == '^' || c == '$' || c == '.' || c == '|' || c == '?' || c == '*' || c == '+' || c == '(' || c == ')' || c == '[' || c == ']' || c == '{' || c == '}') {
            result += '\\';
        }
        result += c;
    }
    return result;
}

CoreBPE::CoreBPE(const merge_rank_t &encoder, const std::unordered_map<std::string, rank_t> &special_tokens_encoder,
                 const std::string &pattern) :
    encoder(encoder),
    special_tokens_encoder(special_tokens_encoder), pattern({pattern}) {
    view_encoder = convert_keys_to_string_views(this->encoder);

    for (const auto &pair : encoder) {
        decoder[pair.second] = pair.first;
    }

    assert(this->encoder.size() == decoder.size());

    for (const auto &pair : special_tokens_encoder) {
        special_tokens_decoder[pair.second] = pair.first;
    }

    assert(this->special_tokens_encoder.size() == special_tokens_decoder.size());

    for (auto &[token, _] : special_tokens_encoder) {
        special_tokens_pattern.push_back(regex_escape(token));
    }
}

std::vector<rank_t> CoreBPE::encode_native(const std::string &text, std::unordered_set<std::string> allowed_special_tokens) {
    std::vector<rank_t> result;
    std::unordered_set<std::string> special_tokens_set;
    std::vector<std::string> special_tokens_pat;
    for (auto &[token, _] : special_tokens_encoder) {
        if (allowed_special_tokens.find(token) != allowed_special_tokens.end()) {
            special_tokens_pat.push_back(regex_escape(token));
            special_tokens_set.insert(token);
        }
    }

    for (auto &fragment : unicode_regex_split_naive(text, special_tokens_pat)) {
        if (special_tokens_set.find(fragment) != special_tokens_set.end()) {
            result.push_back(special_tokens_encoder.at(fragment));
            continue;
        }
        for (auto &piece : unicode_regex_split_naive(fragment, pattern)) {
            auto it = encoder.find(piece);
            if (it == encoder.end()) {
                auto parts = byte_pair_encode(view_encoder, piece);
                result.insert(result.end(), parts.begin(), parts.end());
            } else {
                result.push_back(it->second);
            }
        }
    }

    return result;
}

std::vector<rank_t> CoreBPE::encode_ordinary_naive(const std::string &text) {
    std::vector<rank_t> result;
    for (auto &piece : unicode_regex_split_naive(text, pattern)) {
        auto it = encoder.find(piece);
        if (it == encoder.end()) {
            auto parts = byte_pair_encode(view_encoder, piece);
            result.insert(result.end(), parts.begin(), parts.end());
        } else {
            result.push_back(it->second);
        }
    }
    return result;
}

std::string CoreBPE::decode(const std::vector<rank_t> &tokens) {
    std::string result;
    for (auto token : tokens) {
        result += decode_token(token);
    }
    return result;
}

std::string CoreBPE::decode_token(rank_t token) {
    if (decoder.find(token) != decoder.end()) {
        return decoder.at(token);
    } else if (special_tokens_decoder.find(token) != special_tokens_decoder.end()) {
        return special_tokens_decoder.at(token);
    } else {
        throw std::runtime_error("Token not found: " + std::to_string(token));
    }
}

merge_rank_t load_tiktoken_bpe(const std::string &filename) {
    merge_rank_t ranks;

    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    std::string line;
    //    int i = 0;
    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }

        size_t pos = line.find(' ');
        if (pos == std::string::npos) {
            throw std::runtime_error("Invalid line format: no space found in line: " + line);
        }

        std::string token = line.substr(0, pos);
        rank_t rank;
        try {
            rank = std::stoul(line.substr(pos + 1));
        } catch (const std::invalid_argument &e) {
            throw std::runtime_error("Invalid rank format in line: " + line);
        } catch (const std::out_of_range &e) {
            throw std::runtime_error("Rank out of range in line: " + line);
        }

        auto token_decoded = base64_decode(token);

        ranks[token_decoded] = rank;
        //        if (i ++ < 10)
        //            std::cout << "token: " << token_decoded << "  " << token << " rank: " << rank << std::endl;
    }

    if (file.bad()) {
        throw std::runtime_error("Error while reading file: " + filename);
    }

    return ranks;
}

std::string base64_encode(const std::string &input) {
    static const std::string base64_chars =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789+/";

    std::string encoded;
    int i = 0, j = 0;
    unsigned char char_array_3[3], char_array_4[4];

    for (char c : input) {
        char_array_3[i++] = c;
        if (i == 3) {
            char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
            char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
            char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
            char_array_4[3] = char_array_3[2] & 0x3f;

            for (i = 0; i < 4; i++) {
                encoded += base64_chars[char_array_4[i]];
            }
            i = 0;
        }
    }

    if (i) {
        for (j = i; j < 3; j++) {
            char_array_3[j] = '\0';
        }

        char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
        char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
        char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
        char_array_4[3] = char_array_3[2] & 0x3f;

        for (j = 0; j < i + 1; j++) {
            encoded += base64_chars[char_array_4[j]];
        }

        while (i++ < 3) {
            encoded += '=';
        }
    }

    return encoded;
}

std::string base64_decode(const std::string &input) {
    static const std::string base64_chars =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789+/";

    std::string decoded;
    int i = 0, j = 0;
    unsigned char char_array_4[4], char_array_3[3];

    for (char c : input) {
        if (c == '=') break;

        char_array_4[i++] = c;
        if (i == 4) {
            for (i = 0; i < 4; i++) {
                char_array_4[i] = base64_chars.find(char_array_4[i]);
            }

            char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

            for (i = 0; i < 3; i++) {
                decoded += char_array_3[i];
            }
            i = 0;
        }
    }

    if (i) {
        for (j = i; j < 4; j++) {
            char_array_4[j] = 0;
        }

        for (j = 0; j < 4; j++) {
            char_array_4[j] = base64_chars.find(char_array_4[j]);
        }

        char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
        char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
        char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

        for (j = 0; j < i - 1; j++) {
            decoded += char_array_3[j];
        }
    }

    return decoded;
}

void TiktokenTokenizer::tokenize(const string &text, vector<token_id_t> &tokens, bool bos) {
    std::string text_copy;
    if (bos) {
        text_copy = this->bos_token + text;
    } else {
        text_copy = text;
    }

    auto token_ids = core.encode_native(text_copy, special_tokens);
    tokens = std::move(token_ids);
}
string TiktokenTokenizer::detokenize(const vector<token_id_t> &tokens) {
    return core.decode(tokens);
}

} // namespace mllm