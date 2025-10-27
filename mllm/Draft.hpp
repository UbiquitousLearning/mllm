/**
 * @file Draft.hpp
 * @author Zhiyang Chen (zhiyangchen@stu.pku.edu.cn)
 * @brief 
 * @date 2025-2-24
 *
 *
 */
#pragma once
#ifndef MLLM_DRAFT_HPP
#define MLLM_DRAFT_HPP
#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>
#include <deque>
#include <algorithm>
#include <cassert>

namespace mllm {


class TracePool {
public:
    struct Trace {
        std::vector<unsigned int> trace_tokens;
        Trace(const std::vector<unsigned int> &tokens) : trace_tokens(tokens) {}
    };

    void add_trace(const std::vector<unsigned int> &tokens) {
        if (tokens.empty()) {
            return;
        }
        traces.push_back(Trace(tokens));
    }

    void clear_trace() {
        traces.clear();
    }

    void reset() {
        is_decoding = false;
        draft_length = 0;
        last_accept_cid = 0;
        last_accept_length = 0;
        last_draft_length = 0;
        traces.clear();
        last_accept_position_ids.clear();
        trace_position_ids.clear();
    }

    inline const Trace& get_accepted_trace() {
        return traces[last_accept_cid];
    }
    inline unsigned int get_accepted_length() {
        return last_accept_length;
    }
    inline unsigned int get_draft_length() {
        return draft_length;
    }
    // inline unsigned int get_n_trace() {
    //     return traces.size();
    // }

    unsigned int evalPosterior(const std::vector<std::vector<float>> &logit_scores, const std::vector<unsigned int> &sampled_token_ids) {
        std::vector<unsigned int> accept_lengths;
        int n_candidate = traces.size();
        unsigned int best_candidate_idx = 0;
        unsigned int max_accept_length = 0;
        unsigned int best_next_token_id = sampled_token_ids[0];

        int idx_offset = 0; // draft token被放到input_ids后的偏移量
        for (int tid = 0; tid < n_candidate; tid++) {
            const std::vector<unsigned int> &trace_tokens = traces[tid].trace_tokens;
            unsigned int trace_length = trace_tokens.size();
            unsigned int accept_length = 0;
            for (int i = 0; i < trace_length; i++) {
                int src_idx = i;
                int tgt_idx = (i == 0)? (0) : (idx_offset + i);
                if (trace_tokens[src_idx] == sampled_token_ids[tgt_idx]) {
                    accept_length += 1;
                } else {
                    break;
                }
            }
            if (accept_length > max_accept_length) {
                max_accept_length = accept_length;
                best_candidate_idx = tid;
                best_next_token_id = sampled_token_ids[idx_offset + accept_length];
            }
            idx_offset += trace_length;
            accept_lengths.push_back(accept_length);
        }
        
        this->last_draft_length = this->draft_length;
        this->last_accept_cid = best_candidate_idx;
        this->last_accept_length = max_accept_length;
        this->last_accept_position_ids.clear();
        for (int i = 0; i < max_accept_length; i++) {
            this->last_accept_position_ids.push_back(this->trace_position_ids[best_candidate_idx][i]);
        }
        // std::cout << "Accept length: " << max_accept_length << std::endl;
        return best_next_token_id;
    }
    

    unsigned int generate_draft(std::vector<unsigned int> &input_ids, std::vector<unsigned int> &position_ids,
            std::vector<int> &tree_ancestors, unsigned int cur_seq_length) {
        unsigned int draft_len = 0;
        this->trace_position_ids.clear();
        for (int i = 0; i < traces.size(); i++) {
            unsigned int trace_len = traces[i].trace_tokens.size();
            input_ids.insert(input_ids.end(), traces[i].trace_tokens.begin(), traces[i].trace_tokens.end());
            tree_ancestors.push_back(0); // 每个trace的首节点总是指向start token
            std::vector<unsigned int> pos;
            for (int j = 0; j < trace_len; j++) {
                position_ids.push_back(draft_len + j + cur_seq_length);
                pos.push_back(draft_len + j + cur_seq_length);
                if (j > 0) {
                    tree_ancestors.push_back(draft_len + j);
                }
            }
            this->trace_position_ids.push_back(pos);
            draft_len += trace_len;
        }
        this->draft_length = draft_len;
        return draft_len;
    }

    std::vector<Trace> traces;
    bool is_decoding = false;
    unsigned int draft_length = 0; // draft部分的总长度
    // 记录上一次verify的结果
    unsigned int last_accept_cid = 0;
    unsigned int last_accept_length = 0;
    unsigned int last_draft_length = 0;
    std::vector<unsigned int> last_accept_position_ids;
    std::vector<std::vector<unsigned int>> trace_position_ids;

private:
    // std::vector<std::vector<unsigned int>> candidate_token_ids;
    // std::vector<std::vector<unsigned int>> candidate_position_ids;
    // std::map<unsigned int, std::vector<unsigned int>> cid2pids;
    // std::vector<int> tree_ancestors;

};
    
    
class SuffixAutomaton {
public:
    struct State {
        std::unordered_map<int, int> next;  // 存储字符ID对应的转移状态
        int link = -1;  // 后缀链接
        int length = 0;  // 当前状态的长度
        int min_endpos = 0;  // 当前状态的最小结束位置
        State() = default;
        State(int link, int length, int min_endpos) : link(link), length(length), min_endpos(min_endpos) {}
    };

    SuffixAutomaton() {
        states.push_back(State(-1, 0, 0));  // 重新初始化状态
        input_ids.push_back(-1);
        last = 0;
        max_length = 0;
        cur_index = 0;
        cur_length = 0;
    }

    void reset() {
        states.clear();
        states.push_back(State(-1, 0, 0));
        input_ids.clear();
        input_ids.push_back(-1);
        last = 0;
        max_length = 0;
        cur_index = 0;
        cur_length = 0;
    }

    void add_tokens(const std::vector<unsigned int>& tokens) {
        for (unsigned int token : tokens) {
            transfer_cur_state(token);
            add_state(token);
        }
        input_ids.insert(input_ids.end(), tokens.begin(), tokens.end());
    }

    std::pair<int, int> lookup(int start_token) const {
        int index = cur_index;
        int length = cur_length;
        transfer_state(index, length, start_token);
        return {index, length};
    }

    int gen_draft(std::vector<unsigned int> &seq, int index, int match_length, unsigned int start_token, int minimum_length = 0) {
        int n = std::min(max_predicts, 1 + static_cast<int>(match_length * alpha));
        if (minimum_length > 0 && match_length > 0) {
            n = std::max(minimum_length, n);
        }
        int endpos = states[index].min_endpos;
        seq.clear();
        for (int i = endpos + 1; i < endpos + n; ++i) {
            if (i >= input_ids.size()) break;
            seq.push_back(input_ids[i]);
        }
        return seq.size();
    }

    void print() const {
        for (size_t i = 1; i < states.size(); ++i) {
            std::cout << "State " << i << ": length = " << states[i].length << ", link = " << states[i].link << ", min_endpos = " << states[i].min_endpos << std::endl;
            for (const auto& [ch, next_state] : states[i].next) {
                std::cout << "  " << char('a' + ch) << " -> " << next_state << std::endl;
            }
        }
    }

private:
    std::vector<State> states;
    int last;
    int max_length;
    int cur_index = 0;
    int cur_length = 0;
    int max_predicts = 40;
    float alpha = 4.0f;
    std::vector<int> input_ids;

    unsigned int expand_state(const State &state) {
        unsigned int new_index = states.size();
        states.push_back(state);
        return new_index;
    }

    void add_state(int c) {
        max_length += 1;
        int cur = expand_state(State(-1, max_length, max_length));
        int p = last;
        while (p != -1 && states[p].next.count(c) == 0) {
            states[p].next[c] = cur;
            p = states[p].link;
        }

        if (p == -1) {
            states[cur].link = 0;
        } else {
            int q = states[p].next[c];
            if (states[p].length + 1 == states[q].length) {
                states[cur].link = q;
            } else {
                int clone = states.size();
                states.push_back(states[q]);
                states[clone].length = states[p].length + 1;
                while (p != -1 && states[p].next[c] == q) {
                    states[p].next[c] = clone;
                    p = states[p].link;
                }
                states[q].link = states[cur].link = clone;
            }
        }
        last = cur;
    }

    void transfer_state(int& index, int& length, int token) const {
        while (index != 0 && states[index].next.count(token) == 0) {
            index = states[index].link;
            length = states[index].length;
        }
        if (states[index].next.count(token)) {
            index = states[index].next.at(token);
            length++;
        } else {
            index = length = 0;
        }
    }

    void transfer_cur_state(int token) {
        transfer_state(cur_index, cur_length, token);
    }

};

} // namespace mllm

#endif //! MLLM_DRAFT_HPP