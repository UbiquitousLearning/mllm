/**
 * @file modeling_qwen_sd.hpp
 * @author Zhiyang Chen (zhiyangchen@stu.pku.edu.cn)
 * @brief
 * @date 2025-3-5
 *
 */
#ifndef MODELING_QWENSD_HPP
#define MODELING_QWENSD_HPP

#include "Backend.hpp"
#include "Layer.hpp"
#include "Module.hpp"
#include "Tensor.hpp"
#include "configuration_qwen.hpp"
#include <cmath>
#include "Draft.hpp"

using namespace mllm;

// Copied from modeling_qwen.hpp
class QWenMLP final : public Module {
public:
    QWenMLP() = default;
    QWenMLP(int hidden_size, int intermediate_size, const QWenNameConfig &names,
            const std::string &base_name) {
        gate_proj =
            Linear(hidden_size, intermediate_size, false, base_name + names._gate_proj_name);
        silu = SiLU(base_name + "act");
        up_proj = Linear(hidden_size, intermediate_size, false, base_name + names._up_proj_name);
        down_proj =
            Linear(intermediate_size, hidden_size, false, base_name + names._down_proj_name);
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto x = gate_proj(inputs[0]);
        x = silu(x);
        auto y = up_proj(inputs[0]);
        x = x * y;
        x = down_proj(x);
        return {x};
    }

private:
    Layer gate_proj;
    Layer up_proj;
    Layer down_proj;

    Layer silu;
};

class QWenAttention final : public Module {
public:
    QWenAttention() = default;
    QWenAttention(const QWenConfig &config, const QWenNameConfig &names, const string &base_name) {
        hidden_size = config.hidden_size;
        num_heads = config.num_attention_heads;
        head_dim = config.hidden_size / num_heads;
        num_key_value_heads = config.num_key_value_heads;
        num_key_value_groups = num_heads / num_key_value_heads;

        // init layers
        q_proj = Linear(hidden_size, num_heads * head_dim, true, base_name + names._q_proj_name);
        k_proj = Linear(hidden_size, num_key_value_heads * head_dim, true,
                        base_name + names._k_proj_name);
        v_proj = Linear(hidden_size, num_key_value_heads * head_dim, true,
                        base_name + names._v_proj_name);
        o_proj = Linear(num_heads * head_dim, hidden_size, false, base_name + names._o_proj_name);

        q_rope = RoPETree(config.RoPE_type, config.rope_theta, config.max_position_embeddings,
                          base_name + "q_rope");
        k_rope = RoPETree(config.RoPE_type, config.rope_theta, config.max_position_embeddings,
                          base_name + "k_rope");
        k_cache = KVCache(num_key_value_heads, head_dim, num_key_value_groups, config.cache_limit, base_name + "k_cache");
        v_cache = KVCache(num_key_value_heads, head_dim, num_key_value_groups, config.cache_limit, base_name + "v_cache");
        mask = CausalTreeMask(base_name + "tree_mask");
        softmax = Softmax(DIMENSION, base_name + "softmax");
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto query_states = q_proj(inputs[0]);
        auto key_states = k_proj(inputs[1]);
        auto value_states = v_proj(inputs[2]);
        auto tree_ancestor = inputs[3];

        // [batch, heads, sequence, dims]
        query_states = query_states.view(-1, num_heads, -1, head_dim);
        key_states = key_states.view(-1, num_key_value_heads, -1, head_dim);
        value_states = value_states.view(-1, num_key_value_heads, -1, head_dim);

        // embedding
        query_states = q_rope(query_states, tree_ancestor);
        key_states = k_rope(key_states, tree_ancestor);

        // kv cache
        key_states = k_cache(key_states);
        value_states = v_cache(value_states);

        // attention weight
        auto atten_weight =
            Tensor::mm(query_states, key_states.transpose(Chl::SEQUENCE, Chl::DIMENSION))
            / std::sqrt(head_dim);
        atten_weight = mask(atten_weight, k_cache.getCacheSeqLen(), tree_ancestor);
        atten_weight = softmax(atten_weight, k_cache.getCacheSeqLen());

        // attention output
        auto atten_output = Tensor::mm(atten_weight, value_states);
        atten_output = atten_output.view(-1, 1, -1, head_dim * num_heads);
        atten_output = o_proj(atten_output);
        return {atten_output};
    }

    vector<KVCache *> get_cache() {
        return {&k_cache, &v_cache};
    }
    vector<RoPETree *> get_rope() {
        return {&q_rope, &k_rope};
    }

private:
    int hidden_size;
    int num_heads;
    int head_dim;
    int num_key_value_heads;
    int num_key_value_groups;
    Layer q_proj;
    Layer k_proj;
    Layer v_proj;
    Layer o_proj;
    RoPETree q_rope;
    RoPETree k_rope;
    KVCache k_cache;
    KVCache v_cache;
    // KVCacheTree k_cache;
    // KVCacheTree v_cache;
    CausalTreeMask mask;
    Softmax softmax;
};

class QWenDecoder final : public Module {
public:
    QWenDecoder() = default;
    QWenDecoder(const QWenConfig &config, const QWenNameConfig &names, const string &base_name) {
        self_atten = QWenAttention(config, names, base_name + names._attn_base_name);
        mlp = QWenMLP(config.hidden_size, config.intermediate_size, names,
                      base_name + names._ffn_base_name);
        input_layernorm =
            RMSNorm(config.hidden_size, config.rms_norm_eps, base_name + names._attn_norm_name);
        post_attention_layernorm =
            RMSNorm(config.hidden_size, config.rms_norm_eps, base_name + names._ffn_norm_name);
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto tree_ancestors = inputs[1];
        auto x = input_layernorm(inputs[0]);
        x = self_atten({x, x, x, tree_ancestors})[0];
        auto tmp = x + inputs[0];
        x = post_attention_layernorm(tmp);
        x = mlp({x})[0];
        x = x + tmp;
        return {x};
    }

    QWenAttention &get_attention() {
        return self_atten;
    }

private:
    QWenAttention self_atten;
    QWenMLP mlp;
    Layer input_layernorm;
    Layer post_attention_layernorm;
};

class QWenModel final : public Module {
public:
    QWenModel() = default;
    QWenModel(const QWenConfig &config, const QWenNameConfig &names, const string &base_name) {
        blocks = List<QWenDecoder>(config.num_hidden_layers, config, names, base_name);
        norm = RMSNorm(config.hidden_size, config.rms_norm_eps, names.post_norm_name);
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto x = inputs[0];
        auto &tree_ancestors = inputs[1];
        for (auto &block : blocks) { x = block({x, tree_ancestors})[0]; }
        x = norm(x);
        return {x};
    }

    void clear_kvcache() override {
        for (auto &block : blocks) {
            auto kvcache = block.get_attention().get_cache();
            for (auto &cache : kvcache) { cache->clearCache(); }
            auto ropes = block.get_attention().get_rope();
            for (auto &rope : ropes) { rope->clearCache(); }
        }
    }

    // void update_state(const Pool &candidates) {
    //     unsigned int draft_length = candidates.last_draft_length;
    //     unsigned int withdrawn_length = candidates.last_draft_length - candidates.last_accept_length;

    //     for (auto &block : blocks) {
    //         auto kvcache = block.get_attention().get_cache();
    //         for (auto &cache : kvcache) { cache->updateVerifiedKVCache(candidates.last_accept_position_ids, draft_length); }
    //         auto ropes = block.get_attention().get_rope();
    //         for (auto &rope : ropes) { rope->updateVerifiedRoPECache(withdrawn_length); }
    //     }
    // }

private:
    std::vector<QWenDecoder> blocks;
    Layer norm;
};

class QWenForCausalLM final : public Module {
public:
    QWenForCausalLM(QWenConfig &config) {
        auto names = config.names_config;
        hidden_size = config.hidden_size;
        tie_embedding_words = config.tie_embedding_words;
        embedding = Embedding(config.vocab_size, config.hidden_size, names.token_embd_name);
        model = QWenModel(config, names, names.blk_name);

        // Qwen-0.5 use tied embedding
        // Others use nn.Linear()
        if (tie_embedding_words) {
            lm_head = Parameter(1, config.vocab_size, 1, config.hidden_size,
                                names.token_embd_name + ".weight");
        } else {
            lm_head_layer =
                Linear(config.hidden_size, config.vocab_size, false, names.lm_head_name);
        }

        tp.is_decoding = false;
        // TODO: load datastore
        sa.add_tokens({40, 1079, 264, 3460});
        sa.add_tokens({13, 3017, 829, 374, 1207, 1103});
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        // if (args.size() > 0 && Tensor::tensor_status == TENSOR_STATIC_INIT) { // 只在第一次调Forward的时候更新kv cache
        //     try {
        //         Pool* candidates = std::any_cast<Pool*>(args[0]);
        //         if (candidates->is_decoding) {
        //             model.update_state(*candidates);
        //         }
        //     } catch (const std::bad_any_cast&) {
        //     }
        // }

        auto x = embedding(inputs[0]);
        auto tree_ancestors = inputs[1];

        // go through model
        auto outputs = model({x, tree_ancestors})[0];
        if (tie_embedding_words) {
            outputs = Tensor::mm(outputs, lm_head().transpose(Chl::SEQUENCE, Chl::DIMENSION));
        } else {
            outputs = lm_head_layer(outputs);
        }
        return {outputs};
    }

    void clear_kvcache() override {
        model.clear_kvcache();
    }

    void generate(
        Tensor &input_ids, const LlmTextGeneratorOpts &opt, const std::function<bool(unsigned int)> &call_back)
        override {
        auto post_processing_for_SD = [](const std::vector<unsigned int> &token_indices, const std::vector<int> &tree_anc,
                                         unsigned int input_length, Tensor &input_ids, Tensor &tree_ancestors, const vector<Tensor *> &clean_tensors) {
            input_ids.reshape(1, 1, input_length, 1);
            input_ids.alloc();
            tree_ancestors.reshape(1, 1, input_length, 1);
            tree_ancestors.alloc();
            for (auto seq_id = 0; seq_id < input_length; ++seq_id) {
                input_ids.setDataAt<float>(0, 0, seq_id, 0, token_indices[seq_id]);
                tree_ancestors.setDataAt<int32_t>(0, 0, seq_id, 0, tree_anc[seq_id]);
            }
            for (auto tensor : clean_tensors) {
                tensor->reshape(0, 0, 0, 0);
                tensor->alloc();
            }
        };

        if (!opt.do_sample) {
            // greedy search
            if (!text_generator_ || text_generator_->type() != LLmTextGeneratorType::kGreedySearch) {
                text_generator_ = std::make_shared<LlmTextGenerator>(LLmTextGeneratorType::kGreedySearchForSD, opt);
            }
        } else {
            // TODO nucleus sample for SD
            throw std::runtime_error("Not implemented yet");
        }

        std::vector<unsigned int> seq(input_ids.sequence());
        for (int i = 0; i < input_ids.sequence(); ++i) {
            auto value = input_ids.dataAt<float>(0, 0, i, 0);
            seq[i] = ((unsigned int)(value));
        }
        updateContext(seq);

        tree_ancestors = Tensor(1, 1, 1, 1, input_ids.backend(), true);
        tree_ancestors.setName("tree_ancestors");
        tree_ancestors.setDtype(MLLM_TYPE_I32);
        tp.is_decoding = false;
        static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->setUsingDraft(false); // prefill时不使用
        static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->setLastDraftLength(0);
        static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->setLastVerifiedPositionIds({});

        unsigned int cur_seq_length = input_ids.sequence();
        std::vector<unsigned int> predicted_token_ids;

        for (int step = 0; step < opt.max_new_tokens; ++step) {
            auto _out = (*this)({input_ids, tree_ancestors}, &tp); // batch_size * seq_len * 1 * vocab_size
            auto out_token = text_generator_->generate_SD(_out[0], tp);
            tp.is_decoding = true;

            // 流式输出
            bool is_end_generate = false;
            std::vector<unsigned int> new_predicted_token_ids;
            if (step > 0) {
                const auto &trace = tp.get_accepted_trace();
                auto accept_length = tp.get_accepted_length();
                for (unsigned int i = 0; i < accept_length; i++) {
                    auto accept_token_idx = trace.trace_tokens[i];
                    if (!call_back(accept_token_idx)) {
                        is_end_generate = true;
                        break;
                    }
                    cur_seq_length += 1;
                    new_predicted_token_ids.push_back(accept_token_idx);
                }
            }
            if (is_end_generate || !call_back(out_token)) {
                break;
            }
            // 至少会verifiy一个token
            predicted_token_ids.push_back(out_token);
            new_predicted_token_ids.push_back(out_token);
            cur_seq_length += 1;

            updateContext(new_predicted_token_ids);
            updateTracePool(cur_seq_length, out_token);

            // 为下一次forward准备draft
            std::vector<unsigned int> new_token_ids = {out_token};
            std::vector<unsigned int> position_ids = {cur_seq_length - 1};
            std::vector<int> tree_anc = {-1};
            unsigned int draft_len = tp.generate_draft(new_token_ids, position_ids, tree_anc, cur_seq_length);
            post_processing_for_SD(new_token_ids, tree_anc, draft_len + 1, input_ids, tree_ancestors, {});

            if (step == 0) {
                static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->setUsingDraft(true);
            }
            static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->setLastDraftLength(tp.last_draft_length);
            static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->setLastVerifiedPositionIds(tp.last_accept_position_ids);
        }
        tp.reset();
        sa.reset();
        // std::cout << std::endl;
        // for (int i = 0; i < predicted_token_ids.size(); i++) {
        //     std::cout << predicted_token_ids[i] << ' ';
        // }
    }

private:
    int hidden_size;
    bool tie_embedding_words;
    Layer embedding;
    Parameter lm_head;
    Layer lm_head_layer;
    QWenModel model;

    TracePool tp;
    SuffixAutomaton sa;
    Tensor tree_ancestors;

    void updateTracePool(unsigned int cur_seq_length, unsigned int last_token_id) {
        auto [idx, len] = sa.lookup(last_token_id);
        std::vector<unsigned int> seq;
        auto dlen = sa.gen_draft(seq, idx, len, last_token_id, 10);

        // TODO 保留重用策略
        tp.clear_trace();

        if (dlen > 0) {
            tp.add_trace(seq);
            // std::cout << dlen << std::endl;
        }

        // 测试使用固定的trace
        // if (last_token_id == 40) {
        //     tp.add_trace({88, 3017, 829, 374, 88});
        //     tp.add_trace({1079, 264, 3460, 99});
        // }
    };

    void updateContext(const std::vector<unsigned int> &new_predicted_token_ids) {
        sa.add_tokens(new_predicted_token_ids);
    }
};

#endif //! MODELING_QWENSD_HPP