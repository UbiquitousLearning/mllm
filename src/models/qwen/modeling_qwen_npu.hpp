#ifndef MODELING_QWENNPU_HPP
#define MODELING_QWENNPU_HPP

#include "Backend.hpp"
#include "Layer.hpp"
#include "Module.hpp"
#include "Tensor.hpp"
#include "Types.hpp"
#include "configuration_qwen.hpp"

using namespace mllm;

// NPU QKV part
class QwenDecoderNPUPart1 final : public Module {
    int hidden_size;
    int num_heads;
    int head_dim;
    int num_key_value_heads;
    int num_key_value_groups;

    // it is for speed up the QNN linear implemented by conv, TODO: should integrate into QNNLinear
    Layer pre_attn_view;

    Layer q_proj;
    Layer k_proj;
    Layer v_proj;

    Layer q_view;
    Layer k_view;
    Layer v_view;

    Layer q_dequant;
    Layer k_dequant;
    Layer v_dequant;
    Layer v_transpose;

public:
    QwenDecoderNPUPart1() = default;
    QwenDecoderNPUPart1(const QWenConfig &config, const QWenNameConfig &names, int chunk_size, const string &base_name) {
        hidden_size = config.hidden_size;
        num_heads = config.num_attention_heads;
        head_dim = config.hidden_size / num_heads;
        num_key_value_heads = config.num_key_value_heads;
        num_key_value_groups = num_heads / num_key_value_heads;

        pre_attn_view = View(-1, 1, -1, num_heads * head_dim, base_name + "ires_split-00_view_");

        q_proj = Linear(hidden_size, num_heads * head_dim, true, base_name + names._q_proj_name);
        k_proj = Linear(hidden_size, num_key_value_heads * head_dim, true, base_name + names._k_proj_name);
        v_proj = Linear(hidden_size, num_key_value_heads * head_dim, true, base_name + names._v_proj_name);

        q_view = View(-1, num_heads, -1, head_dim, base_name + names._q_proj_name + "-00_view_");
        k_view = View(-1, num_key_value_heads, -1, head_dim, base_name + names._k_proj_name + "-00_view_");
        v_view = View(-1, num_key_value_heads, -1, head_dim, base_name + names._v_proj_name + "-00_view_");

        q_dequant = Dequantize(true, base_name + names._q_proj_name + ".dequantize");
        k_dequant = Dequantize(true, base_name + names._k_proj_name + ".dequantize", false);
        v_dequant = Dequantize(true, base_name + names._v_proj_name + ".dequantize", false);

        v_transpose = Transpose({0, 2, 3, 1}, base_name + names._v_proj_name + ".transpose");
    }

    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto x = pre_attn_view(inputs[0]);

        auto query_states = q_proj(x);
        auto key_states = k_proj(x);
        auto value_states = v_proj(x);

        query_states = q_view(query_states);
        key_states = k_view(key_states);
        value_states = v_view(value_states);

        query_states = q_dequant(query_states);
        key_states = k_dequant(key_states);
        value_states = v_dequant(value_states);

        value_states = v_transpose(value_states);
        return {query_states, key_states, value_states};
    }
};

// CPU QKV MM part
class QwenQKVmm final : public Module {
    RoPE q_rope;
    RoPE k_rope;
    KVCache k_cache;
    KVCache v_cache;
    Softmax softmax;
    Layer o_quantize;

    int hidden_size;
    int num_heads;
    int head_dim;
    int num_key_value_heads;
    int num_key_value_groups;

public:
    QwenQKVmm() = default;
    QwenQKVmm(const QWenConfig &config, const QWenNameConfig &names, int chunk_size, const string &base_name) {
        hidden_size = config.hidden_size;
        num_heads = config.num_attention_heads * config.hidden_size / config.num_attention_heads;

        q_rope = RoPE(config.RoPE_type, config.rope_theta, config.max_position_embeddings, base_name + "q_rope");
        k_rope = RoPE(config.RoPE_type, config.rope_theta, config.max_position_embeddings, base_name + "k_rope");

        k_cache = KVCache(config.num_attention_heads / config.num_key_value_heads, config.cache_limit, base_name + "k_cache", true);
        v_cache = KVCache(config.num_attention_heads / config.num_key_value_heads, config.cache_limit, base_name + "v_cache", true);

        softmax = Softmax(DIMENSION, true, base_name + "softmax");

        o_quantize = Quantize(true, base_name + names._o_proj_name + ".quantize");
    }

    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto q = inputs[0];
        auto k = inputs[1];
        auto v = inputs[2];

        q = q_rope(q);
        k = k_rope(k);

        k = k_cache(k);
        v = v_cache(v);

        auto qk = Tensor::mm(q, k.transpose(Chl::SEQUENCE, Chl::DIMENSION));
        qk = softmax(qk);
        auto o = Tensor::mm(qk, v);

        o = o_quantize(o);

        return {o};
    }
};

// QNN mlp part
class QwenDecoderNPUPart2 final : public Module {
    int hidden_size;
    int num_heads;
    int head_dim;
    int num_key_value_heads;
    int num_key_value_groups;
    int intermediate_size;

    // NPU part2 of attention
    Layer pre_oproj_view;
    Layer out_proj;
    Layer post_oproj_view;
    Layer post_oproj_dequantize;

    // NPU mlp
    Layer pre_mlp_quantize;
    Layer pre_mlp_view;
    Layer gate_proj;
    Layer up_proj;
    Layer post_up_proj_dequantize;
    Layer post_gate_proj_dequantize;
    Layer silu;
    Layer post_attn_layernorm;

    Layer down_proj;
    Layer pre_down_proj_quantize;
    Layer post_down_proj_dequantize;
    Layer post_mlp_view;

    Layer post_atten_res_add;
    Layer post_mlp_res_add;
    Layer mlp_mul;

public:
    QwenDecoderNPUPart2() = default;
    QwenDecoderNPUPart2(const QWenConfig &config, const QWenNameConfig &names, int chunk_size, const string &base_name) {
        hidden_size = config.hidden_size;
        num_heads = config.num_attention_heads;
        head_dim = config.hidden_size / num_heads;
        intermediate_size = config.intermediate_size;
        num_key_value_heads = config.num_key_value_heads;
        num_key_value_groups = num_heads / num_key_value_heads;

        // for QNN linear speed up
        pre_oproj_view = View(1, utils::closestFactors(chunk_size).first, utils::closestFactors(chunk_size).second, head_dim * num_heads, base_name + names._attn_base_name + "or_split-00_view_");
        out_proj = Linear(hidden_size, hidden_size, false, base_name + names._attn_base_name + names._o_proj_name);
        post_oproj_dequantize = Dequantize(true, base_name + names._attn_base_name + names._o_proj_name + ".dequantize");
        post_oproj_view = View(1, 1, chunk_size, hidden_size, base_name + names._attn_base_name + names._o_proj_name + ".dequantize-00_view_");
        post_atten_res_add = Add(base_name + names._attn_base_name + "post_atten_add");

        post_attn_layernorm =
            RMSNorm(config.hidden_size, config.rms_norm_eps, base_name + names._ffn_norm_name);

        auto mlp_base_name = base_name + names._ffn_base_name;
        pre_mlp_quantize = Quantize(true, mlp_base_name + names._up_proj_name + ".quantize");
        pre_mlp_view = View(1, utils::closestFactors(chunk_size).first, utils::closestFactors(chunk_size).second, hidden_size, mlp_base_name + names._up_proj_name + ".quantize-00_view_");
        gate_proj = Linear(hidden_size, intermediate_size, false, mlp_base_name + names._gate_proj_name);
        silu = SiLU(mlp_base_name + "act");
        up_proj = Linear(hidden_size, intermediate_size, false, mlp_base_name + names._up_proj_name);
        post_up_proj_dequantize = Dequantize(true, mlp_base_name + names._up_proj_name + ".dequantize", false);
        post_gate_proj_dequantize = Dequantize(true, mlp_base_name + names._gate_proj_name + ".dequantize", false);

        down_proj = Linear(intermediate_size, hidden_size, false, mlp_base_name + names._down_proj_name);
        pre_down_proj_quantize = Quantize(true, mlp_base_name + names._down_proj_name + ".quantize");
        post_down_proj_dequantize = Dequantize(true, mlp_base_name + names._down_proj_name + ".dequantize");
        post_mlp_view = View(1, 1, chunk_size, hidden_size, mlp_base_name + names._down_proj_name + ".dequantize-00_view_");

        mlp_mul = Mul(mlp_base_name + "mul");
        post_mlp_res_add = Add(mlp_base_name + "res_add");
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto atten_output = inputs[0];
        auto res = inputs[1];

        atten_output = pre_oproj_view(atten_output);
        atten_output = out_proj(atten_output);
        atten_output = post_oproj_dequantize(atten_output);
        atten_output = post_oproj_view(atten_output);

        auto tmp = post_atten_res_add(atten_output, res);

        auto x = post_attn_layernorm(tmp);

        x = pre_mlp_quantize(x);
        // reshape to 32,2
        x = pre_mlp_view(x);

        auto gate_out = gate_proj(x);
        auto up_out = up_proj(x);

        gate_out = post_gate_proj_dequantize(gate_out);
        gate_out = silu(gate_out);

        up_out = post_up_proj_dequantize(up_out);
        gate_out = mlp_mul(gate_out, up_out);

        gate_out = pre_down_proj_quantize(gate_out);
        gate_out = down_proj(gate_out);
        gate_out = post_down_proj_dequantize(gate_out);

        // reshape to 64,1
        gate_out = post_mlp_view(gate_out);

        gate_out = post_mlp_res_add(gate_out, tmp);
        return {gate_out};
    }
};

class QwenDecoderNPUPart2WithShadow final : public Module {
    int hidden_size;
    int num_heads;
    int head_dim;
    int num_key_value_heads;
    int num_key_value_groups;
    int intermediate_size;

    // NPU part2 of attention
    Layer pre_oproj_view;
    Layer out_proj;
    Layer post_oproj_view;
    Layer post_oproj_dequantize;

    // NPU mlp
    Layer pre_mlp_quantize;
    Layer pre_mlp_view;
    Layer gate_proj;
    Layer up_proj;
    Layer post_up_proj_dequantize;
    Layer post_gate_proj_dequantize;
    Layer silu;
    Layer post_attn_layernorm;

    Layer down_proj;
    Layer pre_down_proj_quantize;
    Layer post_down_proj_dequantize;
    Layer post_mlp_view;

    Layer post_atten_res_add;
    Layer post_mlp_res_add;
    Layer mlp_mul;

public:
    QwenDecoderNPUPart2WithShadow() = default;
    QwenDecoderNPUPart2WithShadow(const QWenConfig &config, const QWenNameConfig &names, int chunk_size, const string &base_name) {
        hidden_size = config.hidden_size;
        num_heads = config.num_attention_heads;
        head_dim = config.hidden_size / num_heads;
        intermediate_size = config.intermediate_size;
        num_key_value_heads = config.num_key_value_heads;
        num_key_value_groups = num_heads / num_key_value_heads;

        // for QNN linear speed up
        pre_oproj_view = View(1, utils::closestFactors(chunk_size).first, utils::closestFactors(chunk_size).second, head_dim * num_heads, base_name + names._attn_base_name + "or_split-00_view_");
        out_proj = Linear(hidden_size, hidden_size, false, base_name + names._attn_base_name + names._o_proj_name);
        post_oproj_dequantize = Dequantize(true, base_name + names._attn_base_name + names._o_proj_name + ".dequantize");
        post_oproj_view = View(1, 1, chunk_size, hidden_size, base_name + names._attn_base_name + names._o_proj_name + ".dequantize-00_view_");
        post_atten_res_add = Add(base_name + names._attn_base_name + "post_atten_add");

        post_attn_layernorm =
            RMSNorm(config.hidden_size, config.rms_norm_eps, base_name + names._ffn_norm_name);

        auto mlp_base_name = base_name + names._ffn_base_name;
        pre_mlp_quantize = Quantize(true, mlp_base_name + names._up_proj_name + ".quantize");
        pre_mlp_view = View(1, utils::closestFactors(chunk_size).first, utils::closestFactors(chunk_size).second, hidden_size, mlp_base_name + names._up_proj_name + ".quantize-00_view_");
        gate_proj = Linear(hidden_size, intermediate_size, false, mlp_base_name + names._gate_proj_name);
        silu = SiLU(mlp_base_name + "act");
        up_proj = Linear(hidden_size, intermediate_size, false, mlp_base_name + names._up_proj_name);
        post_up_proj_dequantize = Dequantize(true, mlp_base_name + names._up_proj_name + ".dequantize");
        post_gate_proj_dequantize = Dequantize(true, mlp_base_name + names._gate_proj_name + ".dequantize");

        down_proj = Linear(intermediate_size, hidden_size, false, mlp_base_name + names._down_proj_name);
        pre_down_proj_quantize = Quantize(true, mlp_base_name + names._down_proj_name + ".quantize");
        post_down_proj_dequantize = Dequantize(true, mlp_base_name + names._down_proj_name + ".dequantize");
        post_mlp_view = View(1, 1, chunk_size, hidden_size, mlp_base_name + names._down_proj_name + ".dequantize-00_view_");

        mlp_mul = Mul(mlp_base_name + "mul");
        post_mlp_res_add = Add(mlp_base_name + "res_add");
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto atten_output = inputs[0];
        auto res = inputs[1];

        atten_output = pre_oproj_view(atten_output);
        atten_output = out_proj(atten_output);
        atten_output = post_oproj_dequantize(atten_output);
        atten_output = post_oproj_view(atten_output);

        auto tmp = post_atten_res_add(atten_output, res);

        auto x = post_attn_layernorm(tmp);

        x = pre_mlp_quantize(x);
        // reshape to 32,2
        x = pre_mlp_view(x);

        auto gate_out = gate_proj(x);
        auto up_out = up_proj(x);

        gate_out = post_gate_proj_dequantize(gate_out);
        gate_out = silu(gate_out);

        up_out = post_up_proj_dequantize(up_out);
        gate_out = mlp_mul(gate_out, up_out);

        auto shadow_input_1 = gate_out;

        gate_out = pre_down_proj_quantize(gate_out);
        gate_out = down_proj(gate_out);
        auto shadow_input_2 = gate_out;
        gate_out = post_down_proj_dequantize(gate_out);

        // reshape to 64,1
        gate_out = post_mlp_view(gate_out);

        gate_out = post_mlp_res_add(gate_out, tmp);
        return {shadow_input_1, shadow_input_2, gate_out};
    }
};

class QwenNPU_CPUDecoder final : public Module {
    int hidden_size;
    int num_heads;
    int head_dim;
    int num_key_value_heads;
    int num_key_value_groups;

    Layer input_layernorm;
    Layer pre_attn_quantize;
    QwenDecoderNPUPart1 part1;
    QwenQKVmm qkv_mm;
    QwenDecoderNPUPart2 part2;

public:
    QwenNPU_CPUDecoder() = default;
    QwenNPU_CPUDecoder(const QWenConfig &config, const QWenNameConfig &names, int chunk_size, const string &base_name) {
        hidden_size = config.hidden_size;
        num_heads = config.num_attention_heads;
        head_dim = config.hidden_size / num_heads;
        num_key_value_heads = config.num_key_value_heads;
        num_key_value_groups = num_heads / num_key_value_heads;

        input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, base_name + names._attn_norm_name);
        pre_attn_quantize = Quantize(true, base_name + names._attn_base_name + names._q_proj_name + ".quantize");

        part1 = QwenDecoderNPUPart1(config, names, chunk_size, base_name + names._attn_base_name);
        part1.to(MLLM_QNN);

        qkv_mm = QwenQKVmm(config, names, chunk_size, base_name + names._attn_base_name);
        qkv_mm.to(MLLM_CPU);

        part2 = QwenDecoderNPUPart2(config, names, chunk_size, base_name);
        part2.to(MLLM_QNN);
    }

    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto x = input_layernorm(inputs[0]);
        x = pre_attn_quantize(x);

        x = Tensor::toQNN({x})[0];
        auto q_k_v = part1({x}); // q,k,v
        q_k_v = Tensor::toCPU(q_k_v);

        auto o_x = qkv_mm(q_k_v)[0];

        auto qnn_tensor = Tensor::toQNN({o_x, inputs[0]});
        o_x = qnn_tensor[0];
        inputs[0] = qnn_tensor[1];
        x = part2({o_x, inputs[0]})[0];
        x = Tensor::toCPU({x})[0];

        return {x};
    }
};

class QwenNPU_CPUDecoderWithShadow final : public Module {
    int hidden_size;
    int num_heads;
    int head_dim;
    int num_key_value_heads;
    int num_key_value_groups;

    Layer input_layernorm;
    Layer pre_attn_quantize;
    Layer shadow_linear;
    QwenDecoderNPUPart1 part1;
    QwenQKVmm qkv_mm;
    QwenDecoderNPUPart2WithShadow part2;

public:
    QwenNPU_CPUDecoderWithShadow() = default;
    QwenNPU_CPUDecoderWithShadow(const QWenConfig &config, const QWenNameConfig &names, int chunk_size, const string &base_name) {
        hidden_size = config.hidden_size;
        num_heads = config.num_attention_heads;
        head_dim = config.hidden_size / num_heads;
        num_key_value_heads = config.num_key_value_heads;
        num_key_value_groups = num_heads / num_key_value_heads;

        input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, base_name + names._attn_norm_name);
        pre_attn_quantize = Quantize(true, base_name + names._attn_base_name + names._q_proj_name + ".quantize");

        part1 = QwenDecoderNPUPart1(config, names, chunk_size, base_name + names._attn_base_name);
        part1.to(MLLM_QNN);

        qkv_mm = QwenQKVmm(config, names, chunk_size, base_name + names._attn_base_name);
        qkv_mm.to(MLLM_CPU);

        part2 = QwenDecoderNPUPart2WithShadow(config, names, chunk_size, base_name);
        part2.to(MLLM_QNN);

        shadow_linear = ShadowLinear(config.intermediate_size, hidden_size, 1024, false, base_name + names._ffn_base_name + names._down_proj_name + ".shadow");
    }

    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto x = input_layernorm(inputs[0]);
        x = pre_attn_quantize(x);

        x = Tensor::toQNN({x})[0];
        auto q_k_v = part1({x}); // q,k,v
        q_k_v = Tensor::toCPU(q_k_v);

        auto o_x = qkv_mm(q_k_v)[0];

        auto qnn_tensor = Tensor::toQNN({o_x, inputs[0]});
        o_x = qnn_tensor[0];
        inputs[0] = qnn_tensor[1];
        auto decoder_out = part2({o_x, inputs[0]});
        decoder_out = Tensor::toCPU(decoder_out);

        auto shadow_input_1 = decoder_out[0];
        auto shadow_input_2 = decoder_out[1];
        x = decoder_out[2];
        x = shadow_linear(shadow_input_1, shadow_input_2, x);

        return {x};
    }
};

// Copied from GemmaModel with Gemma->Qwen and set RmsNorm(without add_unit_offset)
class QWenModel_NPU final : public Module {
    template <typename T1, typename SHADOW, typename... Args>
    static vector<unique_ptr<Module>> ListWithShadow(int n, Args &&...args) {
        static_assert(std::is_base_of<Module, T1>::value, "T1 must be a subclass of Module");
        static_assert(std::is_base_of<Module, SHADOW>::value, "SHADOW must be a subclass of Module");
        listIdx = 0;
        vector<unique_ptr<Module>> modules;
        std::set shadowLayers = {1, 2, 26};
        // for index in shadowLayers, create shadow decoder, for others, create normal decoder
        for (int i = 0; i < n; i++) {
            auto new_args = change_last(args...); // 创建新的参数包，最后一个参数被修改为原来的值+ std::to_string(listIdx)+ "."
            if (shadowLayers.find(listIdx) != shadowLayers.end()) {
                modules.push_back(std::make_unique<SHADOW>(std::apply([&](auto &&...args) { return SHADOW(std::forward<decltype(args)>(args)...); }, new_args)));
            } else {
                modules.push_back(std::make_unique<T1>(std::apply([&](auto &&...args) { return T1(std::forward<decltype(args)>(args)...); }, new_args)));
            }
            listIdx++;
        }
        listIdx = 0;
        return modules;
    }

public:
    QWenModel_NPU() = default;
    QWenModel_NPU(const QWenConfig &config, const QWenNameConfig &names, int chunk_size, const string &base_name) {
        // blocks = List<QwenNPU_CPUDecoder>(1, config, names, base_name);
        blocks = ListWithShadow<QwenNPU_CPUDecoder, QwenNPU_CPUDecoderWithShadow>(24, config, names, chunk_size, base_name);
        norm = RMSNorm(config.hidden_size, config.rms_norm_eps, names.post_norm_name);
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto x = inputs[0];
        for (auto &block : blocks) {
            x = (*block)({x})[0];
        }
        x = norm(x);
        return {x};
    }

private:
    std::vector<unique_ptr<Module>> blocks;
    Layer norm;
};

class QWenForCausalLM_NPU final : public Module {
public:
    QWenForCausalLM_NPU(QWenConfig &config, int chunk_size = 64) {
        auto names = config.names_config;
        hidden_size = config.hidden_size;
        tie_embedding_words = config.tie_embedding_words;
        embedding = Embedding(config.vocab_size, config.hidden_size, names.token_embd_name);
        model = QWenModel_NPU(config, names, chunk_size, names.blk_name);

        // Qwen-0.5 use tied embedding
        // Others use nn.Linear()
        if (tie_embedding_words) {
            lm_head = Parameter(1, config.vocab_size, 1, config.hidden_size, names.token_embd_name + ".weight");
        } else {
            lm_head_layer = HeadLinear(config.hidden_size, config.vocab_size, false, names.lm_head_name);
        }
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto x = embedding(inputs[0]);

        // go through model
        auto outputs = model({x})[0];
        if (tie_embedding_words) {
            outputs = Tensor::mm(outputs, lm_head().transpose(Chl::SEQUENCE, Chl::DIMENSION));
        } else {
            outputs = lm_head_layer(outputs);
        }
        return {outputs};
    }

    virtual void generate(
        Tensor &input_ids, const LlmTextGeneratorOpts &opt, const std::function<bool(unsigned int)> &call_back = [](unsigned int) -> bool { return true; }) override {
        auto chatPostProcessing = [](unsigned token_idx, Tensor &tokens_tensor, const vector<Tensor *> &clean_tensors) {
            tokens_tensor.reshape(1, 1, 1, 1);
            tokens_tensor.alloc();
            tokens_tensor.setDataAt<float>(0, 0, 0, 0, token_idx);

            for (auto tensor : clean_tensors) {
                tensor->reshape(0, 0, 0, 0);
                tensor->alloc();
            }
        };

        if (!opt.do_sample) {
            // fail to greedy search
            if (!text_generator_ || text_generator_->type() != LLmTextGeneratorType::kGreedySearch)
                text_generator_ = std::make_shared<LlmTextGenerator>(LLmTextGeneratorType::kGreedySearch, opt);
        } else if (opt.do_sample && !opt.top_k && opt.top_p != 0.f) {
            // fail to top p sampling
            if (!text_generator_ || text_generator_->type() != LLmTextGeneratorType::kToppSampling)
                text_generator_ = std::make_shared<LlmTextGenerator>(LLmTextGeneratorType::kToppSampling, opt);
        } else if (opt.do_sample && opt.top_k) {
            // fail to top k sampling
            if (!text_generator_ || text_generator_->type() != LLmTextGeneratorType::kTopkSampling)
                text_generator_ = std::make_shared<LlmTextGenerator>(LLmTextGeneratorType::kTopkSampling, opt);
        }

        for (int step = 0; step < opt.max_new_tokens; ++step) {
            auto _out = (*this)({input_ids});
            auto out_token = text_generator_->generate(_out[0], opt);
            if (!call_back(out_token)) break;
            chatPostProcessing(out_token, input_ids, {});
            return;
        }
    }

private:
    int hidden_size;
    bool tie_embedding_words;
    Layer embedding;
    Parameter lm_head;
    Layer lm_head_layer;
    QWenModel_NPU model;
};

#endif //! MODELING_QWENNPU_HPP chunk_size,