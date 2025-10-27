#ifndef MODELING_QWEN2VL_NPU_HPP
#define MODELING_QWEN2VL_NPU_HPP

#include "Layer.hpp"
#include "Module.hpp"
#include "Tensor.hpp"
#include "Timing.hpp"
#include "Types.hpp"
#include "configuration_qwen2_vl.hpp"
#include "models/qwen2_vl/modeling_qwen2_vl.hpp"
#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

using namespace mllm;

// NPU QKV part
class QwenDecoderNPUPart1 : public Module {
protected:
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
    QwenDecoderNPUPart1(const Qwen2VLNPUConfig &config, const QWenNameConfig &names, int chunk_size, const string &base_name) {
        hidden_size = config.hidden_size;
        num_heads = config.num_attention_heads;
        head_dim = config.hidden_size / num_heads;
        num_key_value_heads = config.num_key_value_heads;
        num_key_value_groups = num_heads / num_key_value_heads;

        pre_attn_view = View(1, utils::closestFactors(chunk_size).first, utils::closestFactors(chunk_size).second, num_heads * head_dim, base_name + "ires_split-00_view_");

        q_proj = Linear(hidden_size, num_heads * head_dim, config.use_i32_bias, base_name + names._q_proj_name);
        k_proj = Linear(hidden_size, num_key_value_heads * head_dim, config.use_i32_bias, base_name + names._k_proj_name);
        v_proj = Linear(hidden_size, num_key_value_heads * head_dim, config.use_i32_bias, base_name + names._v_proj_name);

        q_view = View(1, num_heads, chunk_size, head_dim, base_name + names._q_proj_name + "-00_view_");
        k_view = View(1, num_key_value_heads, chunk_size, head_dim, base_name + names._k_proj_name + "-00_view_");
        v_view = View(1, num_key_value_heads, chunk_size, head_dim, base_name + names._v_proj_name + "-00_view_");

        if (config.use_i32_bias) {
            q_dequant = Dequantize(true, base_name + names._q_proj_name + ".dequantize", true, MLLM_TYPE_I16);
            k_dequant = Dequantize(true, base_name + names._k_proj_name + ".dequantize", false, MLLM_TYPE_I16);
            v_dequant = Dequantize(true, base_name + names._v_proj_name + ".dequantize", false, MLLM_TYPE_I16);
        } else {
            q_dequant = DequantizeAdd(true, num_heads * head_dim, base_name + names._q_proj_name + ".dequantize", true, MLLM_TYPE_I16);
            k_dequant = DequantizeAdd(true, num_key_value_heads * head_dim, base_name + names._k_proj_name + ".dequantize", false, MLLM_TYPE_I16);
            v_dequant = DequantizeAdd(true, num_key_value_heads * head_dim, base_name + names._v_proj_name + ".dequantize", false, MLLM_TYPE_I16);
        }

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

        // return {query_states, key_states, value_states};

        query_states = q_dequant(query_states);
        key_states = k_dequant(key_states);
        value_states = v_dequant(value_states);

        value_states = v_transpose(value_states);
        return {query_states, key_states, value_states};
    }
};

class QwenDecoderNPUPart1WithRes final : public QwenDecoderNPUPart1 {
    Layer input_layernorm;
    Layer pre_attn_quantize;

public:
    QwenDecoderNPUPart1WithRes() = default;
    QwenDecoderNPUPart1WithRes(const Qwen2VLNPUConfig &config, const QWenNameConfig &names, int chunk_size, const string &base_name) {
        hidden_size = config.hidden_size;
        num_heads = config.num_attention_heads;
        head_dim = config.hidden_size / num_heads;
        num_key_value_heads = config.num_key_value_heads;
        num_key_value_groups = num_heads / num_key_value_heads;

        // remove "self_attn." in base_name
        auto layer_base_name = base_name.substr(0, base_name.size() - 10);
        input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, layer_base_name + names._attn_norm_name);
        pre_attn_quantize = Quantize(true, layer_base_name + names._attn_base_name + names._q_proj_name + ".quantize", MLLM_TYPE_I16);

        pre_attn_view = View(1, utils::closestFactors(chunk_size).first, utils::closestFactors(chunk_size).second, num_heads * head_dim, base_name + "ires_split-00_view_");

        q_proj = Linear(hidden_size, num_heads * head_dim, config.use_i32_bias, base_name + names._q_proj_name);
        k_proj = Linear(hidden_size, num_key_value_heads * head_dim, config.use_i32_bias, base_name + names._k_proj_name);
        v_proj = Linear(hidden_size, num_key_value_heads * head_dim, config.use_i32_bias, base_name + names._v_proj_name);

        q_view = View(1, num_heads, chunk_size, head_dim, base_name + names._q_proj_name + "-00_view_");
        k_view = View(1, num_key_value_heads, chunk_size, head_dim, base_name + names._k_proj_name + "-00_view_");
        v_view = View(1, num_key_value_heads, chunk_size, head_dim, base_name + names._v_proj_name + "-00_view_");

        if (config.use_i32_bias) {
            q_dequant = Dequantize(true, base_name + names._q_proj_name + ".dequantize", true, MLLM_TYPE_I16);
            k_dequant = Dequantize(true, base_name + names._k_proj_name + ".dequantize", false, MLLM_TYPE_I16);
            v_dequant = Dequantize(true, base_name + names._v_proj_name + ".dequantize", false, MLLM_TYPE_I16);
        } else {
            q_dequant = DequantizeAdd(true, num_heads * head_dim, base_name + names._q_proj_name + ".dequantize", true, MLLM_TYPE_I16);
            k_dequant = DequantizeAdd(true, num_key_value_heads * head_dim, base_name + names._k_proj_name + ".dequantize", false, MLLM_TYPE_I16);
            v_dequant = DequantizeAdd(true, num_key_value_heads * head_dim, base_name + names._v_proj_name + ".dequantize", false, MLLM_TYPE_I16);
        }

        v_transpose = Transpose({0, 2, 3, 1}, base_name + names._v_proj_name + ".transpose");
    }

    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto x = input_layernorm(inputs[0]);
        x = pre_attn_quantize(x);

        x = pre_attn_view(x);

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
        return {query_states, key_states, value_states, inputs[0]};
    }
};

// CPU QKV MM part
class QwenQKVmm final : public Module {
    MultimodalRoPE q_rope;
    MultimodalRoPE k_rope;
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
    QwenQKVmm(const Qwen2VLNPUConfig &config, const QWenNameConfig &names, int chunk_size, const string &base_name) {
        hidden_size = config.hidden_size;
        num_heads = config.num_attention_heads;
        head_dim = config.hidden_size / num_heads;

        q_rope = MultimodalRoPE(config.rope_theta, config.max_position_embeddings, config.mrope_section, base_name + "q_rope");
        k_rope = MultimodalRoPE(config.rope_theta, config.max_position_embeddings, config.mrope_section, base_name + "k_rope");

        k_cache = KVCache(config.num_attention_heads / config.num_key_value_heads, config.cache_limit, base_name + "k_cache", true);
        v_cache = KVCache(config.num_attention_heads / config.num_key_value_heads, config.cache_limit, base_name + "v_cache", true);

        softmax = Softmax(DIMENSION, true, base_name + "softmax");

        o_quantize = Quantize(true, base_name + names._o_proj_name + ".quantize");
    }

    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto position_ids = inputs[3];

        auto q = inputs[0];
        auto k = inputs[1];
        auto v = inputs[2];

        q = q_rope(q, position_ids);
        k = k_rope(k, position_ids);

        k = k_cache(k);
        v = v_cache(v);

        auto qk = Tensor::mm(q, k.transpose(Chl::SEQUENCE, Chl::DIMENSION));
        qk = qk / std::sqrt(head_dim);
        qk = softmax(qk);
        auto o = Tensor::mm(qk, v);

        o = o_quantize(o);

        return {o};
    }
};

// QNN mlp part
class QwenDecoderNPUPart2 : public Module {
protected:
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
    QwenDecoderNPUPart2(const Qwen2VLNPUConfig &config, const QWenNameConfig &names, int chunk_size, const string &base_name) {
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

        if (config.use_high_precision_silu) {
            silu = SiLU_Full_Precision(mlp_base_name + "act");
        } else {
            silu = SiLU(mlp_base_name + "act");
        }

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
        auto float_oproj = post_oproj_view(atten_output);

        auto tmp = post_atten_res_add(float_oproj, res);

        auto x = post_attn_layernorm(tmp);

        x = pre_mlp_quantize(x);
        // reshape to 32,2
        x = pre_mlp_view(x);

        auto gate_out = gate_proj(x);
        auto up_out = up_proj(x);

        gate_out = post_gate_proj_dequantize(gate_out);
        auto silu_out = silu(gate_out);

        up_out = post_up_proj_dequantize(up_out);
        gate_out = mlp_mul(silu_out, up_out);

        gate_out = pre_down_proj_quantize(gate_out);
        gate_out = down_proj(gate_out);
        gate_out = post_down_proj_dequantize(gate_out);

        // reshape to 64,1
        auto float_gate_out = post_mlp_view(gate_out);

        gate_out = post_mlp_res_add(float_gate_out, tmp);
        return {gate_out, float_oproj, silu_out, float_gate_out};
    }
};

class QwenDecoderNPUPart2WithShadow final : public QwenDecoderNPUPart2 {
public:
    QwenDecoderNPUPart2WithShadow() = default;
    QwenDecoderNPUPart2WithShadow(const Qwen2VLNPUConfig &config, const QWenNameConfig &names, int chunk_size, const string &base_name) {
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

        if (config.use_high_precision_silu) {
            silu = SiLU_Full_Precision(mlp_base_name + "act");
        } else {
            silu = SiLU(mlp_base_name + "act");
        }

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

    int layer_idx;
    int num_layers;

    SubgraphStart _SubgraphStart_1, _SubgraphStart_2;
    SubgraphFinalize _SubgraphEnd_1, _SubgraphEnd_2;

    Layer input_layernorm;
    Layer pre_attn_quantize;
    unique_ptr<QwenDecoderNPUPart1> part1;
    QwenQKVmm qkv_mm;
    unique_ptr<QwenDecoderNPUPart2> part2;

    std::set<int> shadowLayer;

public:
    QwenNPU_CPUDecoder() = default;
    QwenNPU_CPUDecoder(const Qwen2VLNPUConfig &config, const QWenNameConfig &names, int chunk_size, const string &base_name) :
        shadowLayer(config.shadow_layers) {
        hidden_size = config.hidden_size;
        num_heads = config.num_attention_heads;
        head_dim = config.hidden_size / num_heads;
        num_key_value_heads = config.num_key_value_heads;
        num_key_value_groups = num_heads / num_key_value_heads;

        // extract layer index from base_name like "model.layers.10."
        std::regex re(R"(\d+)");
        std::smatch match;
        std::regex_search(base_name, match, re);
        layer_idx = std::stoi(match[0]);
        num_layers = config.num_hidden_layers;

        if (layer_idx == 0 || shadowLayer.find(layer_idx - 1) != shadowLayer.end()) {
            input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, base_name + names._attn_norm_name);
            pre_attn_quantize = Quantize(true, base_name + names._attn_base_name + names._q_proj_name + ".quantize", MLLM_TYPE_I16);
            part1 = make_unique<QwenDecoderNPUPart1>(config, names, chunk_size, base_name + names._attn_base_name);
        } else {
            part1 = make_unique<QwenDecoderNPUPart1WithRes>(config, names, chunk_size, base_name + names._attn_base_name);
        }

        qkv_mm = QwenQKVmm(config, names, chunk_size, base_name + names._attn_base_name);

        part2 = make_unique<QwenDecoderNPUPart2>(config, names, chunk_size, base_name);

        _SubgraphStart_1 = SubgraphStart(base_name + "subgraph_start1");
        _SubgraphEnd_1 = SubgraphFinalize(base_name + "subgraph_end1");
        _SubgraphStart_2 = SubgraphStart(base_name + "subgraph_start2");
        _SubgraphEnd_2 = SubgraphFinalize(base_name + "subgraph_end2");
    }

    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto position_ids = inputs[1];

        Tensor x, q, k, v, res;
        if (layer_idx == 0 || shadowLayer.find(layer_idx - 1) != shadowLayer.end()) {
            x = input_layernorm(inputs[0]);

            x = pre_attn_quantize(x);

            _SubgraphStart_1({x});

            auto q_k_v = (*part1)({x}); // q,k,v
            q = q_k_v[0];
            k = q_k_v[1];
            v = q_k_v[2];
            res = inputs[0];
            _SubgraphEnd_1(q_k_v);

        } else {
            auto q_k_v_res = (*part1)(inputs); // q,k,v,res
            q = q_k_v_res[0];
            k = q_k_v_res[1];
            v = q_k_v_res[2];
            res = q_k_v_res[3];
            _SubgraphEnd_1(q_k_v_res);
        }

        auto o_x = qkv_mm({q, k, v, position_ids})[0];

        _SubgraphStart_2({o_x, res});

        auto out_part2 = (*part2)({o_x, res});

        if (layer_idx == num_layers - 1) {
            _SubgraphEnd_2(out_part2);
        }

        return out_part2;
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
    unique_ptr<QwenDecoderNPUPart1> part1;
    QwenQKVmm qkv_mm;
    unique_ptr<QwenDecoderNPUPart2WithShadow> part2;

    int layer_idx;
    int num_layers;

    SubgraphStart _SubgraphStart_1, _SubgraphStart_2;
    SubgraphFinalize _SubgraphEnd_1, _SubgraphEnd_2;

    std::set<int> shadowLayer;

public:
    QwenNPU_CPUDecoderWithShadow() = default;
    QwenNPU_CPUDecoderWithShadow(const Qwen2VLNPUConfig &config, const QWenNameConfig &names, int chunk_size, const string &base_name) :
        shadowLayer(config.shadow_layers) {
        hidden_size = config.hidden_size;
        num_heads = config.num_attention_heads;
        head_dim = config.hidden_size / num_heads;
        num_key_value_heads = config.num_key_value_heads;
        num_key_value_groups = num_heads / num_key_value_heads;

        // extract layer index from base_name like "model.layers.10."
        std::regex re(R"(\d+)");
        std::smatch match;
        std::regex_search(base_name, match, re);
        layer_idx = std::stoi(match[0]);
        num_layers = config.num_hidden_layers;

        if (layer_idx == 0 || shadowLayer.find(layer_idx - 1) != shadowLayer.end()) {
            input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, base_name + names._attn_norm_name);
            pre_attn_quantize = Quantize(true, base_name + names._attn_base_name + names._q_proj_name + ".quantize", MLLM_TYPE_I16);
            part1 = make_unique<QwenDecoderNPUPart1>(config, names, chunk_size, base_name + names._attn_base_name);
        } else {
            part1 = make_unique<QwenDecoderNPUPart1WithRes>(config, names, chunk_size, base_name + names._attn_base_name);
        }

        qkv_mm = QwenQKVmm(config, names, chunk_size, base_name + names._attn_base_name);

        part2 = make_unique<QwenDecoderNPUPart2WithShadow>(config, names, chunk_size, base_name);

        shadow_linear = ShadowLinear(config.intermediate_size, hidden_size, 1024, false, base_name + names._ffn_base_name + names._down_proj_name + ".shadow");

        _SubgraphStart_1 = SubgraphStart(base_name + "subgraph_start1");
        _SubgraphEnd_1 = SubgraphFinalize(base_name + "subgraph_end1");
        _SubgraphStart_2 = SubgraphStart(base_name + "subgraph_start2");
        _SubgraphEnd_2 = SubgraphFinalize(base_name + "subgraph_end2");
    }

    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto position_ids = inputs[1];

        Tensor x, q, k, v, res;
        if (layer_idx == 0 || shadowLayer.find(layer_idx - 1) != shadowLayer.end()) {
            x = input_layernorm(inputs[0]);
            x = pre_attn_quantize(x);

            _SubgraphStart_1({x});

            auto q_k_v = (*part1)({x}); // q,k,v
            q = q_k_v[0];
            k = q_k_v[1];
            v = q_k_v[2];
            res = inputs[0];
            _SubgraphEnd_1(q_k_v);
        } else {
            auto q_k_v_res = (*part1)(inputs); // q,k,v,res
            q = q_k_v_res[0];
            k = q_k_v_res[1];
            v = q_k_v_res[2];
            res = q_k_v_res[3];
            _SubgraphEnd_1(q_k_v_res);
        }

        auto o_x = qkv_mm({q, k, v, position_ids})[0];

        _SubgraphStart_2({o_x, res});

        auto decoder_out = (*part2)({o_x, res});
        decoder_out = Tensor::toCPU(decoder_out);

        _SubgraphEnd_2(decoder_out);

        auto shadow_input_1 = decoder_out[0];
        auto shadow_input_2 = decoder_out[1];
        x = decoder_out[2];

        x = shadow_linear(shadow_input_1, shadow_input_2, x);

        return {x};
    }
};

class Qwen2VL_ImagePatchAndEmbedding final : public Module {
    Qwen2VisionModel visual;
    Layer embed_tokens;

    Layer norm;
    Parameter lm_head;
    Layer lm_head_layer;

    bool tie_embedding_words;

    int64_t spatial_merge_size;
    int64_t image_token_id;
    int64_t video_token_id;
    int64_t vision_start_token_id;

public:
    explicit Qwen2VL_ImagePatchAndEmbedding(const Qwen2VLNPUConfig &config) {
        auto vocab_size = config.vocab_size;
        auto hidden_dim = config.hidden_size;
        auto head_size = config.num_attention_heads;
        auto ffn_hidden = config.intermediate_size;
        auto projection_cls = config.projection_cls;
        auto vision_embed_dim = config.vision_embed_dim;
        image_token_id = config.image_token_id;
        auto vision_names = config.vision_names_config;
        auto qwen_names = config.names_config;
        tie_embedding_words = config.tie_embedding_words;
        spatial_merge_size = config.spatial_merge_size;
        image_token_id = config.image_token_id;
        video_token_id = config.video_token_id;
        vision_start_token_id = config.vision_start_token_id;

        embed_tokens = Embedding(vocab_size, hidden_dim, qwen_names.token_embd_name);
        // visual = Qwen2VisionModel(hidden_dim, vision_embed_dim, 16, vision_embed_dim * 4, "QuickGELU", 14, 336, 32, spatial_merge_size, vision_names, vision_names.vison_model_name);
        visual = Qwen2VisionModel(hidden_dim, vision_embed_dim, 16, vision_embed_dim * 4, "QuickGELU", 14, 336, 32, spatial_merge_size, config.attn_implementation, vision_names, vision_names.vison_model_name);
    }

    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto hidden_states = embed_tokens({inputs[0]});

        auto image_embeds = visual({inputs[1], inputs[2]})[0];
        auto n_image_features = image_embeds.sequence();
        auto where_idx = inputs[0].where(image_token_id, SEQUENCE);
        hidden_states = hidden_states.index_put(image_embeds, where_idx, false);

        return {hidden_states};
    }

    // changed from get_position_ids in CPU Qwen2VL, enable padding
    // when prefilling, padding_to should be the max length of the input
    // when decoding, real_seq should be the real length of the input, thus get the correct position_ids for decoding
    void get_position_ids(vector<Tensor> &inputs, int padding_to = 0, int real_seq = 0) {
        if (inputs[0].sequence() > 1) {
            Tensor video_grid_thw(0, 0, 0, 0, MLLM_CPU, true);
            auto rope_indices = get_rope_index_cpp(inputs[0], inputs[2], video_grid_thw, padding_to);
            auto position = rope_indices[0];
            if (inputs.size() == 4) {
                inputs[3] = position;
            } else {
                inputs.push_back(position);
            }
        } else {
            auto &position_ids = inputs[3];
            auto last_pos = real_seq == 0 ? position_ids.dataAt<float>(0, 0, 0, position_ids.dimension() - 1) : real_seq - 1;
            position_ids.reshape(position_ids.batch(), 1, position_ids.sequence(), 1);
            for (int b = 0; b < position_ids.batch(); b++) {
                for (int s = 0; s < position_ids.sequence(); s++) {
                    position_ids.setDataAt<float>(b, 0, s, 0, last_pos + 1);
                }
            }
        }
    }

private:
    vector<Tensor> get_rope_index_cpp(
        Tensor input_ids,
        Tensor image_grid_thw,
        Tensor video_grid_thw,
        int padding_to = 0) {
        vector<vector<int64_t>> attention_mask;
        auto attention_mask_shape = input_ids.sequence();
        for (int b = 0; b < input_ids.batch(); b++) {
            attention_mask.emplace_back(attention_mask_shape, 1);
        }
        const size_t batch_size = input_ids.batch(); // input_ids.size();

        // NOTE: changed from original
        const size_t seq_len = batch_size > 0 ? (padding_to > input_ids.sequence() ? padding_to : input_ids.sequence()) : 0; // batch_size > 0 ? input_ids[0].size() : 0;

        // Tensor position_ids(3, 1, batch_size, seq_len, Backend::global_backends[MLLM_CPU].get()), true);
        // Tensor mrope_position_deltas(1, 1, 1, batch_size, Backend::global_backends[MLLM_CPU].get()), true);
        Tensor position_ids(3, 1, batch_size, seq_len, Backend::global_backends[MLLM_CPU].get(), true);
        Tensor mrope_position_deltas(1, 1, 1, batch_size, Backend::global_backends[MLLM_CPU].get(), true);
        bool has_vision = (image_grid_thw.sequence() > 0) || (video_grid_thw.sequence() > 0); // image_grid_thw || video_grid_thw;
        if (!has_vision) {
            // Pure text case
            for (size_t i = 0; i < batch_size; ++i) {
                const auto &mask = !attention_mask.empty() ? attention_mask[i] : vector<int64_t>(seq_len, 1);
                vector<int64_t> positions;
                int64_t pos = 0;
                for (size_t j = 0; j < seq_len; ++j) {
                    if (mask[j] == 1) {
                        positions.push_back(pos++);
                    } else {
                        positions.push_back(1); // Will be overwritten by mask
                    }
                }
                for (int dim = 0; dim < 3; ++dim) {
                    for (size_t j = 0; j < seq_len; ++j) {
                        position_ids.setDataAt<float>(dim, 0, i, j, (float)(mask[j] == 1 ? positions[j] : 1));
                    }
                }
                int64_t max_pos = pos - 1;
                mrope_position_deltas.setDataAt<float>(0, 0, 0, i, (float)((max_pos + 1) - static_cast<int64_t>(input_ids.sequence())));
            }
            position_ids.setName("position_ids");
            mrope_position_deltas.setName("mrope_position_deltas");
            return {position_ids, mrope_position_deltas};
        }
        // Process vision cases
        size_t image_idx = 0, video_idx = 0;
        for (size_t i = 0; i < batch_size; ++i) {
            const auto &mask = !attention_mask.empty() ? attention_mask[i] : vector<int64_t>(seq_len, 1);
            // Extract valid tokens
            vector<int64_t> valid_tokens;
            for (size_t j = 0; j < input_ids.sequence(); ++j) {
                if (mask[j] == 1) valid_tokens.push_back((int)input_ids.dataAt<float>(i, 0, j, 0));
            }
            // Find vision start positions
            vector<size_t> vision_starts;
            vector<int64_t> vision_types;
            for (size_t j = 0; j < valid_tokens.size(); ++j) {
                if (valid_tokens[j] == vision_start_token_id && j + 1 < valid_tokens.size()) {
                    vision_starts.push_back(j);
                    vision_types.push_back(valid_tokens[j + 1]);
                }
            }
            int64_t image_count = count(vision_types.begin(), vision_types.end(), image_token_id);
            int64_t video_count = vision_types.size() - image_count;
            vector<vector<int64_t>> llm_positions(3);
            size_t st = 0;
            int64_t current_max = 0;
            int64_t remain_images = image_count;
            int64_t remain_videos = video_count;
            // Process each vision segment
            for (size_t vs = 0; vs < vision_starts.size(); ++vs) {
                // Find next vision token
                size_t ed_image = valid_tokens.size();
                size_t ed_video = valid_tokens.size();
                if (remain_images > 0) {
                    auto it = find(valid_tokens.begin() + st, valid_tokens.end(), image_token_id);
                    if (it != valid_tokens.end()) ed_image = it - valid_tokens.begin();
                }
                if (remain_videos > 0) {
                    auto it = find(valid_tokens.begin() + st, valid_tokens.end(), video_token_id);
                    if (it != valid_tokens.end()) ed_video = it - valid_tokens.begin();
                }
                size_t ed = min(ed_image, ed_video);
                if (ed == valid_tokens.size()) break;
                // Get grid parameters
                int64_t t, h, w;
                bool is_image = (ed == ed_image);
                if (is_image) {
                    t = (int64_t)image_grid_thw.dataAt<float>(0, 0, image_idx, 0);
                    h = (int64_t)image_grid_thw.dataAt<float>(0, 0, image_idx, 1);
                    w = (int64_t)image_grid_thw.dataAt<float>(0, 0, image_idx, 2);
                    image_idx++;
                    remain_images--;
                } else {
                    t = (int64_t)video_grid_thw.dataAt<float>(0, 0, video_idx, 0);
                    h = (int64_t)video_grid_thw.dataAt<float>(0, 0, video_idx, 1);
                    w = (int64_t)video_grid_thw.dataAt<float>(0, 0, video_idx, 2);
                    video_idx++;
                    remain_videos--;
                }
                // Calculate grid dimensions
                int64_t llm_grid_t = t;
                int64_t llm_grid_h = h / spatial_merge_size;
                int64_t llm_grid_w = w / spatial_merge_size;
                // Process text segment
                size_t text_len = ed - st;
                if (text_len > 0) {
                    int64_t start_idx = current_max;
                    for (int64_t k = 0; k < text_len; ++k) {
                        for (int dim = 0; dim < 3; ++dim) {
                            llm_positions[dim].push_back(start_idx + k);
                        }
                    }
                    current_max += text_len;
                }
                for (int64_t ti = 0; ti < llm_grid_t; ++ti) {
                    for (int64_t hi = 0; hi < llm_grid_h; ++hi) {
                        for (int64_t wi = 0; wi < llm_grid_w; ++wi) {
                            llm_positions[0].push_back(current_max + ti);
                            llm_positions[1].push_back(current_max + hi);
                            llm_positions[2].push_back(current_max + wi);
                        }
                    }
                }
                current_max = std::max({llm_positions[0][llm_positions[0].size() - 1],
                                        llm_positions[1][llm_positions[1].size() - 1],
                                        llm_positions[2][llm_positions[2].size() - 1]});
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w;
            }
            // Process remaining text
            if (st < valid_tokens.size()) {
                size_t text_len = valid_tokens.size() - st;
                int64_t st_idx = current_max + 1;
                for (int64_t k = 0; k < text_len; ++k) {
                    for (int dim = 0; dim < 3; ++dim) {
                        llm_positions[dim].push_back(st_idx + k);
                    }
                }
                current_max += text_len;
            }
            // Fill position_ids with valid positions
            size_t valid_idx = 0;
            for (size_t j = 0; j < seq_len; ++j) {
                if (mask[j] == 1) {
                    if (valid_idx < llm_positions[0].size()) {
                        position_ids.setDataAt<float>(0, 0, i, j, (float)llm_positions[0][valid_idx]);
                        position_ids.setDataAt<float>(1, 0, i, j, (float)llm_positions[1][valid_idx]);
                        position_ids.setDataAt<float>(2, 0, i, j, (float)llm_positions[2][valid_idx]);
                        valid_idx++;
                    }
                }
            }
            // Calculate delta
            int64_t max_pos = 0;
            for (const auto &dim : llm_positions) {
                for (auto val : dim) {
                    max_pos = max(max_pos, val);
                }
            }
            mrope_position_deltas.setDataAt<float>(0, 0, 0, i, (float)((max_pos + 1) - static_cast<int64_t>(input_ids.sequence())));
        }
        position_ids.setName("position_ids");
        mrope_position_deltas.setName("mrope_position_deltas");
        return {position_ids, mrope_position_deltas};
    }
};

class Qwen2VL_PrefillBody final : public Module {
    std::vector<unique_ptr<Module>> blocks;
    Layer norm;
    Parameter lm_head;
    Layer lm_head_layer;
    int num_layer;

    bool tie_embedding_words;

    template <typename T1, typename SHADOW, typename... Args>
    static vector<unique_ptr<Module>> ListWithShadow(int n, std::set<int> &shadowLayer, Args &&...args) {
        static_assert(std::is_base_of<Module, T1>::value, "T1 must be a subclass of Module");
        static_assert(std::is_base_of<Module, SHADOW>::value, "SHADOW must be a subclass of Module");
        listIdx = 0;
        vector<unique_ptr<Module>> modules;

        // for index in shadowLayers, create shadow decoder, for others, create normal decoder
        for (int i = 0; i < n; i++) {
            auto new_args = change_last(args...); // 创建新的参数包，最后一个参数被修改为原来的值+ std::to_string(listIdx)+ "."
            if (shadowLayer.find(listIdx) != shadowLayer.end()) {
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
    explicit Qwen2VL_PrefillBody(const Qwen2VLNPUConfig &config, int chunk_size, std::set<int> &shadowLayer) {
        auto vocab_size = config.vocab_size;
        auto hidden_dim = config.hidden_size;
        auto head_size = config.num_attention_heads;
        auto qwen_names = config.names_config;
        tie_embedding_words = config.tie_embedding_words;

        num_layer = config.num_hidden_layers;

        blocks = ListWithShadow<QwenNPU_CPUDecoder, QwenNPU_CPUDecoderWithShadow>(config.num_hidden_layers, shadowLayer, config, qwen_names, chunk_size, qwen_names.blk_name);
        norm = RMSNorm(hidden_dim, 1e-6, qwen_names.post_norm_name);
        if (tie_embedding_words) {
            lm_head = Parameter(1, config.vocab_size, 1, config.hidden_size, qwen_names.token_embd_name + ".weight");
        } else {
            lm_head_layer = HeadLinear(config.hidden_size, config.vocab_size, false, qwen_names.token_embd_name);
        }
    }

    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto hidden_states = inputs[0];
        auto position_ids = inputs[1];

        for (auto i = 0; i < blocks.size(); ++i) {
            hidden_states = (*blocks[i])({hidden_states, position_ids})[0];
        }

        hidden_states = norm(hidden_states);

        if (tie_embedding_words) {
            hidden_states = Tensor::mm(hidden_states, lm_head().transpose(Chl::SEQUENCE, Chl::DIMENSION));
        } else {
            hidden_states = lm_head_layer(hidden_states);
        }

        return {hidden_states};
    }
};

// CPU decoding model with only the LLM backbone
class Qwen2VL_Decoding_Model final : public Module {
    Layer embed_tokens;

    vector<QWen2Decoder> blocks;
    Layer norm;
    Parameter lm_head;
    Layer lm_head_layer;

    bool tie_embedding_words;

    int64_t spatial_merge_size;
    int64_t image_token_id;
    int64_t video_token_id;
    int64_t vision_start_token_id;

public:
    explicit Qwen2VL_Decoding_Model(const Qwen2VLConfig &config) {
        auto vocab_size = config.vocab_size;
        auto hidden_dim = config.hidden_size;
        auto head_size = config.num_attention_heads;
        auto ffn_hidden = config.intermediate_size;
        auto projection_cls = config.projection_cls;
        auto vision_embed_dim = config.vision_embed_dim;
        image_token_id = config.image_token_id;
        auto vision_names = config.vision_names_config;
        auto qwen_names = config.names_config;
        tie_embedding_words = config.tie_embedding_words;
        spatial_merge_size = config.spatial_merge_size;
        image_token_id = config.image_token_id;
        video_token_id = config.video_token_id;
        vision_start_token_id = config.vision_start_token_id;

        embed_tokens = Embedding(vocab_size, hidden_dim, qwen_names.token_embd_name);

        blocks = List<QWen2Decoder>(config.num_hidden_layers, config, qwen_names, qwen_names.blk_name);
        norm = RMSNorm(hidden_dim, 1e-6, qwen_names.post_norm_name);
        if (tie_embedding_words) {
            lm_head = Parameter(1, config.vocab_size, 1, config.hidden_size, qwen_names.token_embd_name + ".weight");
        } else {
            lm_head_layer = Linear(config.hidden_size, config.vocab_size, false, qwen_names.lm_head_name);
        }
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto position_ids = inputs[3];

        auto hidden_states = embed_tokens({inputs[0]});

        for (auto &block : blocks) {
            hidden_states = block({hidden_states, position_ids})[0];
        }
        hidden_states = norm(hidden_states);
        if (tie_embedding_words) {
            hidden_states = Tensor::mm(hidden_states, lm_head().transpose(Chl::SEQUENCE, Chl::DIMENSION));
        } else {
            hidden_states = lm_head_layer(hidden_states);
        }
        return {hidden_states};
    }
    void clear_kvcache() override {
        for (auto &block : blocks) {
            auto kvcahce = block.get_attention().get_cache();
            for (auto &cache : kvcahce) {
                cache->clearCache();
            }
        }
    }
};

#endif // MODELING_QWEN2VL_NPU_HPP