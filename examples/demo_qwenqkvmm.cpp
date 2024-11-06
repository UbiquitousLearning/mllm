#include "cmdline.h"
#include "Backend.hpp"
#include "Layer.hpp"
#include "Module.hpp"
#include "Tensor.hpp"
#include "Types.hpp"
#include "models/qwen/configuration_qwen.hpp"
#include "backends/xnnpack/XnnpackBackend.hpp"
#include "backends/xnnpack/Utils/Logger.hpp"

using namespace mllm;

class QwenQKVmmXnnPart final : public Module {
    Layer sdpa;

public:
    QwenQKVmmXnnPart() = default;

    QwenQKVmmXnnPart(const QWenConfig &config, const QWenNameConfig &names, const string &base_name) {
        sdpa = ScaledDotProductAttention(".sdpa");
    }

    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto q = inputs[0];
        auto k = inputs[1];
        auto v = inputs[2];

        q = q.transpose(SEQUENCE, HEAD);
        k = k.transpose(SEQUENCE, HEAD);
        v = v.transpose(SEQUENCE, HEAD);

        auto o = sdpa(q, k, v);

        return {o};
    }
};

class QuantizeModule final : public Module {
    Layer o_quantize;

public:
    QuantizeModule() = default;
    QuantizeModule(const QWenConfig &config, const QWenNameConfig &names, const string &base_name) {
        o_quantize = Quantize(true, base_name + names._o_proj_name + ".quantize");
    }

    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        return {o_quantize(inputs[0])};
    }
};

class QwenQKVmm final : public Module {
    Layer q_rope;
    Layer k_rope;
    Layer k_cache;
    Layer v_cache;

    int hidden_size;
    int num_heads;

    QwenQKVmmXnnPart xnn_part;
    QuantizeModule o_part;

public:
    QwenQKVmm() = default;
    QwenQKVmm(const QWenConfig &config, const QWenNameConfig &names, const string &base_name) {
        hidden_size = config.hidden_size;
        num_heads = config.num_attention_heads * config.hidden_size / config.num_attention_heads;

        q_rope = RoPE(config.RoPE_type, config.rope_theta, config.max_position_embeddings, base_name + "q_rope");
        k_rope = RoPE(config.RoPE_type, config.rope_theta, config.max_position_embeddings, base_name + "k_rope");

        k_cache = XP_KVCache(config.num_attention_heads / config.num_key_value_heads, config.cache_limit, base_name + "k_cache");
        v_cache = XP_KVCache(config.num_attention_heads / config.num_key_value_heads, config.cache_limit, base_name + "v_cache");

        xnn_part = QwenQKVmmXnnPart(config, names, base_name);
        xnn_part.to(MLLM_XNNPACK);

        o_part = QuantizeModule(config, names, base_name);
        o_part.to(MLLM_CPU);
    }

    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto q = inputs[0];
        auto k = inputs[1];
        auto v = inputs[2];

        q = q_rope(q);
        k = k_rope(k);

        k = k_cache(k);
        v = v_cache(v);

        q.to(MLLM_XNNPACK);
        k.to(MLLM_XNNPACK);
        v.to(MLLM_XNNPACK);
        auto o = xnn_part({q, k, v})[0];
        o.to(MLLM_CPU);

        o = o_part({o})[0];

        return {o};
    }
};

int main(int argc, char **argv) {
    mllm::xnnpack::Log::log_level = mllm::xnnpack::Log::ERROR;

    cmdline::parser cmdParser;
    cmdParser.add<int>("seq-len", 's', "sequence length", true, 64);
    cmdParser.parse_check(argc, argv);

    QWenConfig config(1280, "1.8B", RoPEType::HFHUBROPE);
    auto model = QwenQKVmm(config, config.names_config, "base");
    model.setNoLoadWeightsDtype(MLLM_TYPE_F32);

    Layer::use_layername_2_tensorname = false;
    mllm::xnnpack::XnnpackBackend::enable_dynamic_shape = true;
    mllm::xnnpack::XnnpackBackend::enable_legacy_wrapper = false;

    auto s = cmdParser.get<int>("seq-len");

    Tensor q(1, 1, s, config.hidden_size, Backend::global_backends[MLLM_CPU], true);
    Tensor k(1, 1, s, config.hidden_size, Backend::global_backends[MLLM_CPU], true);
    Tensor v(1, 1, s, config.hidden_size, Backend::global_backends[MLLM_CPU], true);
    q.setTtype(TensorType::INPUT_TENSOR);
    k.setTtype(TensorType::INPUT_TENSOR);
    v.setTtype(TensorType::INPUT_TENSOR);

    // warm up
    auto o = model({q, k, v})[0];

    // start
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 4; ++i) {
        auto o = model({q, k, v})[0];
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    mllm::xnnpack::Log::error("QwenQKVmm, time={} microseconds", duration.count() / 4);
}
