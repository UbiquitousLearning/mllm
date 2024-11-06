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

class QwenQKVmm final : public Module {
    Layer softmax;
    Layer q_rope;
    Layer k_rope;
    Layer k_cache;
    Layer v_cache;
    Layer qk_mm;
    Layer qkv_mm;
    Layer o_quantize;

    int hidden_size;
    int num_heads;
    int head_dim;
    int num_key_value_heads;
    int num_key_value_groups;

public:
    QwenQKVmm() = default;
    QwenQKVmm(const QWenConfig &config, const QWenNameConfig &names, const string &base_name) {
        hidden_size = config.hidden_size;
        num_heads = config.num_attention_heads * config.hidden_size / config.num_attention_heads;

        q_rope = RoPE(config.RoPE_type, config.rope_theta, config.max_position_embeddings, base_name + "q_rope");
        k_rope = RoPE(config.RoPE_type, config.rope_theta, config.max_position_embeddings, base_name + "k_rope");

        k_cache = KVCache(config.num_attention_heads / config.num_key_value_heads, config.cache_limit, base_name + "k_cache", true);
        v_cache = KVCache(config.num_attention_heads / config.num_key_value_heads, config.cache_limit, base_name + "v_cache", true);

        qk_mm = Matmul(false, true, base_name + "qk");
        qkv_mm = Matmul(false, false, base_name + "qkv");

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

        // auto qk = qk_mm(q, k);
        auto qk = Tensor::mm(q, k.transpose(Chl::SEQUENCE, Chl::DIMENSION));
        qk = softmax(qk);
        // auto o = qkv_mm(qk, v);
        auto o = Tensor::mm(qk, v);

        o = o_quantize(o);

        return {o};
    }
};

int main(int argc, char **argv) {
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
    mllm::xnnpack::Log::warn("QwenQKVmm, time={} microseconds", duration.count() / 4);
}
