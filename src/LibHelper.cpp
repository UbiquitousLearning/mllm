//
// Created by 咸的鱼 on 2023/12/16.
//
#ifdef ANDROID_API
#include "LibHelper.hpp"
#include <iostream>
#include <Types.hpp>
#include <utility>
#include <valarray>
#include "Net.hpp"
#include "Executor.hpp"
#include "NetParameter.hpp"
#include "express/Express.hpp"
#include "tokenizers/BPE/Bpe.hpp"

using namespace mllm;
inline bool exists_test (const std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
}
void fullTensor(shared_ptr<Tensor> input_tensor, Net net, vector<int> shape, float value) {
    input_tensor->setBackend(net.backends()[BackendType::MLLM_CPU].get());
    input_tensor->reshape(shape[0], shape[1], shape[2], shape[3]);
    input_tensor->alloc();
    input_tensor->fullData<float>(value);
}

void token2Tensor(shared_ptr<Tensor> input_tensor, Net &net, vector<token_id_t> tokens) {
    input_tensor->setBackend(net.backends()[BackendType::MLLM_CPU].get());
    input_tensor->reshape(1, 1, static_cast<int>(tokens.size()), 1);
    input_tensor->alloc();
    input_tensor->fullData<float>(1);
    for (int idx = 0; idx < tokens.size(); ++idx) {
        input_tensor->setDataAt<float>(0, 0, idx, 0, tokens[idx]);
    }
}

unsigned int argmax(const std::vector<float> &scores) {
    if (scores.empty()) {
        throw std::invalid_argument("Input vector is empty");
    }
    unsigned int maxIndex = 0;
    float maxValue = scores[0];
    for (size_t i = 1; i < scores.size(); ++i) {
        if (scores[i] > maxValue) {
            maxIndex = i;
            maxValue = scores[i];
        }
    }
    return maxIndex;
}

unsigned int postProcessing(shared_ptr<Tensor> result, shared_ptr<Tensor> &out_result) {
    CHECK_EQ(result->batch(), 1);
    CHECK_EQ(result->head(), 1);
    out_result->reshape(1, 1, 1, 1);
    out_result->alloc();
    vector<float> scores;
    for (int i = 0; i < result->dimension(); ++i) {
        auto value = result->dataAt<float>(0, 0, result->sequence() - 1, i);
        scores.push_back(value);
    }
    auto token_idx = argmax(scores);
    out_result->setDataAt<float>(0, 0, 0, 0, token_idx);
    return token_idx;
}

NetTensor *Attention(Context *ctx, NetTensor *x, int embedding_size, int hidden_size, int head_size, string name) {
    auto *q = _Linear(ctx, {x}, embedding_size, hidden_size * head_size, false, name + ".wq");
    auto *k = _Linear(ctx, {x}, embedding_size, hidden_size * head_size, false, name + ".wk");
    auto *v = _Linear(ctx, {x}, embedding_size, hidden_size * head_size, false, name + ".wv");
    q = _View(ctx, {q}, {-1, head_size, -1, -1}, {BATCH, DIMENSION, SEQUENCE, DIMENSION}, name + ".q_view");
    k = _View(ctx, {k}, {-1, head_size, -1, -1}, {BATCH, DIMENSION, SEQUENCE, DIMENSION}, name + ".k_view");
    v = _View(ctx, {v}, {-1, head_size, -1, -1}, {BATCH, DIMENSION, SEQUENCE, DIMENSION}, name + ".v_view");
    q = _RoPE(ctx, {q}, 2, name + ".q_rope");
    k = _RoPE(ctx, {k}, 2, name + ".k_rope");
    k = _KVCache(ctx, {k}, true, name + ".k_cache");
    v = _KVCache(ctx, {v}, true, name + ".v_cache");
    auto *qk = _Matmul(ctx, {q, k}, false, true, name + ".qk");
    qk = _Scale(ctx, {qk}, 1.0F / std::sqrt(hidden_size), 0.0F, false, name + ".scale");
    qk = _Causalmask(ctx, {qk}, name + ".mask");
    qk = _Softmax(ctx, {qk}, DIMENSION, name + ".softmax");
    auto *o = _Matmul(ctx, {qk, v}, false, false, name + ".qkv");
    o = _View(ctx, {o}, {-1, -1, -1, -1}, {BATCH, -1, SEQUENCE, HEAD + DIMENSION}, name + ".qkv_view");
    o = _Linear(ctx, {o}, hidden_size * head_size, embedding_size, false, name + ".wo");
    return o;
}

NetTensor *FFN(Context *ctx, NetTensor *i, int hidden_dim, int ffn_hidden_dim, string name) {
    auto *x = _Linear(ctx, {i}, hidden_dim, ffn_hidden_dim, false, name + ".w1");
    x = _SiLU(ctx, {x});
    auto *y = _Linear(ctx, {i}, hidden_dim, ffn_hidden_dim, false, name + ".w3");
    x = _Mul(ctx, {x, y});
    x = _Linear(ctx, {x}, ffn_hidden_dim, hidden_dim, false, name + ".w2");
    return x;
}

void llama2(Context *c, int vocab_size = 32000, int hidden_dim = 4096, int ffn_hidden_dim = 11008, int mutil_head_size = 32) {
    auto *i = _Input(c);
    i = _Embedding(c, {i}, vocab_size, hidden_dim, (string)"tok_embeddings");
    // loop
    for (int layer = 0; layer < 32; ++layer) {
        auto *x = _RMSNorm(c, {i}, (string)"layers." + std::to_string(layer) + ".attention_norm");
        //x = _Attention(c, {x}, hidden_dim, hidden_dim / mutil_head_size, mutil_head_size, (string)"layers."+std::to_string(layer)+".attention");
        x = Attention(c, x, hidden_dim, hidden_dim / mutil_head_size, mutil_head_size, (string)"layers." + std::to_string(layer) + ".attention");
        i = _Add(c, {x, i});
        x = _RMSNorm(c, {i}, (string)"layers." + std::to_string(layer) + ".ffn_norm");
        x = FFN(c, x, hidden_dim, ffn_hidden_dim, (string)"layers." + std::to_string(layer) + ".feed_forward");
        i = _Add(c, {x, i});
        _SubgraphBegin(c);
    }
    // end loop
    i = _RMSNorm(c, {i}, (string)"norm");
    i = _Linear(c, {i}, hidden_dim, vocab_size, false, "output");
}


bool LibHelper::setUp(const std::string &base_path, std::string weights_path, std::string vacab_path, PreDefinedModel model, MLLMBackendType backend_type) {
    c = new Context();
    BackendConfig bn;
    weights_path = base_path + weights_path;
    vacab_path = base_path + vacab_path;
    LOGI("Setup!");
    //check path exists
    if(!exists_test(weights_path)||!exists_test(vacab_path)){
        return false;
    }

    const auto param_loader = new ParamLoader(std::move(weights_path));
    executor_ = new Executor(param_loader);
    net_ = new Net(bn);
    if (net_ == nullptr || executor_ == nullptr || !param_loader->isAvailible()) {
        return false;
    }
    auto size =param_loader->getParamSize();
    LOGI("param size:%d",size);

    switch (model) {
    case PreDefinedModel::LLAMA: {
        int vocab_size = 32000;
        int hidden_dim = 4096;
        int ffn_hidden_dim = 11008;
        int mutil_head_size = 32;
        llama2(c, vocab_size, hidden_dim, ffn_hidden_dim, mutil_head_size);
        net_->convert(c->sub_param_, BackendType::MLLM_CPU);
        tokenizer_ = new BPETokenizer(vacab_path);
        break;
    }
    default: {
        return false;
    }
    }
     size =tokenizer_->getVocabSize();
    LOGI("tokenizer size:%d",size);
    if (!tokenizer_->isAvailible()) {
        return false;
    }
    return true;
}

void LibHelper::setCallback(callback_t callback) {
    this->callback_ = std::move(callback);
}

void LibHelper::run(const std::string &input_str, unsigned int max_step) const {
    auto tokens_id = vector<token_id_t>();
    tokenizer_->tokenize(input_str, tokens_id, true);
    auto out_string = input_str;
    shared_ptr<Tensor> input = std::make_shared<Tensor>();
    token2Tensor(input, *net_, tokens_id);

    std::cout << input << std::flush;
    for (int step = 0; step < max_step; step++) {
        executor_->execute(net_, {input});
        auto result = executor_->result();
        auto token_idx = postProcessing(result[0], input);
        const auto out_token = tokenizer_->detokenize({token_idx});
        out_string += out_token;
        //TODO: End with EOS
        callback_(out_string, step == max_step - 1);
    }
}

LibHelper::~LibHelper() {
    delete c;
    delete net_;
    delete executor_;
    delete tokenizer_;
}
#endif
