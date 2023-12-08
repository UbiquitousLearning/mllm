#include <vector>
#include <iostream>
#include <valarray>
#include <csignal>
#include "Net.hpp"
#include "Executor.hpp"
#include "NetParameter.hpp"
#include "express/Express.hpp"
#include "tokenizers/BPE/Bpe.hpp"
#include "tokenizers/Unigram/Unigram.hpp"
#include "preprocess/FuyuPreProcess.hpp"

using namespace std;

void token2Tensor(shared_ptr<Tensor> input_tensor, Net &net, vector<token_id_t> tokens) {
    input_tensor->setBackend(net.backends()[BackendType::MLLM_CPU].get());
    input_tensor->reshape(1, 1, static_cast<int>(tokens.size()), 1);
    input_tensor->setDtype(MLLM_TYPE_F32);
    input_tensor->alloc();
    input_tensor->fullData<float>(1);
    for (int idx = 0; idx < tokens.size(); ++idx) {
        input_tensor->setDataAt<float>(0, 0, idx, 0, tokens[idx]);
    }
}

void testFull(shared_ptr<Tensor> input_tensor, Net &net, vector<int> shape) {
    input_tensor->setBackend(net.backends()[BackendType::MLLM_CPU].get());
    input_tensor->reshape(shape[0], shape[1], shape[2], shape[3]);
    input_tensor->setDtype(MLLM_TYPE_F32);
    input_tensor->alloc();
    input_tensor->fullData<float>(1);
}

unsigned int argmax(const std::vector<float>& scores) {
    if(scores.empty()) {
        throw std::invalid_argument("Input vector is empty");
    }
    unsigned int maxIndex = 0;
    float maxValue = scores[0];
    for(size_t i = 1; i < scores.size(); ++i) {
        if(scores[i] > maxValue) {
            maxIndex = i;
            maxValue = scores[i];
        }
    }
    return maxIndex;
}
unsigned int postProcessing(shared_ptr<Tensor> result, shared_ptr<Tensor>& out_result){
    CHECK_EQ(result->batch(), 1);
    CHECK_EQ(result->head(), 1);
    out_result->reshape(1, 1, 1, 1);
    out_result->alloc();
    vector<float> scores;
    for (int i = 0; i < result->dimension(); ++i) {
        auto value = result->dataAt<float>(0, 0, result->sequence()-1, i);
        scores.push_back(value);
    }
    auto token_idx =  argmax(scores);
    out_result->setDataAt<float>(0, 0, 0, 0, token_idx);
    return token_idx;
}

NetTensor *Attention(Context *ctx, NetTensor * x, int embedding_size, int hidden_size, int head_size, string name){
    x =_Linear(ctx, {x}, embedding_size, hidden_size * head_size * 3, true, name + ".query_key_value");
    auto skv = _Split(ctx, {x}, 3, Chl::D_HD, head_size, name + ".split");
    auto *q = skv[0];
    auto *k = skv[1];
    auto *v = skv[2];
    q = _LayerNorm(ctx, {q}, true, name + ".q_layernorm");
    k = _LayerNorm(ctx, {k}, true, name + ".k_layernorm");
    q = _RoPE(ctx, {q}, name + ".q_rope");
    k = _RoPE(ctx, {k}, name + ".k_rope");
    k = _KVCache(ctx, {k}, true, name + ".k_cache");
    v = _KVCache(ctx, {v}, true, name + ".v_cache");
    auto *qk = _Matmul(ctx, {q, k}, false, true, name + ".qk");
    qk = _Scale(ctx, {qk}, 1.0F / std::sqrt(head_size), 0.0F, false, name + ".scale");
    qk = _Causalmask(ctx, {qk}, name + ".mask");
    qk = _Softmax(ctx, {qk}, SEQUENCE, name + ".softmax");
    auto *o = _Matmul(ctx, {qk, v}, false, false, name + ".qkv");
    o = _View(ctx, {o}, {-1, -1, -1, -1}, {BATCH, -1, SEQUENCE, HEAD+DIMENSION}, name + ".qkv_view");
    o = _Linear(ctx, {o}, hidden_size * head_size, embedding_size, true, name + ".dense");
    return o;
}
NetTensor *MLP(Context *ctx, NetTensor * i, int hidden_dim, int ffn_hidden_dim, string name){
    auto *x = _Linear(ctx, {i}, hidden_dim, ffn_hidden_dim, true, name+".dense_h_to_4h");
    x = _ReLUSquaredActivation(ctx, {x});
    x = _Linear(ctx, {x}, ffn_hidden_dim, hidden_dim, true, name+".dense_4h_to_h");
    return x;
}
NetTensor *Persimmon(Context* c, NetTensor * i, int hidden_dim= 4096, int ffn_hidden_dim = 4096*4, int mutil_head_size = 64, string name = "language_model.model"){
    // loop
    for(int layer=0; layer<1; ++layer) {
        auto *x = _LayerNorm(c, {i}, true, name + (string)".layers."+std::to_string(layer)+".input_layernorm");
        x = Attention(c, x, hidden_dim, hidden_dim / mutil_head_size, mutil_head_size, name + (string)".layers."+std::to_string(layer)+".self_attn");
        i = _Add(c, {x, i});
        x = _LayerNorm(c, {i}, true, name + (string)".layers."+std::to_string(layer)+".post_attention_layernorm");
        x = MLP(c, x, hidden_dim, ffn_hidden_dim, name + (string)".layers."+std::to_string(layer) +".mlp");
        i = _Add(c, {x, i});
        _SubgraphBegin(c);
    }
    // end loop
    i = _LayerNorm(c, {i}, true, name + (string)".final_layernorm");
    return i;
}
void Fuyu(Context* c, int vocab_size= 262144, int patch_size = 30, int cnl_size = 3,int hidden_dim= 4096, int ffn_hidden_dim = 4096*4, int mutil_head_size = 32){
    auto *i = _Input(c, {}, "input_ids");
    i = _Embedding(c, {i}, vocab_size, hidden_dim, (string)"language_model.model.embed_tokens");
    auto *p = _Input(c, {}, "image_patches");
    p = _Linear(c, {p}, patch_size*patch_size*cnl_size, hidden_dim, true, "vision_embed_tokens");
    auto *id = _Input(c, {}, "image_patches_indices");
    i = _Gather(c, {i, p, id}, "gather");
    i = Persimmon(c, i, hidden_dim, ffn_hidden_dim, mutil_head_size, "language_model.model");
    i = _Linear(c, {i}, hidden_dim, vocab_size, false, "language_model.lm_head");
}
int main() {
    /*
    int width, height, channel;
    unsigned char *data = stbi_load("test.jpg", &width, &height, &channel, 0);
    if (data == nullptr) {
        cout << "load image failed" << endl;
        return -1;
    }
    cout << "width: " << width << " height: " << height << " channel: " << channel << endl;
    vector<float> data_f32(width * height * channel);
    for (int i = 0; i < width * height * channel; i++) {
        data_f32[i] = data[i] / 255.0;
    }
    stbi_image_free(data);
    */

    auto tokenizer = UnigramTokenizer("vocab_uni.mllm");
    tokenizer.setSpecialToken("|ENDOFTEXT|");
    auto tokens_id = vector<token_id_t>();
    string in_str = "Generate a caption";
    std::string text_ = "";
    for (auto &ch : in_str) {
        if (ch == ' ') {
            text_ += "▁";
        }else {
            text_ += ch;
        }

    }
    std::string new_text = "▁" + std::string(text_);
    tokenizer.tokenize(new_text, tokens_id, true);
//    for (auto id : tokens_id) {
//        std::cout << id << " ";
//    }
//    auto result_ = tokenizer.detokenize(tokens_id);
//    std::cout << result_ << std::endl;
//    return 0;

    // auto *preprocess = new FuyuPreProcess(&tokenizer, 30, 30, 10);
    // preprocess->Process(in_str);


    int vocab_size = 262144;
    int hidden_dim = 4096;
    int ffn_hidden_dim = 4096*4;
    int mutil_head_size = 64;
    int patch_size = 30;

    std::unique_ptr<Context> c_ptr(new Context());
    auto *c = c_ptr.get();
    Fuyu(c, vocab_size, patch_size, 3, hidden_dim, ffn_hidden_dim, mutil_head_size);

    BackendConfig bn;
    Net net(bn);
    net.convert(c->sub_param_);


    ParamLoader param_loader("../models/fuyu-8b-fp32.mllm");
    Executor ex(&param_loader);


    shared_ptr<Tensor> input_seq = std::make_shared<Tensor>();
    token2Tensor(input_seq, net, tokens_id);
    shared_ptr<Tensor> img_patch = std::make_shared<Tensor>();
    testFull(img_patch,net, {1, 1,(int)input_seq->sequence(), patch_size*patch_size*3});
    // testFull(img_patch,net, {0, 0, 0, 0});
    shared_ptr<Tensor> img_patch_id = std::make_shared<Tensor>();
    testFull(img_patch_id, net,{1, 1,(int)input_seq->sequence(), 1});
    // testFull(img_patch,net, {0, 0, 0, 0});

    std::cout << in_str << std::flush;
    ex.execute(&net, {input_seq, img_patch, img_patch_id});
    auto result = ex.result();
    auto token_idx = postProcessing(result[0], input_seq);
    testFull(img_patch,net, {0, 0, 0, 0});
    testFull(img_patch,net, {0, 0, 0, 0});

    auto out_token = tokenizer.detokenize({token_idx});

    // free memory
    for (auto *op : c->net_ops) {
        delete op;
    }
    for (auto *tensor : c->net_tensors) {
        delete tensor;
    }
    return 0;
}