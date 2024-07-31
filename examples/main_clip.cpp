#include <iostream>
#include <valarray>
#include <csignal>
#include "cmdline.h"
#include "Net.hpp"
#include "Executor.hpp"
#include "express/Express.hpp"
#include "tokenizers/BPE/Bpe.hpp"
#include "processor/ClipPreProcess.hpp"
#include <cmath>
#include <vector>
#include <numeric>

using namespace mllm;

vector<float> softmax(const vector<float>& scores) {
    vector<float> exps;
    float max_val = *max_element(scores.begin(), scores.end());
    for (float score : scores) {
        exps.push_back(exp(score - max_val));
    }
    float sum_exps = accumulate(exps.begin(), exps.end(), 0.0f);
    for (float& exp : exps) {
        exp /= sum_exps;
    }
    return exps;
}
vector<float> postProcessing(shared_ptr<Tensor> result){
    vector<float> scores;
    for (int i = 0; i < result->batch(); ++i) {
        auto value = result->dataAt<float>(i, 0, 0, 0);
        scores.push_back(value);
    }
    auto token_idx =  softmax(scores);
    return token_idx;
}

NetTensor *Attention(NetTensor *x, int embedding_size, int hidden_size, int head_size, string name) {
    auto *q = _Linear( {x}, embedding_size, hidden_size * head_size, true, name + ".q_proj");
    auto *k = _Linear( {x}, embedding_size, hidden_size * head_size, true, name + ".k_proj");
    auto *v = _Linear( {x}, embedding_size, hidden_size * head_size, true, name + ".v_proj");
    q = q->view(-1, head_size, -1, hidden_size);
    k = k->view(-1, head_size, -1, hidden_size);
    v = v->view(-1, head_size, -1, hidden_size);
    auto *qk = _Matmul( {q, k}, false, true, name + ".qk");
    qk = _Scale( {qk}, 1.0F / std::sqrt(hidden_size), 0.0F, false, name + ".scale");
    if(name.find("text_model") != std::string::npos){
        // qk = _Causalmask( {qk}, name + ".mask");
        qk = _Softmax( {qk}, DIMENSION, true, name + ".softmax");
    } else{
        qk = _Softmax( {qk}, DIMENSION, false, name + ".softmax");
    }
    auto *o = _Matmul( {qk, v}, false, false, name + ".qkv");
    o = o->view(-1, 1, -1, hidden_size * head_size);
    o = _Linear( {o}, hidden_size * head_size, embedding_size, true, name + ".out_proj");
    return o;
}
NetTensor *MLP(  NetTensor *i, int hidden_dim, int ffn_hidden_dim, string name) {
    auto *x = _Linear( {i}, hidden_dim, ffn_hidden_dim, true, name + ".fc1");
    x = _QuickGELU( {x}, name + ".act_fn");
    x = _Linear( {x}, ffn_hidden_dim, hidden_dim, true, name + ".fc2");
    return x;
}
NetTensor *VisionEmbedding(Context *c, NetTensor * i, int hidden_size, string name) {
    i = _Convolution2D({i}, 3, 768, {32, 32}, {32, 32}, VALID, false, name +".patch_embedding");
    i = i->transpose(SEQUENCE, DIMENSION);
    i = i->flatten(HEAD, SEQUENCE);
    auto *s = _Parameter(c, {}, 1, 1, 1, 768, name +".class_embedding");
    i = _Cat( {s, i}, SEQUENCE, name +".class_embedding.cat");
    s = _Parameter(c, {}, 1, 50, 1, 1, name +".position_ids");
    i = *_Embedding( {s}, 50, 768, name +".position_embedding") + i;
    return i;
}
NetTensor *TextEmbedding(Context *c, NetTensor * i,  int vocab_size, int hidden_dim, int max_position_embeddings, string name) {
    i = _Embedding( {i}, vocab_size, hidden_dim, name +".token_embedding");
    auto *s = _Parameter(c, {}, 1, max_position_embeddings, 1, 1, name +".position_ids");
    s = s->_clip({}, {}, {0, i->shape(SEQUENCE)}, {});
    i = *_Embedding( {s}, max_position_embeddings, hidden_dim, name +".position_embedding") +i;
    return i;
}
NetTensor *transformer(Context *c, NetTensor * i,  int vocab_size = 49408, int hidden_dim = 512, int ffn_hidden_dim = 2048, int mutil_head_size = 8, string name="text_model") {
    // auto *i = _Input(c);
    i = TextEmbedding(c, i, vocab_size, hidden_dim, 77, name+".embeddings");
    // loop
    for (int layer = 0; layer < 12; ++layer) {
        auto *x = _LayerNorm( {i}, hidden_dim, true, 1e-6, name+".encoder.layers." + std::to_string(layer) + ".layer_norm1");
        i = *Attention( x, hidden_dim, hidden_dim / mutil_head_size, mutil_head_size, name+".encoder.layers." + std::to_string(layer) + ".self_attn")+i;
        x = _LayerNorm( {i}, hidden_dim, true, 1e-6, name+".encoder.layers." + std::to_string(layer) + ".layer_norm2");
        i = *MLP( x, hidden_dim, ffn_hidden_dim, name+".encoder.layers." + std::to_string(layer) + ".mlp") +i;
        //_SubgraphBegin(c);
    }
    // end loop
    i = _LayerNorm( {i}, hidden_dim,true, 1e-6, name + ".final_layer_norm");
    i = i->clip( {}, {}, {-1}, {});
    return i;
}
NetTensor *vit(Context* c, NetTensor * i,  int hidden_dim= 768, int ffn_hidden_dim = 3072, int class_size=1000, int mutil_head_size = 12, string name = "vision_model"){
    // auto *i = _Input(c, {}, "input_ids");
    i = VisionEmbedding(c, i, hidden_dim, name+".embeddings");
    i = _LayerNorm( {i},  hidden_dim,  true,1e-6, name + ".pre_layrnorm");
    for(int layer=0; layer<12; ++layer) {
        auto *x = _LayerNorm( {i},  hidden_dim,  true,1e-6, name + ".encoder.layers."+std::to_string(layer)+".layer_norm1");
        i = *Attention( x, hidden_dim, hidden_dim / mutil_head_size, mutil_head_size, name + ".encoder.layers."+std::to_string(layer)+".self_attn")+i;
        x = _LayerNorm( {i}, hidden_dim, true, 1e-6, name + ".encoder.layers."+std::to_string(layer)+".layer_norm2");
        i = *MLP( x, hidden_dim, ffn_hidden_dim, name + ".encoder.layers."+std::to_string(layer)+ ".mlp") +i;
        _SubgraphBegin(c);
    }
    i = i->clip( {}, {}, {0}, {});
    i = _LayerNorm( {i}, hidden_dim, true,  1e-6, name + ".post_layernorm");
    return i;
}

void CLIP(Context* c) {
    auto *i = _Input(c, {}, "input_ids");
    i = transformer(c, i);
    auto *p = _Input(c, {}, "input_imgs");
    p = vit(c, p);
    i = _Linear( {i}, 512, 512, false, "text_projection");
    i = *i/i->norm(2);
    p = _Linear( {p}, 768, 512, false, "visual_projection");
    p = *p/p->norm(2);
    auto *o = _Matmul( {i, p}, false, true, "matmul");
    o = _Scale( {o}, 100.0, 0.0F, false, "scale");
}
int main(int argc, char **argv) {
    cmdline::parser cmdParser;
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "../vocab/clip_vocab.mllm");
    cmdParser.add<string>("model", 'm', "specify mllm model path", false, "../models/clip-vit-base-patch32-q4_k.mllm");
    cmdParser.add<string>("merges", 'f', "specify mllm tokenizer merges.txt path", false, "../vocab/clip_merges.txt");
    cmdParser.add<int>("thread", 't', "num of threads", false, 4);
    cmdParser.parse_check(argc, argv);

    string vocab_path = cmdParser.get<string>("vocab");
    string model_path = cmdParser.get<string>("model");
    string merges_path = cmdParser.get<string>("merges");
    int thread_num = cmdParser.get<int>("thread");

    std::unique_ptr<Context> c_ptr(new Context());
    auto *c = c_ptr.get();

    CLIP(c);

    BackendConfig bn;
    Net net(bn);
    net.convert(c->sub_param_, BackendType::MLLM_CPU, thread_num);

    ParamLoader param_loader(model_path);
    Executor ex(&param_loader);
    ex.setup(&net);

    auto tokenizer = new BPETokenizer(vocab_path);
    std::unordered_map<string,unsigned> merge_rank;
    auto merge_file = std::ifstream(merges_path);
    std::string line;
    unsigned rank=0;
    while (std::getline(merge_file, line)) {
        if (line.empty()) {
            continue;
        }
        if (line[0]=='#'){
            continue;
        }
        merge_rank[line]=rank;
        rank++;
    }
    tokenizer->setMergeRank(merge_rank);
    tokenizer->setSpecialToken("<|startoftext|>","<|endoftext|>");

    vector<string> in_strs = {"a photo of a cat", "a photo of a dog"};
    auto tokens_ids = vector<vector<token_id_t>>();
    for (auto in_str : in_strs) {
        vector<mllm::token_id_t> tokens_id={};
        tokenizer->tokenize(in_str, tokens_id, true, true, "</w>");
        tokens_ids.push_back(tokens_id);
    }
    shared_ptr<Tensor> input_text = std::make_shared<Tensor>();
    BPETokenizer::tokens2Tensor(&net, tokens_ids, input_text);

    shared_ptr<Tensor> input_img = std::make_shared<Tensor>();
    auto *clip_processor = new ClipPreProcessor(tokenizer);
    clip_processor->PreProcessImages({"../assets/cat.jpg"});
    auto images = clip_processor->pixel_values_[0];
    clip_processor->Img2Tensor(net.backends()[BackendType::MLLM_CPU].get(), input_img, images);
    ex.run(&net, {input_text, input_img});
    auto result = ex.result();
    auto probs = postProcessing(result[0]);
    for (auto prob : probs) {
        std::cout << prob << "  ";
    }
    std::cout << std::endl;
    // ex.perf();

    // free memory
    for (auto *op : c->net_ops) {
        delete op;
    }
    for (auto *tensor : c->net_tensors) {
        delete tensor;
    }
    return 0;
}
