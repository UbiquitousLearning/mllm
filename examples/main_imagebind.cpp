#include <iostream>
#include <valarray>
#include <csignal>
#include "cmdline.h"
#include "Net.hpp"
#include "Executor.hpp"
#include "express/Express.hpp"
#include "tokenizers/BPE/Bpe.hpp"
// #ifndef  STB_IMAGE_IMPLEMENTATION
// #define STB_IMAGE_STATIC
// #define STB_IMAGE_IMPLEMENTATION
// #endif
// #include "imageHelper/stb_image.h"
// #include "imageHelper/stb_image_resize2.h"
#include "processor/ClipPreProcess.hpp"
// #include "processor/PreProcess.hpp"
#include <cmath>
#include <vector>
#include <numeric>

using namespace mllm;

std::string toLowercase(const std::string& input) {
    std::string output = input;
    std::transform(output.begin(), output.end(), output.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    return output;
}

void tokens2Tensor(Net *net, vector<vector<token_id_t>> tokens, shared_ptr<Tensor> input_tensor, shared_ptr<Tensor> input_text_lens) {
    input_tensor->setBackend(net->backends()[BackendType::MLLM_CPU].get());
    const auto bsize = static_cast<int>(tokens.size());
    input_tensor->reshape(bsize, 1, 77, 1);
    input_tensor->alloc();

    input_text_lens->setBackend(net->backends()[BackendType::MLLM_CPU].get());
    input_text_lens->reshape(1, 1, 1, bsize);
    input_text_lens->alloc();

    for (int b = 0; b < bsize; ++b){
        input_text_lens->setDataAt<float>(0, 0, 0, b, tokens[b].size()-1);
        for (int idx = 0; idx < tokens[b].size(); ++idx) {
            input_tensor->setDataAt<float>(b, 0, idx, 0, tokens[b][idx]);
        }
    }
}

/*
void img2Tensor(shared_ptr<Tensor> input_tensor, Net &net, vector<float*> imgs, int height, int width, int channel) {
    input_tensor->setBackend(net.backends()[BackendType::MLLM_CPU].get());
    input_tensor->reshape(imgs.size(), channel, 2, height, width);
    input_tensor->setDtype(MLLM_TYPE_F32);
    input_tensor->alloc();
    for (int bi = 0; bi < imgs.size(); ++bi) {
        for (int t = 0; t < 2; ++t) {
            for (int h = 0; h < height; ++h) {
                for (int c = 0; c < channel; ++c) {
                    for (int w = 0; w < width; ++w) {
                        input_tensor->setDataAt<float>(bi, c, t, h, w, imgs[bi][(h * width + w) * channel + c]);
                    }
                }
            }
        }
    }
}
*/
void img2Tensor(shared_ptr<Tensor> input_tensor, Net &net, vector<vector<vector<vector<float>>>> imgs) {
    int channel = imgs[0].size();
    int height = imgs[0][0].size();
    int width= imgs[0][0][0].size();
    input_tensor->setBackend(net.backends()[BackendType::MLLM_CPU].get());
    input_tensor->reshape(imgs.size(), channel, 2, height, width);
    input_tensor->setDtype(MLLM_TYPE_F32);
    input_tensor->alloc();
    for (int bi = 0; bi < imgs.size(); ++bi) {
        for (int t = 0; t < 2; ++t) {
            for (int h = 0; h < height; ++h) {
                for (int c = 0; c < channel; ++c) {
                    for (int w = 0; w < width; ++w) {
                        input_tensor->setDataAt<float>(bi, c, t, h, w, imgs[bi][c][h][w]);
                    }
                }
            }
        }
    }
}
/*
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
*/

NetTensor *Attention(NetTensor *x, int embedding_size, int hidden_size, int head_size, string name) {
    x =_Linear({x}, embedding_size, hidden_size * head_size * 3, true, name + ".in_proj");
    auto skv = _Split( {x}, 3, Chl::HD, head_size, name + ".split");
    auto *q = skv[0];
    auto *k = skv[1];
    auto *v = skv[2];
    auto *qk = _Matmul( {q, k}, false, true, name + ".qk");
    qk = _Scale( {qk}, 1.0F / std::sqrt(hidden_size), 0.0F, false, name + ".scale");
    if(name.find("text") != std::string::npos){
        qk = _Causalmask( {qk}, name + ".mask");
    }
    qk = _Softmax( {qk}, DIMENSION, name + ".softmax");
    auto *o = _Matmul( {qk, v}, false, false, name + ".qkv");
    o = o->view(-1, 1, -1, hidden_size * head_size);
    o = _Linear( {o}, hidden_size * head_size, embedding_size, true, name + ".out_proj");
    return o;
}
NetTensor *MLP( NetTensor *i, int hidden_dim, int ffn_hidden_dim, string name) {
    auto *x = _Linear( {i}, hidden_dim, ffn_hidden_dim, true, name + ".fc1");
    x = _GELU( {x}, name + ".act_fn");
    x = _Linear( {x}, ffn_hidden_dim, hidden_dim, true, name + ".fc2");
    return x;
}
NetTensor *VisionEmbedding(Context *c, NetTensor * i, int hidden_size, string name) { //TODO
    i = _Convolution3D({i}, 3, 1280, {2, 14, 14}, {2, 14, 14}, VALID, false, name +".rgbt_stem.proj.1");
    i = i->transpose(SEQUENCE, DIMENSION);
    i = i->flatten(TIME, CHANNLE);
    auto *s = _Parameter(c, {}, 1, 1, 1, 1280, name +".cls_token");
    i = _Cat( {s, i}, SEQUENCE, name +".rgbt_cls.cat");
    s = _Parameter(c, {}, 1, 257, 1, 1280, name +".pos_embedding_helper.pos_embed");
    i = _Add({i, s}, name +".pos_embed.add");
    return i;
}
NetTensor *VisonModel(Context* c, NetTensor * i,  int hidden_dim= 1280, int ffn_hidden_dim = 5120, int mutil_head_size = 16, string name = "vision"){
    i = VisionEmbedding(c, i, hidden_dim, "modality_preprocessors."+name);
    i = _LayerNorm( {i},  hidden_dim,  true,1e-6, "modality_trunks."+name + ".pre_transformer_layer.0");
    for(int layer=0; layer<32; ++layer) {
        auto *x = _LayerNorm( {i},  hidden_dim,  true,1e-6, "modality_trunks."+name + ".blocks."+std::to_string(layer)+".norm_1");
        x = Attention( x, hidden_dim, hidden_dim / mutil_head_size, mutil_head_size, "modality_trunks."+name + ".blocks."+std::to_string(layer)+".attn");
        i = _Add( {x, i}, "modality_trunks."+name + ".blocks."+std::to_string(layer)+".add_attn");
        x = _LayerNorm( {i}, hidden_dim, true, 1e-6, "modality_trunks."+name + ".blocks."+std::to_string(layer)+".norm_2");
        x = MLP( x, hidden_dim, ffn_hidden_dim, "modality_trunks."+name + ".blocks."+std::to_string(layer)+ ".mlp");
        i = _Add( {x, i}, "modality_trunks."+name + ".blocks."+std::to_string(layer)+".add_mlp");
    }
    i = _LayerNorm( {i}, hidden_dim, true,  1e-6, "modality_heads."+ name + ".0");
    i = i->clip( {}, {}, {0}, {});
    i = _Linear( {i}, hidden_dim, 1024, false, "modality_heads."+ name + ".2");
    i = _Division( {i, i->norm(2)}, "modality_postprocessors."+name +".division");
    return i;
}


NetTensor *TextEmbedding(Context *c, NetTensor * i,  int vocab_size, int hidden_dim, int max_position_embeddings, string name) {
    i = _Embedding( {i}, vocab_size, hidden_dim, name +".token_embedding");
    auto *s = _Parameter(c, {}, 1, max_position_embeddings, 1, hidden_dim, name +".pos_embed");
    i = _Add( {s, i}, name+".add_embd");
    return i;
}
NetTensor *TextModel(Context *c, NetTensor * i,  NetTensor * in_len, int vocab_size = 49408, int hidden_dim = 1024, int ffn_hidden_dim = 4096, int mutil_head_size = 16, string name="text") {
    i = TextEmbedding(c, i, vocab_size, hidden_dim, 77, "modality_preprocessors."+name);
    for (int layer = 0; layer < 24; ++layer) {
        auto *x = _LayerNorm( {i}, hidden_dim, true, 1e-6, "modality_trunks."+name+".blocks." + std::to_string(layer) + ".norm_1");
        x = Attention( x, hidden_dim, hidden_dim / mutil_head_size, mutil_head_size, "modality_trunks."+name+".blocks." + std::to_string(layer) + ".attn");
        i = _Add( {x, i}, "modality_trunks."+name+".blocks." + std::to_string(layer) + ".add_attn");
        x = _LayerNorm( {i}, hidden_dim, true, 1e-6, "modality_trunks."+name+".blocks." + std::to_string(layer) + ".norm_2");
        x = MLP( x, hidden_dim, ffn_hidden_dim, "modality_trunks."+name+".blocks." + std::to_string(layer) + ".mlp");
        i = _Add( {x, i}, "modality_trunks."+name+".blocks." + std::to_string(layer) + ".add_mlp");
    }
    i = i->_clip({}, {}, {in_len}, {});
    i = _LayerNorm( {i}, hidden_dim,true, 1e-6,"modality_heads."+ name + ".proj.0");
    i = _Linear( {i}, hidden_dim, 1024, false, "modality_heads."+ name + ".proj.1");
    i = _Division( {i, i->norm(2)}, "modality_postprocessors."+name +".division");
    i = _Scale( {i}, 100.0, 0.0F, false, "modality_postprocessors."+name +".logit_scale");
    return i;
}

NetTensor *AudioEmbedding(Context *c, NetTensor * i, int hidden_size, string name) { //input: 9, 1, 128, 204
    i = _Convolution2D({i}, 1, 768, {16, 16}, {10, 10}, VALID, false, name +".rgbt_stem.proj"); // 9, 768, 12, 19
    i = i->transpose(SEQUENCE, DIMENSION);
    i = i->flatten(HEAD, SEQUENCE);
    i = _LayerNorm( {i}, hidden_size, true, 1e-6, name +".norm_layer.proj_norm");
    auto *s = _Parameter(c, {}, 1, 1, 1, 768, name +".cls_token");
    i = _Cat( {s, i}, SEQUENCE, name +".cls_token.cat");
    s = _Parameter(c, {}, 1, 229, 1, 768, name +".pos_embedding_helper.pos_embed");
    i = _Add( {i, s}, name +".position_embeddings.add");
    return i;
}
NetTensor *AudioModel(Context* c, NetTensor * i,  int hidden_dim= 768, int ffn_hidden_dim = 3072, int mutil_head_size = 12, string name = "audio"){
    i = AudioEmbedding(c, i, hidden_dim, "modality_preprocessors."+name);
    i = _LayerNorm( {i},  hidden_dim,  true,1e-6, "modality_trunks."+name + ".pre_transformer_layer.0");
    for(int layer=0; layer<12; ++layer) {
        auto *x = _LayerNorm( {i},  hidden_dim,  true,1e-6, "modality_trunks."+name + ".blocks."+std::to_string(layer)+".norm_1");
        x = Attention( x, hidden_dim, hidden_dim / mutil_head_size, mutil_head_size, "modality_trunks."+name + ".blocks."+std::to_string(layer)+".attn");
        i = _Add( {x, i}, "modality_trunks."+name + ".blocks."+std::to_string(layer)+".add_attn");
        x = _LayerNorm( {i}, hidden_dim, true, 1e-6, "modality_trunks."+name + ".blocks."+std::to_string(layer)+".norm_2");
        x = MLP( x, hidden_dim, ffn_hidden_dim, "modality_trunks."+name + ".blocks."+std::to_string(layer)+ ".mlp");
        i = _Add( {x, i}, "modality_trunks."+name + ".blocks."+std::to_string(layer)+".add_mlp");
    }
    i = _LayerNorm( {i}, hidden_dim, true,  1e-6, "modality_heads."+ name + ".0");
    i = i->clip( {}, {}, {0}, {});
    i = _Linear( {i}, hidden_dim, 1024, false, "modality_heads."+ name + ".2");
    i = _Division( {i, i->norm(2)}, "modality_postprocessors."+name +".division");
    return i;
}

void ImageBind(Context* c) {
    auto *i = _Input(c, {}, "input_ids");
    auto *i_len = _Input(c, {}, "input_lens");
    i = TextModel(c, i, i_len);
    auto *p = _Input(c, {}, "input_imgs");
    p = VisonModel(c, p);
    i = i->transpose(BATCH, SEQUENCE);
    p = p->transpose(BATCH, SEQUENCE);
    i = _Matmul( {p, i}, false, true, "final.vision@text");
    i = _Softmax( {i}, DIMENSION, "final.softmax");
}
int main(int argc, char **argv) {
    cmdline::parser cmdParser;
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "./clip_vocab.mllm");
    cmdParser.add<string>("model", '\0', "specify mllm model path", false, "../models/imagebind_huge-q4_k.mllm");
    cmdParser.parse_check(argc, argv);

    string vocab_path = cmdParser.get<string>("vocab");
    string model_path = cmdParser.get<string>("model");

    // auto tokenizer = BPETokenizer(vocab_path);

    std::unique_ptr<Context> c_ptr(new Context());
    auto *c = c_ptr.get();
    ImageBind(c);

    BackendConfig bn;
    Net net(bn);
    net.convert(c->sub_param_);

    ParamLoader param_loader(model_path);
    Executor ex(&param_loader);
    ex.setup(&net);

    auto tokenizer = new BPETokenizer(vocab_path);
    std::unordered_map<string,unsigned> merge_rank;
    auto merge_file = std::ifstream("./clip_merges.txt");
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

    vector<string> in_strs = {"a dog.", "A car", "A bird"};
    auto tokens_ids = vector<vector<token_id_t>>();
    for (auto in_str : in_strs) {
        in_str = toLowercase(in_str);
        auto tokens_id = vector<token_id_t>();
        tokenizer->tokenize(in_str, tokens_id, true);
        for (auto t :tokens_id) {
            std::cout << t<<" ";
        }
        std::cout<<std::endl;
        tokens_ids.push_back(tokens_id);
    }
    //TODO Tokenizer
    tokens_ids[0] = {49406,   320,  1929,   269, 49407};
    tokens_ids[1] = {49406,   320,  1615, 49407};
    tokens_ids[2] = {49406,   320,  3329, 49407};
    shared_ptr<Tensor> input_text = std::make_shared<Tensor>();
    shared_ptr<Tensor> input_text_lens = std::make_shared<Tensor>();
    tokens2Tensor(&net, tokens_ids, input_text, input_text_lens);

    vector<string> img_names = {"dog_image.jpg", "car_image.jpg", "bird_image.jpg"};
    // vector<float*> data_imgs;
    vector< vector< vector<vector<float>>>> data_imgs;
    auto* clip = new ClipProcessor(tokenizer);
    clip->PreProcessImages(img_names);
    data_imgs = clip->pixel_values_;
    shared_ptr<Tensor> input_img = std::make_shared<Tensor>();
    img2Tensor(input_img, net, data_imgs);

    ex.run(&net, {input_text, input_text_lens, input_img});
    auto result = ex.result();
    result[0]->printData<float>();

    // free memory
    for (auto *op : c->net_ops) {
        delete op;
    }
    for (auto *tensor : c->net_tensors) {
        delete tensor;
    }
    return 0;
}
