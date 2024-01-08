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
// #include "processor/PreProcess.hpp"
#include "processor/ClipPreProcess.hpp"
#include <cmath>
#include <vector>
#include <numeric>

using namespace mllm;


void img2Tensor(shared_ptr<Tensor> input_tensor, Net &net, float* img, int height, int width, int channel) {
    input_tensor->setBackend(net.backends()[BackendType::MLLM_CPU].get());
    input_tensor->reshape(1, height, channel, width);
    input_tensor->setDtype(MLLM_TYPE_F32);
    input_tensor->alloc();
    for (int h = 0; h < height; ++h) {
        for (int c = 0; c < channel; ++c) {
            for (int w = 0; w < width; ++w) {
                input_tensor->setDataAt<float>(0, h, c, w, img[(h * width + w) * channel + c]);
            }
        }
    }
}
void img2Tensor(shared_ptr<Tensor> input_tensor, Net &net, vector<vector<vector<float>>> img) {
    int channel = img.size();
    int height = img[0].size();
    int width= img[0][0].size();
    input_tensor->setBackend(net.backends()[BackendType::MLLM_CPU].get());
    input_tensor->reshape(1, height, channel, width);
    input_tensor->setDtype(MLLM_TYPE_F32);
    input_tensor->alloc();
    for (int h = 0; h < height; ++h) {
        for (int c = 0; c < channel; ++c) {
            for (int w = 0; w < width; ++w) {
                input_tensor->setDataAt<float>(0, h, c, w, img[c][h][w]);
            }
        }
    }
}
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
        qk = _Causalmask( {qk}, name + ".mask");
    }
    qk = _Softmax( {qk}, DIMENSION, name + ".softmax");
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
    s = _Embedding( {s}, 50, 768, name +".position_embedding");
    i = _Add( {i, s}, name +".position_embeddings.add");
    return i;
}
NetTensor *TextEmbedding(Context *c, NetTensor * i,  int vocab_size, int hidden_dim, int max_position_embeddings, string name) {
    i = _Embedding( {i}, vocab_size, hidden_dim, name +".token_embedding");
    auto *s = _Parameter(c, {}, 1, max_position_embeddings, 1, 1, name +".position_ids");
    s = s->_clip({}, {}, {0, i->shape(SEQUENCE)}, {});
    s = _Embedding( {s}, max_position_embeddings, hidden_dim, name +".position_embedding");
    i = _Add( {s, i}, name+".add_embd");
    return i;
}
NetTensor *transformer(Context *c, NetTensor * i,  int vocab_size = 49408, int hidden_dim = 512, int ffn_hidden_dim = 2048, int mutil_head_size = 8, string name="text_model") {
    // auto *i = _Input(c);
    i = TextEmbedding(c, i, vocab_size, hidden_dim, 77, name+".embeddings");
    // loop
    for (int layer = 0; layer < 12; ++layer) {
        auto *x = _LayerNorm( {i}, hidden_dim, true, 1e-6, name+".encoder.layers." + std::to_string(layer) + ".layer_norm1");
        x = Attention( x, hidden_dim, hidden_dim / mutil_head_size, mutil_head_size, name+".encoder.layers." + std::to_string(layer) + ".self_attn");
        i = _Add( {x, i}, name+".encoder.layers." + std::to_string(layer) + ".add_attn");
        x = _LayerNorm( {i}, hidden_dim, true, 1e-6, name+".encoder.layers." + std::to_string(layer) + ".layer_norm2");
        x = MLP( x, hidden_dim, ffn_hidden_dim, name+".encoder.layers." + std::to_string(layer) + ".mlp");
        i = _Add( {x, i}, name+".encoder.layers." + std::to_string(layer) + ".add_mlp");
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
        x = Attention( x, hidden_dim, hidden_dim / mutil_head_size, mutil_head_size, name + ".encoder.layers."+std::to_string(layer)+".self_attn");
        i = _Add( {x, i}, name + ".encoder.layers."+std::to_string(layer)+".add_attn");
        x = _LayerNorm( {i}, hidden_dim, true, 1e-6, name + ".encoder.layers."+std::to_string(layer)+".layer_norm2");
        x = MLP( x, hidden_dim, ffn_hidden_dim, name + ".encoder.layers."+std::to_string(layer)+ ".mlp");
        i = _Add( {x, i}, name + ".encoder.layers."+std::to_string(layer)+".add_mlp");
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
    i = _Division( {i, _Norm( {i}, 2, "text_norm")}, "text_division");
    p = _Linear( {p}, 768, 512, false, "visual_projection");
    p = _Division( {p, _Norm( {p}, 2, "visual_norm")}, "visual_division");
    auto *o = _Matmul( {i, p}, false, true, "matmul");
    o = _Scale( {o}, 100.0, 0.0F, false, "scale");
}
int main(int argc, char **argv) {
    cmdline::parser cmdParser;
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "./clip_vocab.mllm");
    cmdParser.add<string>("model", '\0', "specify mllm model path", false, "../models/clip-q4_k.mllm");
    // cmdParser.add<string>("input", 'i', "specify input string", false, " Structured pruning and unstructured pruning represent two distinct categories within the realm of parameter pruning for LLMs. Structured pruning involves the removal of entire structured components, such as neurons, channels, or layers, based on predefined criteria. This method aims to simplify the model architecture by discarding specific structural elements that contribute less to overall performance. On the other hand, unstructured pruning targets individual weights within the model, irrespective of their structural context. This approach aims to enhance the model's sparsity by selectively eliminating less influential parameters, thereby reducing the model's footprint.The significance of parameter pruning lies in its ability to strike a balance between model size and performance. By judiciously removing redundant weights, LLMs can achieve substantial compression without compromising their capabilities. This becomes particularly relevant in scenarios where computational resources, memory constraints, or deployment on edge devices necessitate a more streamlined and resource-efficient model.");
    // cmdParser.add<string>("input", 'i', "specify input string", false, " Hello, who are you?");// I think the meaning of life is
    cmdParser.parse_check(argc, argv);

    // string in_str = cmdParser.get<string>("input");
    string vocab_path = cmdParser.get<string>("vocab");
    string model_path = cmdParser.get<string>("model");

    // auto tokenizer = BPETokenizer(vocab_path);

    // int vocab_size = 32000;
    // int hidden_dim = 4096;
    // int ffn_hidden_dim = 11008;
    // int mutil_head_size = 32;

    std::unique_ptr<Context> c_ptr(new Context());
    auto *c = c_ptr.get();
    // transformer(c);
    CLIP(c);

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

    vector<string> in_strs = {"a photo of a cat", "a photo of a dog"};
    auto tokens_ids = vector<vector<token_id_t>>();
    for (auto in_str : in_strs) {
        vector<mllm::token_id_t> tokens_id={};
        tokenizer->tokenize(in_str, tokens_id, true);
        tokens_ids.push_back(tokens_id);
    }
    shared_ptr<Tensor> input_text = std::make_shared<Tensor>();
    BPETokenizer::tokens2Tensor(&net, tokens_ids, input_text);

    shared_ptr<Tensor> input_img = std::make_shared<Tensor>();
    auto* clip = new ClipProcessor(tokenizer);
    clip->PreProcessImages({"cat.jpg"});
    auto images = clip->pixel_values_[0];
//    std::cout << "size: " << images.size()<<" " <<images[0].size()  << " " << images[0][0].size() << std::endl;
    // img2Tensor(input_img, net, data_f32, 224, 224, 3);
    img2Tensor(input_img, net, images);
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
