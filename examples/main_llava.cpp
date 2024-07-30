#include <iostream>
#include <valarray>
#include <csignal>
#include "cmdline.h"
#include "Net.hpp"
#include "Executor.hpp"
#include "express/Express.hpp"
#include "tokenizers/BPE/Bpe.hpp"
#include "processor/ClipPreProcess.hpp"

void print2DVetcors(std::vector<std::vector<float>> chunk_feats) {
    std::cout << std::fixed;
    std::cout << std::setprecision(4);
    for (size_t i = 0; i < chunk_feats.size(); ++i) {
        for (size_t j = 0; j < chunk_feats[i].size(); ++j) {
            std::cout << chunk_feats[i][j] << ",";
        }
        std::cout << std::endl;
    }
}
void print3DVetcors(std::vector<std::vector<std::vector<float>>> all_clips) {
    for (auto all_clip : all_clips) {
        print2DVetcors(all_clip);
        std::cout << "======================================" << std::endl;
    }
    std::cout << " [" << all_clips.size() << ", " << all_clips[0].size() << ", " << all_clips[0][0].size() << "]" << std::endl;
}

int cache_max = 700;

using namespace mllm;
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
unsigned int postProcessing(shared_ptr<Tensor> result, shared_ptr<Tensor> &out_result, shared_ptr<Tensor> &input_img) {
    assert(result->batch() == 1);
    assert(result->head() == 1);
    out_result->reshape(1, 1, 1, 1);
    out_result->alloc();
    vector<float> scores;
    for (int i = 0; i < result->dimension(); ++i) {
        auto value = result->dataAt<float>(0, 0, result->sequence() - 1, i);
        scores.push_back(value);
    }
    auto token_idx = argmax(scores);
    out_result->setDataAt<float>(0, 0, 0, 0, token_idx);
    input_img->reshape(0, 0, 0, 0);
    input_img->alloc();
    return token_idx;
}
NetTensor *Attention(NetTensor *x, int embedding_size, int hidden_size, int head_size, int cache_max, string name) {
    auto *q = _Linear({x}, embedding_size, hidden_size * head_size, false, name + ".q_proj");
    auto *k = _Linear({x}, embedding_size, hidden_size * head_size, false, name + ".k_proj");
    auto *v = _Linear({x}, embedding_size, hidden_size * head_size, false, name + ".v_proj");
    q = q->view(-1, head_size, -1, hidden_size);
    k = k->view(-1, head_size, -1, hidden_size);
    v = v->view(-1, head_size, -1, hidden_size);
    q = _RoPE({q}, HFHUBROPE, name + ".q_rope");
    k = _RoPE({k}, HFHUBROPE, name + ".k_rope");
    k = _KVCache({k}, cache_max, name + ".k_cache");
    v = _KVCache({v}, cache_max, name + ".v_cache");
    auto *qk = _Matmul({q, k}, false, true, name + ".qk");
    qk = *qk / std::sqrt(hidden_size);
    // qk = _Causalmask({qk}, name + ".mask");
    qk = _Softmax({qk}, DIMENSION, true, name + ".softmax");
    auto *o = _Matmul({qk, v}, false, false, name + ".qkv");
    o = o->view(-1, 1, -1, hidden_size * head_size);
    o = _Linear({o}, hidden_size * head_size, embedding_size, false, name + ".o_proj");
    return o;
}
NetTensor *FFN(NetTensor *i, int hidden_dim, int ffn_hidden_dim, string name) {
    auto *x = _Linear({i}, hidden_dim, ffn_hidden_dim, false, name + ".gate_proj");
    x = _SiLU({x}, name + ".silu");
    auto *y = _Linear({i}, hidden_dim, ffn_hidden_dim, false, name + ".up_proj");
    x = *x * y; // x = _Mul( {x, y}, name+".dot");
    x = _Linear({x}, ffn_hidden_dim, hidden_dim, false, name + ".down_proj");
    return x;
}

NetTensor *text_embd(NetTensor *i, int vocab_size = 32064, int hidden_dim = 4096, string name = "language_model") {
    // auto *i = _Input(c);
    i = _Embedding({i}, vocab_size, hidden_dim, name + ".model.embed_tokens");
    return i;
}
NetTensor *llama(NetTensor *i, int vocab_size = 32064, int hidden_dim = 4096, int ffn_hidden_dim = 11008, int mutil_head_size = 32, int cache_max = 700, string name = "language_model") {
    // i = _Embedding( {i}, vocab_size, hidden_dim, name+"model.embed_tokens");
    for (int layer = 0; layer < 32; ++layer) {
        auto *x = _RMSNorm({i}, hidden_dim, 1e-6, name + ".model.layers." + std::to_string(layer) + ".input_layernorm");
        i = *Attention(x, hidden_dim, hidden_dim / mutil_head_size, mutil_head_size, cache_max, name + ".model.layers." + std::to_string(layer) + ".self_attn") + i;
        x = _RMSNorm({i}, hidden_dim, 1e-6, name + ".model.layers." + std::to_string(layer) + ".post_attention_layernorm");
        i = *FFN(x, hidden_dim, ffn_hidden_dim, name + ".model.layers." + std::to_string(layer) + ".mlp") + i;
        //_SubgraphBegin(c);
    }
    i = _RMSNorm({i}, hidden_dim, 1e-6, name + ".model.norm");
    i = _Linear({i}, hidden_dim, vocab_size, false, name + ".lm_head");
    return i;
}

NetTensor *VisionAttention(NetTensor *x, int embedding_size, int hidden_size, int head_size, string name) {
    auto *q = _Linear({x}, embedding_size, hidden_size * head_size, true, name + ".q_proj");
    auto *k = _Linear({x}, embedding_size, hidden_size * head_size, true, name + ".k_proj");
    auto *v = _Linear({x}, embedding_size, hidden_size * head_size, true, name + ".v_proj");
    q = q->view(-1, head_size, -1, hidden_size);
    k = k->view(-1, head_size, -1, hidden_size);
    v = v->view(-1, head_size, -1, hidden_size);
    auto *qk = _Matmul({q, k}, false, true, name + ".qk");
    qk = _Scale({qk}, 1.0F / std::sqrt(hidden_size), 0.0F, false, name + ".scale");
    if (name.find("text_model") != std::string::npos) {
        qk = _Softmax( {qk}, DIMENSION, true, name + ".softmax");
    } else{
        qk = _Softmax( {qk}, DIMENSION, false, name + ".softmax");
    }
    auto *o = _Matmul({qk, v}, false, false, name + ".qkv");
    o = o->view(-1, 1, -1, hidden_size * head_size);
    o = _Linear({o}, hidden_size * head_size, embedding_size, true, name + ".out_proj");
    return o;
}
NetTensor *MLP(NetTensor *i, int hidden_dim, int ffn_hidden_dim, string name) {
    auto *x = _Linear({i}, hidden_dim, ffn_hidden_dim, true, name + ".fc1");
    x = _QuickGELU({x}, name + ".act_fn");
    x = _Linear({x}, ffn_hidden_dim, hidden_dim, true, name + ".fc2");
    return x;
}
NetTensor *VisionEmbedding(Context *c, NetTensor *i, int hidden_size, string name) {
    i = _Convolution2D({i}, 3, 1024, {14, 14}, {14, 14}, VALID, false, name + ".patch_embedding");
    i = i->transpose(SEQUENCE, DIMENSION);
    i = i->flatten(HEAD, SEQUENCE);
    auto *s = _Parameter(c, {}, 1, 1, 1, 1024, name + ".class_embedding");
    i = _Cat({s, i}, SEQUENCE, name + ".class_embedding.cat");
    s = _Range(c, {}, 0, 577, name + ".position_ids");
    i = *_Embedding({s}, 577, 1024, name + ".position_embedding") + i;
    return i;
}
NetTensor *vision_tower(Context *c, NetTensor *i, int hidden_dim = 1024, int ffn_hidden_dim = 4096, int mutil_head_size = 16, string name = "vision_tower.vision_model") {
    i = VisionEmbedding(c, i, hidden_dim, name + ".embeddings");
    i = _LayerNorm({i}, hidden_dim, true, 1e-6, name + ".pre_layrnorm");
    for (int layer = 0; layer < 23; ++layer) {
        auto *x = _LayerNorm({i}, hidden_dim, true, 1e-6, name + ".encoder.layers." + std::to_string(layer) + ".layer_norm1");
        i = *VisionAttention(x, hidden_dim, hidden_dim / mutil_head_size, mutil_head_size, name + ".encoder.layers." + std::to_string(layer) + ".self_attn") + i;
        x = _LayerNorm({i}, hidden_dim, true, 1e-6, name + ".encoder.layers." + std::to_string(layer) + ".layer_norm2");
        i = *MLP(x, hidden_dim, ffn_hidden_dim, name + ".encoder.layers." + std::to_string(layer) + ".mlp") + i;
    }
    i = i->clip({}, {}, {1, 577}, {});
    i = _Linear({i}, hidden_dim, ffn_hidden_dim, true, "multi_modal_projector.linear_1");
    i = _GELU({i}, "multi_modal_projector.act_fn");
    i = _Linear({i}, ffn_hidden_dim, ffn_hidden_dim, true, "multi_modal_projector.linear_2");
    // i = _LayerNorm( {i}, hidden_dim, true,  1e-6, name + ".post_layernorm");
    return i;
}

void llava(Context *c, int cache_max = 700) {
    auto *i = _Input(c, {}, "input_text");
    auto *e = text_embd(i);
    i = i->where(32000, SEQUENCE);
    auto *v = _Input(c, {}, "input_imgs");
    v = vision_tower(c, v);
    i = _Replace({e, v, i});
    i = llama(i, 32064, 4096, 11008, 32, cache_max, "language_model");
    i = i->clip({}, {}, {-1}, {});
}

int main(int argc, char **argv) {
    cmdline::parser cmdParser;
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "../vocab/llava_vocab.mllm");
    cmdParser.add<string>("model", 'm', "specify mllm model path", false, "../models/llava-1.5-7b-q4_k.mllm");
    cmdParser.add<int>("limits", 'l', "max KV cache size", false, 700);
    cmdParser.add<int>("thread", 't', "num of threads", false, 4);
    cmdParser.parse_check(argc, argv);

    string vocab_path = cmdParser.get<string>("vocab");
    string model_path = cmdParser.get<string>("model");
    int tokens_limit = cmdParser.get<int>("limits");
    int thread_num = cmdParser.get<int>("thread");

    std::unique_ptr<Context> c_ptr(new Context());
    auto *c = c_ptr.get();
    llava(c, tokens_limit);

    BackendConfig bn;
    Net net(bn);
    net.convert(c->sub_param_, BackendType::MLLM_CPU, thread_num);

    ParamLoader param_loader(model_path);
    Executor ex(&param_loader);
    ex.setup(&net);

    auto tokenizer = new BPETokenizer(vocab_path);
    std::unordered_map<string,unsigned> merge_rank;
    auto merge_file = std::ifstream("../vocab/llava_merges.txt");
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

    std::vector<vector<string>> in_imgs = {
        {"../assets/australia.jpg"}};
    vector<string> in_strs = {
        "<image>\nUSER: What's the content of the image?\nASSISTANT:"};

    for (int inId = 0; inId < in_strs.size(); ++inId) {
        auto in_str = in_strs[0];
        if (in_str[0] != ' ') {
            in_str = ' ' + in_str;
        }
        auto in_img = in_imgs[0];

        auto tokens_ids = vector<vector<token_id_t>>();
        vector<mllm::token_id_t> tokens_id = {};
        tokenizer->tokenize(BPETokenizer::replaceString(in_str,' ',"▁"), tokens_id, {"<image>", "<pad>", "\n"});
        tokens_ids.push_back(tokens_id);

        shared_ptr<Tensor> input_text = std::make_shared<Tensor>();
        BPETokenizer::tokens2Tensor(&net, tokens_ids, input_text);

        auto *clip_processor = new ClipPreProcessor(tokenizer, 336, 336);
        clip_processor->PreProcessImages(in_img);
        auto images = clip_processor->pixel_values_[0];

        shared_ptr<Tensor> input_img = std::make_shared<Tensor>();
        clip_processor->Img2Tensor(net.backends()[MLLM_CPU].get(), input_img, images);

        std::cout << in_strs[0] << std::flush;
        for (int step = 0; step < 30; step++) {
            ex.run(&net, {input_text, input_img});
            auto result = ex.result();
            auto token_idx = postProcessing(result[0], input_text, input_img);
            if (token_idx == 2) { // "</s>"
                break;
            }
            auto out_token = tokenizer->detokenize({token_idx});
            std::cout << out_token << std::flush;
        }
        std::cout << std::endl;
    }


    ex.perf();

    // free memory
    for (auto *op : c->net_ops) {
        delete op;
    }
    for (auto *tensor : c->net_tensors) {
        delete tensor;
    }
    return 0;
}
