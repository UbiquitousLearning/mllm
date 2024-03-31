#include <iostream>
#include <valarray>
#include <csignal>
#include "cmdline.h"
#include "Net.hpp"
#include "Executor.hpp"
#include "express/Express.hpp"
#include "tokenizers/BPE/Bpe.hpp"
using namespace mllm;

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
    assert(result->batch() == 1);
    assert(result->head() ==  1);
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


NetTensor *Attention( NetTensor * x, int embedding_size, int hidden_size, int head_size, int mutil_key_value_head, int cache_max, string name){
    auto *q =_Linear({x}, embedding_size, hidden_size * head_size, false, name + ".q_proj");
    auto *k =_Linear({x}, embedding_size, hidden_size * mutil_key_value_head, false, name + ".k_proj");
    auto *v =_Linear({x}, embedding_size, hidden_size * mutil_key_value_head, false, name + ".v_proj");
    q = q->view(-1, head_size, -1, hidden_size);
    k = k->view(-1, mutil_key_value_head, -1, hidden_size);
    v = v->view(-1, mutil_key_value_head, -1, hidden_size);
    q = _RoPE( {q}, HFHUBROPE, name + ".q_rope");
    k = _RoPE( {k}, HFHUBROPE, name + ".k_rope");
    k = _KVCache( {k},head_size/mutil_key_value_head,  cache_max,  name + ".k_cache");
    v = _KVCache( {v},head_size/mutil_key_value_head, cache_max,  name + ".v_cache");
    auto *qk = _Matmul( {q, k}, false, true, name + ".qk");
    qk = *qk/std::sqrt(hidden_size);
    qk = _Causalmask( {qk}, name + ".mask");
    qk = _Softmax( {qk}, DIMENSION, name + ".softmax");
    auto *o = _Matmul( {qk, v}, false, false, name + ".qkv");
    o = o->view(-1, 1, -1, hidden_size * head_size);
    o = _Linear( {o}, hidden_size * head_size, embedding_size, false, name + ".o_proj");
    return o;
}
NetTensor *FFN( NetTensor * i, int hidden_dim, int ffn_hidden_dim, string name){
    auto *ids = _Predictor({i}, hidden_dim, ffn_hidden_dim, name);
    auto *x = _SparseIdLinear( {i, ids}, hidden_dim, ffn_hidden_dim, name+".gate_proj");
    x = _ReLU( {x}, name+".relu");
    auto *y = _SparseIdLinear( {i, ids}, hidden_dim, ffn_hidden_dim, name+".up_proj");
    x = *x*y;// x = _Mul( {x, y}, name+".dot");
    x = _SparseLinear( {x}, ffn_hidden_dim, hidden_dim, name+".down_proj");
    return x;
}
void ReLULlama(Context* c, int vocab_size= 32000, int hidden_dim= 2048, int ffn_hidden_dim = 5632, int mutil_head_size = 32, int mutil_key_value_head= 4, int cache_max= 200){
    auto *i = _Input(c);
    i = _Embedding( {i}, vocab_size, hidden_dim, (string)"model.embed_tokens");
    // loop
    for(int layer=0; layer<32; ++layer) {
        auto *x = _RMSNorm( {i}, hidden_dim, 1e-6, (string)"model.layers."+std::to_string(layer)+".input_layernorm");
        i = *Attention( x, hidden_dim, hidden_dim / mutil_head_size, mutil_head_size, mutil_key_value_head, cache_max, (string)"model.layers."+std::to_string(layer)+".self_attn") +i;
        x = _RMSNorm( {i}, hidden_dim, 1e-6, (string)"model.layers."+std::to_string(layer)+".post_attention_layernorm");
        i = *FFN( x, hidden_dim, ffn_hidden_dim, (string)"model.layers."+std::to_string(layer) +".mlp") +i;
        //_SubgraphBegin(c);
    }
    // end loop
    i = _RMSNorm( {i}, hidden_dim, 1e-6, (string)"model.norm");
    i = _Linear( {i}, hidden_dim, vocab_size, false, "lm_head");
}

void run_inference(int argc, char **argv){
    cmdline::parser cmdParser;
    //    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "./vocab/ReLULlama_vocab.mllm");
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "../vocab/relu_llama_vocab.mllm");
    cmdParser.add<string>("model", 'm', "specify mllm model path", false, "../ReLULlama_new.mllm");
    cmdParser.add<string>("predictor", 'p', "specify mllm model predictor path", false, "../ReLULlama_predictor.mllm");
    cmdParser.add<int>("limits", 'l',  "max KV cache size", false, 600);
    cmdParser.add<int>("thread", 't', "num of threads", false, 4);
    cmdParser.parse_check(argc, argv);

    string vocab_path = cmdParser.get<string>("vocab");
    string model_path = cmdParser.get<string>("model");
    string predictor_path = cmdParser.get<string>("predictor");
    int tokens_limit = cmdParser.get<int>("limits");
    int thread_num = cmdParser.get<int>("thread");

    int vocab_size = 32000;
    int hidden_dim = 4096;
    int ffn_hidden_dim = 11008;
    int mutil_head_size = 32;
    int key_value_head_size = 32;

    std::unique_ptr<Context> c_ptr(new Context());
    auto *c = c_ptr.get();
    ReLULlama(c, vocab_size, hidden_dim, ffn_hidden_dim, mutil_head_size, key_value_head_size, tokens_limit);

    BackendConfig bn;
    Net net(bn);
    net.convert(c->sub_param_, BackendType::MLLM_CPU, thread_num);

    // tokenize input
    std::cout << "start to tokenize input" << std::endl;
    auto tokenizer = BPETokenizer(vocab_path);
    auto prompt = " Hello! Who are you?";                     // prompt
    shared_ptr<Tensor> input = std::make_shared<Tensor>();
    input->setName("input");
    auto tokens_id = vector<token_id_t>();
    tokenizer.tokenize(prompt, tokens_id, true);
    BPETokenizer::token2Tensor( &net, tokens_id, input);
    input->printData<float>();
    printf("token_ids:");
    for (auto id:tokens_id)
        printf("%d ", id);
    printf("\n");

    // set up net and load parameters
    MultiFileParamLoader param_loader({model_path, predictor_path});
    Executor ex(&param_loader);
    ex.setup(&net);  // load params

    // forward pass
    {
        std::cout << "[Q] " << prompt << std::endl;
        std::cout << "[A] " << std::flush;
        for (int step = 0; step < 100; step++) {
            ex.run(&net, {input});
            auto result = ex.result();
            auto token_idx = postProcessing(result[0], input);
            // std::cout <<token_idx<<"  " << std::flush;
            if (token_idx == 2) { // "</s>"
                break;
            }
            auto out_token = tokenizer.detokenize({token_idx});
            std::cout << out_token << std::flush;
        }
        printf("\n");
    }

    ex.perf();

    // free memory
    for (auto *op : c->net_ops) {
        delete op;
    }
    for (auto *tensor : c->net_tensors) {
        delete tensor;
    }
}

int main(int argc, char **argv) {
    run_inference(argc, argv);

    return 0;
}
