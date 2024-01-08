#include <iostream>
#include <valarray>
#include <csignal>
#include "cmdline.h"
#include "Net.hpp"
#include "Executor.hpp"
#include "express/Express.hpp"
#include "tokenizers/BPE/Bpe.hpp"
using namespace mllm;
// For Visualization and Debug
void display(NetParameter *net) {
    std::cout << "===NetParameter===" << std::endl;
    for (auto *op : net->net_ops) {
        std::cout << "===NetOP===" << std::endl;
        std::cout << "op->name:" << op->name << std::endl;
        std::cout << "op->type:" << op->type << std::endl;
        std::cout << "op input" << op->in.size() << std::endl;
        for (auto *input : op->in) {
            std::cout << "==Input==\ninput.name:" << input->name << std::endl;
            if (input->in != nullptr) {
                std::cout << "input op:" << input->in->name << std::endl;
            }
            std::cout << "input in subgraph:" << (input->subgraph == net) << std::endl;
            std::cout << std::endl;
        }
        std::cout << "op output" << op->out.size() << std::endl;
        for (auto *output : op->out) {
            std::cout << "output.name:" << output->name << std::endl;
            std::cout << "output op:" << output->out.size() << std::endl;
            if (!output->out.empty()) {
                std::cout << "output op:" << output->out[0]->name << std::endl;
            }
        }
        std::cout << std::endl;
    }
}
void display(Context *c) {
    for (auto sub : c->sub_param_) {
        display(&sub);
    }
}

// void fullTensor(shared_ptr<Tensor> input_tensor, Net net, vector<int> shape, float value) {
//     input_tensor->setBackend(net.backends()[BackendType::MLLM_CPU].get());
//     input_tensor->reshape(shape[0], shape[1], shape[2], shape[3]);
//     input_tensor->alloc();
//     input_tensor->fullData<float>(value);
// }

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
NetTensor *Attention( NetTensor * x, int embedding_size, int hidden_size, int head_size, string name){
    auto *q =_Linear({x}, embedding_size, hidden_size * head_size, false, name + ".wq");
    auto *k =_Linear({x}, embedding_size, hidden_size * head_size, false, name + ".wk");
    auto *v =_Linear({x}, embedding_size, hidden_size * head_size, false, name + ".wv");
    q = q->view(-1, head_size, -1, hidden_size);
    k = k->view(-1, head_size, -1, hidden_size);
    v = v->view(-1, head_size, -1, hidden_size);
    q = _RoPE( {q}, 2, name + ".q_rope");
    k = _RoPE( {k}, 2, name + ".k_rope");
    k = _KVCache( {k}, true, name + ".k_cache");
    v = _KVCache( {v}, true, name + ".v_cache");
    auto *qk = _Matmul( {q, k}, false, true, name + ".qk");
    qk = _Scale( {qk}, 1.0F / std::sqrt(hidden_size), 0.0F, false, name + ".scale");
    qk = _Causalmask( {qk}, name + ".mask");
    qk = _Softmax( {qk}, DIMENSION, name + ".softmax");
    auto *o = _Matmul( {qk, v}, false, false, name + ".qkv");
    o = o->view(-1, 1, -1, hidden_size * head_size);
    o = _Linear( {o}, hidden_size * head_size, embedding_size, false, name + ".wo");
    return o;
}
NetTensor *FFN( NetTensor * i, int hidden_dim, int ffn_hidden_dim, string name){
    auto *x = _Linear( {i}, hidden_dim, ffn_hidden_dim, false, name+".w1");
    x = _SiLU( {x}, name+".silu");
    auto *y = _Linear( {i}, hidden_dim, ffn_hidden_dim, false, name+".w3");
    x = _Mul( {x, y}, name+".dot");
    x = _Linear( {x}, ffn_hidden_dim, hidden_dim, false, name+".w2");
    return x;
}
void llama2(Context* c, int vocab_size= 32000, int hidden_dim= 4096, int ffn_hidden_dim = 11008, int mutil_head_size = 32){
    auto *i = _Input(c);
    i = _Embedding( {i}, vocab_size, hidden_dim, (string)"tok_embeddings");
    // loop
    for(int layer=0; layer<32; ++layer) {
        auto *x = _RMSNorm( {i}, hidden_dim, 1e-6, (string)"layers."+std::to_string(layer)+".attention_norm");
        x = Attention( x, hidden_dim, hidden_dim / mutil_head_size, mutil_head_size, (string)"layers."+std::to_string(layer)+".attention");
        i = _Add( {x, i}, (string)"layers."+std::to_string(layer) +".attention_add");
        x = _RMSNorm( {i}, hidden_dim, 1e-6, (string)"layers."+std::to_string(layer)+".ffn_norm");
        x = FFN( x, hidden_dim, ffn_hidden_dim, (string)"layers."+std::to_string(layer) +".feed_forward");
        i = _Add( {x, i}, (string)"layers."+std::to_string(layer) +".ffn_add");
        //_SubgraphBegin(c);
    }
    // end loop
    i = _RMSNorm( {i}, hidden_dim, 1e-6, (string)"norm");
    i = _Linear( {i}, hidden_dim, vocab_size, false, "output");
}
int main(int argc, char **argv) {
    cmdline::parser cmdParser;
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "./vocab.mllm");
    cmdParser.add<string>("model", '\0', "specify mllm model path", false, "../models/llama-2-7b-chat-q4_k.mllm");
    // cmdParser.add<string>("input", 'i', "specify input string", false, " Structured pruning and unstructured pruning represent two distinct categories within the realm of parameter pruning for LLMs. Structured pruning involves the removal of entire structured components, such as neurons, channels, or layers, based on predefined criteria. This method aims to simplify the model architecture by discarding specific structural elements that contribute less to overall performance. On the other hand, unstructured pruning targets individual weights within the model, irrespective of their structural context. This approach aims to enhance the model's sparsity by selectively eliminating less influential parameters, thereby reducing the model's footprint.The significance of parameter pruning lies in its ability to strike a balance between model size and performance. By judiciously removing redundant weights, LLMs can achieve substantial compression without compromising their capabilities. This becomes particularly relevant in scenarios where computational resources, memory constraints, or deployment on edge devices necessitate a more streamlined and resource-efficient model.");
    // cmdParser.add<string>("input", 'i', "specify input string", false, " Hello, who are you?");// I think the meaning of life is
    cmdParser.parse_check(argc, argv);

    // string in_str = cmdParser.get<string>("input");
    string vocab_path = cmdParser.get<string>("vocab");
    string model_path = cmdParser.get<string>("model");

    auto tokenizer = BPETokenizer(vocab_path);

    int vocab_size = 32000;
    int hidden_dim = 4096;
    int ffn_hidden_dim = 11008;
    int mutil_head_size = 32;

    std::unique_ptr<Context> c_ptr(new Context());
    auto *c = c_ptr.get();
    llama2(c, vocab_size, hidden_dim, ffn_hidden_dim, mutil_head_size);

    BackendConfig bn;
    Net net(bn);
    net.convert(c->sub_param_);

    ParamLoader param_loader(model_path);
    Executor ex(&param_loader);
    ex.setup(&net);


    vector<string> in_strs = {
        " Hello, who are you?",
        " What can you do?",
        "Please introduce Beijing University of Posts and Telecommunications."
    };
    shared_ptr<Tensor> input = std::make_shared<Tensor>();
    for (int str_i = 0; str_i < in_strs.size(); ++str_i)
    {
        auto in_str = in_strs[str_i];
        if(in_str[0] != ' '){
            in_str = ' '+ in_str;
        }
        auto tokens_id = vector<token_id_t>();
        tokenizer.tokenize(in_str, tokens_id, true);
        if(str_i > 0) {
            tokens_id[0] = 13;
        }
        BPETokenizer::token2Tensor( &net, tokens_id, input);
        std::cout << in_str << std::flush;
        for(int step = 0; step<100; step++) {
            ex.run(&net, {input});
            auto result = ex.result();
            auto token_idx = postProcessing(result[0], input);
            if(token_idx == 2){// "</s>"
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
    return 0;
}
