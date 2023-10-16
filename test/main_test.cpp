#include <iostream>
#include "Net.hpp"
#include "Executor.hpp"
#include "NetParameter.hpp"
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

void fullTensor(shared_ptr<Tensor> input_tensor, Net net, vector<int> shape, float value) {
    input_tensor->setBackend(net.backends()[BackendType::MLLM_CPU]);
    input_tensor->reshape(shape);
    input_tensor->alloc();
    input_tensor->fullData<float>(value);
}
void token2Tensor(shared_ptr<Tensor> input_tensor, Net net, vector<token_id_t> tokens){
    input_tensor->setBackend(net.backends()[BackendType::MLLM_CPU]);
    input_tensor->reshape({1,1,static_cast<int>(tokens.size()),1});
    input_tensor->alloc();
    input_tensor->fullData<float>(1);
    for (int idx = 0; idx < tokens.size(); ++idx) {
        input_tensor->setDataAt<float>(0,0,idx,0,tokens[idx]);
    }
}
int main() {
    /*
    //  test model sample
     Context *c = new Context();
     auto* x = _Input(c, {1, 3, 3, 3});
     x = _Softmax(c, {x}, -1);
     auto* z = _SiLU(c, {x});
     _SubgraphBegin(c);
     auto* y = _SiLU(c, {x});
     x = _Matmul(c, {z, y});
     x = _Softmax(c, {x}, -1);
     */

    /*
    // decoder blk sample
    Context *c = new Context();
    auto *in = _Input(c, {1, 1, 1, 1});
    auto *x = _RMSNorm(c, {in});
    auto *q = _Linear(c, {x}, 1, 1, false);
    auto *k = _Linear(c, {x}, 1, 1, false);
    auto *v = _Linear(c, {x}, 1, 1, false);
    auto *o = _Matmul(c, {q, k});
    o = _Softmax(c, {o}, 1);
    o = _Matmul(c, {o, v});
    o = _Linear(c, {o}, 1, 1, false);
    o = _Add(c, {o, in});
    _SubgraphBegin(c);
    x = _RMSNorm(c, {o});
    x = _Linear(c, {x}, 1, 1, false);
    x = _Linear(c, {x}, 1, 1, false);
    o = _Add(c, {o, x});
    */

    auto tokenizer = BPETokenizer("../tools/convertor/vocab.mllm");
    auto tokens_id = vector<token_id_t>();
    //    tokenizer.tokenize(string(" this is 🦙.cpp"), tokens_id, true);
    tokenizer.tokenize(string(" 你所热爱的，就是你的生活"), tokens_id, true);
    for (auto idx : tokens_id) {
        std::cout << idx << ",";
    }
    std::cout << std::endl;
//    std::cout << tokenizer.detokenize(tokens_id) << std::endl;

    int vocab_size = 128;
    int hidden_dim = 80;
    int mutil_head_size = 8;
    Context *c = new Context();
    auto *i = _Input(c);
    i = _Embedding(c, {i}, vocab_size, hidden_dim);
    _SubgraphBegin(c);
    auto *x = _RMSNorm(c, {i});
    x = _Attention(c, {x}, hidden_dim, hidden_dim / mutil_head_size, mutil_head_size);
    auto *j = _Add(c, {x, i});
    i = _RMSNorm(c, {j});
    x = _Linear(c, {i}, hidden_dim, hidden_dim * 4, false);
    x = _SiLU(c, {x});
    auto *y = _Linear(c, {i}, hidden_dim, hidden_dim * 4, false);
    x = _Dot(c, {x, y});
    x = _Linear(c, {x}, hidden_dim * 4, hidden_dim, false);
    x = _Add(c, {x, j});
    x = _Linear(c, {x}, hidden_dim, vocab_size, false);
    // display(c);
    BackendConfig bn;
    Net net(c->sub_param_, bn);
    net.convert();
    // net.Run();
    Executor ex(&net);
    //ParamLoader param_loader("str_name");
    //ex.execute({1, 1, 10, vocab_size});
    //ex.execute({1, 1, 1, vocab_size});
    //ex.execute({1, 1, 1, vocab_size});
    shared_ptr<Tensor> input = std::make_shared<Tensor>();
    //fullTensor(input, net, {1, 1, 10, 1}, 1);
    token2Tensor(input, net, tokens_id);
    ex.execute(input);
    return 0;
    shared_ptr<Tensor> input_2 = std::make_shared<Tensor>();
    //fullTensor(input_2, net, {1, 1, 1, 1}, 1);
    token2Tensor(input_2, net, {1});
    ex.execute(input_2);
    //fullTensor(input_2, net, {1, 1, 1, 1}, 1);
    token2Tensor(input_2, net, {1});
    ex.execute(input_2);

    auto result = ex.result();
    //result[0]->printData<float>();

    /*
    auto *x = _Input(c);
    x = _Embedding(c, {x}, 128, 1000);
    BackendConfig bn;

    Net net(c->sub_param_, bn);
    net.convert();
    // net.Run();
    Executor ex(&net);
    ex.execute({1, 1, 128, 1});
    */
    return 0;
}
