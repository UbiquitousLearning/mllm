#include <iostream>
#include "Net.hpp"
#include "Executor.hpp"
#include "NetParameter.hpp"
#include "Types.hpp"
#include "express/Express.hpp"

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
        std::cout << "==Output==" << op->out.size() << std::endl;
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

int main() {
    std::cout << "===NNAPI Test===" << std::endl;
    Context *ctx = new Context();

    auto *a = _Input(ctx);
    // auto *b = _Linear(ctx, {a}, 4, 2, false);
    auto *b = _Add(ctx, {a, a});

    BackendConfig bn;
    Net net(ctx->sub_param_, bn);
    net.convert(BackendType::MLLM_NNAPI);
    display(ctx);

    Executor ex(&net);
    shared_ptr<Tensor> input = std::make_shared<Tensor>();
    fullTensor(input, net, {1, 1, 2, 4}, 2);
    std::cout << "===NNAPI Execute===" << std::endl;
    ex.execute(input);
    std::cout << "===print result===" << std::endl;
    auto result = ex.result();
    result[0]->printData<float>();
    return 0;
}
