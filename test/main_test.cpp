#include <iostream>
#include "Net.hpp"
#include "Executor.hpp"
#include "NetParameter.hpp"
#include "express/Express.hpp"

using namespace mllm;

void Display(NetParameter *net) {
    std::cout << "===NetParameter===" << std::endl;
    for (auto op : net->net_ops) {
        std::cout << "===NetOP===" << std::endl;
        std::cout << "op->name:" << op->name << std::endl;
        std::cout << "op->type:" << op->type << std::endl;
        std::cout << "op input" << op->in.size() << std::endl;
        for (auto input : op->in) {
            std::cout << "==Input==\ninput.name:" << input->name << std::endl;
            if (input->in) {
                std::cout << "input op:" << input->in->name << std::endl;
            }
            std::cout << "input in subgraph:" << (input->subgraph == net) << std::endl;
            std::cout << std::endl;
        }
        std::cout << "op output" << op->out.size() << std::endl;
        for (auto output : op->out) {
            std::cout << "output.name:" << output->name << std::endl;
            std::cout << "output op:" << output->out.size() << std::endl;
            if (output->out.size() > 0) {
                std::cout << "output op:" << output->out[0]->name << std::endl;
            }
        }
        std::cout << std::endl;
    }
}
void Display(Context *c) {
    for (auto sub : c->sub_param_) {
        Display(sub);
    }
}

int main() {
    Context *c = new Context();
    auto x = _Input(c, {1, 3, 3, 3});
    x = _Softmax(c, {x}, -1);
    auto z = _SiLU(c, {x});
    Subgraph_begin(c);
    auto y = _SiLU(c, {x});
    x = _Matmul(c, {z, y});
    x = _Softmax(c, {x}, -1);

    // Display(c);

    // auto x = _Input({1, 3, 3, 3});
    // auto y = _SiLU({x});
    // x = _MatMul({x, y});
    // x = _Scale({x});
    // x = _SoftMax({x}, -1);

    // // 输出连接的 EOP
    // NetParameter net_param;
    // createNetParem(x, net_param);

    std::vector<NetParameter> net_params;

    for (auto ptr : c->sub_param_) {
        net_params.push_back(*ptr); // 解引用并加入到新的 vector 中
    }
    BackendConfig bn;

    Net net(net_params, bn);
    net.Convert();
    // net.Run();

    Executor ex(&net);
    ex.Execute();
    return 0;
}
