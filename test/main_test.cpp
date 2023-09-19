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
void Display(Context *ctx) {
    for (auto sub : ctx->sub_param_) {
        Display(sub);
    }
}

int main() {
    // Tensor tensor_(1,3,5,5);
    // std::cout << "Init Tensor" << std::endl;
    // tensor_.Reshape(1,5,6,6);
    // std::cout<<"Shape: ["<<tensor_.num()<<", "<<tensor_.channels()<<", "<<tensor_.height()<<", "<<tensor_.width()<<"]"<<std::endl;
    // auto pTensor_ = tensor_.cpu_data();
    // std::cout<<pTensor_<<":data[0]:"<<pTensor_[0]<<std::endl;
    Context *ctx = new Context();
    auto x = _Input(ctx, {1, 3, 3, 3});
    x = _Softmax(ctx, {x}, -1);
    auto z = _SiLU(ctx, {x});
    Subgraph_begin(ctx);
    auto y = _SiLU(ctx, {x});
    x = _Matmul(ctx, {z, y});
    x = _Softmax(ctx, {x}, -1);

    Display(ctx);
    // 输出连接的 EOP
    // NetParameter netParam;
    // createNetParem(x, netParam);

    // NetParameter netParam;

    // 初始化 netParam 的成员变量
    // netParam.input_name = "input";
    // netParam.output_name = "output";

    // NetOp op1 = {OpType::Silu, {0}, {0},{"Input0"}, "silu1"};
    // NetOp op2 = {OpType::Add, {0}, {0}, {"Input0", "silu1"}, "add1"};
    // NetOp op3 = {OpType::Matmul, {0}, {0}, {"add1", "Input0"}, "matmul1"};

    // netParam.net_ops.push_back(op1);
    // netParam.net_ops.push_back(op2);
    // netParam.net_ops.push_back(op3);

    // BackendConfig bn;

    // Net net(netParam, bn);
    // net.Convert();
    // net.Run();

    // Executor ex(&net);
    // ex.Execute();
    return 0;
}
