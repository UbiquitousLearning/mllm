#include <iostream>
#include "Net.hpp"
#include "Executor.hpp"
#include "NetParameter.hpp"
#include "express/Express.hpp"

using namespace mllm;

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
