#include <iostream>
#include "Net.hpp"
#include "Executor.hpp"

#include "express/Express.hpp"

using namespace mllm;

int main() {
    auto x = _Input({1, 3, 3, 3});
    auto y = _SiLU({x});
    x = _MatMul({x, y});
    x = _Scale({x});
    x = _SoftMax({x}, -1);

    // 输出连接的 EOP
    NetParameter net_param;
    createNetParem(x, net_param);

    BackendConfig bn;

    Net net(net_param, bn);
    net.Convert();
    // net.Run();

    Executor ex(&net);
    ex.Execute();
    return 0;
}
