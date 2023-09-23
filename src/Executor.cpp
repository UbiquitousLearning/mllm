#include "Executor.hpp"
namespace mllm {
void Executor::init() {
}

void Executor::execute() {
    // for (auto kv : net_->subGraph()) {
    for (int i = 0; i < (int)net_->subGraph().size(); ++i) {
        string name = "G" + std::to_string(i);
        auto &g = net_->subGraph()[name];
        std::cout << name << std::endl;
        graphSetUp(g);
        auto result = graphForward(g);
    }
}

} // namespace mllm
