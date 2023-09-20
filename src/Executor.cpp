#include "Executor.hpp"
namespace mllm {
void Executor::Init() {
}

void Executor::Execute() {
    // for (auto kv : net_->subGraphFP()) {
    for (int i = 0; i < (int)net_->subGraphFP().size(); ++i) {
        string name = "G" + std::to_string(i);
        auto &g = net_->subGraphFP()[name];
        std::cout << name << std::endl;
        GraphSetup(g);
        auto result = GraphForward(g);
    }
}

} // namespace mllm
