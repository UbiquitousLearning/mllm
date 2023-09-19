#include "Executor.hpp"
namespace mllm {
void Executor::Init() {
}

void Executor::Execute() {
    for (auto kv : net_->subGraphFP()) {
        std::cout << kv.first << std::endl;
        GraphSetup(kv.second);
        auto result = GraphForward(kv.second);
    }
}

} // namespace mllm
