#include "mllm/mllm.hpp"

using namespace mllm;  // NOLINT

class FooNet final : public nn::Module {
  nn::Linear linear_;

 public:
  explicit FooNet(const std::string& name) : nn::Module(name) {
    linear_ = reg<nn::Linear>("linear", /*in_channels*/ 1024, /*out_channels*/ 1024);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override { return {linear_(inputs[0])}; }
};

int main() {
  mllm::initializeContext();
  auto net = FooNet("foo_net");
  print(net);
}
