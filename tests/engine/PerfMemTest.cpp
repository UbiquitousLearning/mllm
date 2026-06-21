#include <string>

#include "mllm/mllm.hpp"

using namespace mllm;  // NOLINT

class FooNet final : public nn::Module {
  nn::Linear linear_0;
  nn::Linear linear_1;
  nn::Linear linear_2;
  nn::Linear linear_3;

 public:
  explicit FooNet(const std::string& name) : nn::Module(name) {
    linear_0 = reg<nn::Linear>("linear_0", /*in_channels*/ 1024, /*out_channels*/ 2048);
    linear_1 = reg<nn::Linear>("linear_1", /*in_channels*/ 1024, /*out_channels*/ 2048);
    linear_2 = reg<nn::Linear>("linear_2", /*in_channels*/ 1024, /*out_channels*/ 2048);
    linear_3 = reg<nn::Linear>("linear_3", /*in_channels*/ 1024, /*out_channels*/ 2048);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    return {
        linear_0(inputs[0]),
        linear_1(inputs[0]),
        linear_2(inputs[0]),
        linear_3(inputs[0]),
    };
  }
};

int main() {
  mllm::initializeContext();
  {
    auto net = FooNet("foo_net");

    // Make some fake weights
    auto params = ParameterFile::create();
    for (int i = 0; i < 4; ++i) {
      auto name = "foo_net.linear_" + std::to_string(i);
      auto w = Tensor::empty({2048, 1024}).setMemType(kParamsNormal).setName(name + ".weight").alloc();
      auto b = Tensor::empty({2048}).setMemType(kParamsNormal).setName(name + ".bias").alloc();
      params->push(w.name(), w);
      params->push(b.name(), b);
    }
    net.load(params);
    auto o = net(Tensor::empty({1, 12, 1024, 1024}, kFloat32).alloc());
  }
  mllm::memoryReport();
}
