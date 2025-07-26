#include "mllm/engine/Context.hpp"
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

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override {
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
    print(net);
    auto o = net(Tensor::empty({1, 12, 1024, 1024}, kFloat32).alloc());
    print(o[0].shape(), o[0].dtype(), o[0].device());
    print(o[1].shape(), o[1].dtype(), o[1].device());
    print(o[2].shape(), o[2].dtype(), o[2].device());
    print(o[3].shape(), o[3].dtype(), o[3].device());
    mllm::memoryReport();
  }
  mllm::memoryReport();

  // Memory report can also be print using print(...)
  auto mm = Context::instance().memoryManager();
  print(mm);
}
