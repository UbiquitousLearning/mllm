#include "mllm/mllm.hpp"

using namespace mllm;  // NOLINT

class FooNet final : public nn::Module {
  nn::Linear linear_0;
  nn::Linear linear_1;
  nn::Linear linear_2;
  nn::Linear linear_3;
  nn::Sequential seq;

 public:
  explicit FooNet(const std::string& name) : nn::Module(name) {
    linear_0 = reg<nn::Linear>("linear_0", /*in_channels*/ 64, /*out_channels*/ 64);
    linear_1 = reg<nn::Linear>("linear_1", /*in_channels*/ 64, /*out_channels*/ 64);
    linear_2 = reg<nn::Linear>("linear_2", /*in_channels*/ 64, /*out_channels*/ 64);
    linear_3 = reg<nn::Linear>("linear_3", /*in_channels*/ 64, /*out_channels*/ 64);
    seq = reg<nn::Sequential>("activation")
              .add<nn::SiLU>()
              .add<nn::Linear>(/*in_channels*/ 64, /*out_channels*/ 64)
              .add<nn::SiLU>();
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    return seq(linear_3(linear_2(linear_1(linear_0(inputs[0])))));
  }
};

int main() {
  mllm::initializeContext();
  {
    auto net = FooNet("foo_net");
    auto params = ParameterFile::create();

    for (int i = 0; i < 4; ++i) {
      auto name = "foo_net.linear_" + std::to_string(i);
      auto w = Tensor::empty({64, 64}).setMemType(kParamsNormal).setName(name + ".weight").alloc();
      auto b = Tensor::empty({64}).setMemType(kParamsNormal).setName(name + ".bias").alloc();
      params->push(w.name(), w);
      params->push(b.name(), b);
    }
    auto w1 = Tensor::empty({64, 64}).setMemType(kParamsNormal).setName("foo_net.activation.1.weight").alloc();
    auto b1 = Tensor::empty({64}).setMemType(kParamsNormal).setName("foo_net.activation.1.bias").alloc();
    params->push(w1.name(), w1);
    params->push(b1.name(), b1);
    net.load(params);
    print(net);
    auto o = net(Tensor::empty({1, 12, 64, 64}, kFloat32).alloc());
    print(o[0].shape(), o[0].dtype(), o[0].device());
    mllm::memoryReport();
  }
  mllm::memoryReport();

  // Memory report can also be print using print(...)
  auto mm = Context::instance().memoryManager();
  print(mm);
}
