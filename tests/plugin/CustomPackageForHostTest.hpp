// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#include "mllm/mllm.hpp"
#include "mllm/backends/base/PluginInterface.hpp"

struct CustomOp1Options : public mllm::BaseOpOptions<CustomOp1Options> {
  int32_t data = 0;
};

class CustomOp1 final : public mllm::plugin::interface::CustomizedOp {
 public:
  explicit CustomOp1(const CustomOp1Options& options) : CustomizedOp("custom_op1"), options_(options) {}

  void load(const mllm::ParameterFile::ptr_t& ploader) override {};

  void trace(void* trace_context, const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) override {};

  void forward(const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) override {
    MLLM_INFO("Hello from custom op1, data: {}", options_.data);
  }

  void reshape(const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) override {}

  void setup(const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) override {}

 protected:
  CustomOp1Options options_;
};

class CustomOp1Factory final : public mllm::plugin::interface::CustomizedOpFactory<CustomOp1Options> {
 public:
  inline std::shared_ptr<mllm::BaseOp> createOpImpl(const CustomOp1Options& cargo) override {
    auto p = std::make_shared<CustomOp1>(cargo);
    p->setOpType(opType());
    return p;
  }
};

struct CustomOp2Options : public mllm::BaseOpOptions<CustomOp2Options> {
  int32_t data = 0;
};

class CustomOp2 final : public mllm::plugin::interface::CustomizedOp {
 public:
  explicit CustomOp2(const CustomOp2Options& options) : CustomizedOp("custom_op2"), options_(options) {}

  void load(const mllm::ParameterFile::ptr_t& ploader) override {};

  void trace(void* trace_context, const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) override {};

  void forward(const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) override {
    MLLM_INFO("Hello from custom op2, data: {}", options_.data);
  }

  void reshape(const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) override {}

  void setup(const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) override {}

 protected:
  CustomOp2Options options_;
};

class CustomOp2Factory final : public mllm::plugin::interface::CustomizedOpFactory<CustomOp2Options> {
 public:
  inline std::shared_ptr<mllm::BaseOp> createOpImpl(const CustomOp2Options& cargo) override {
    auto p = std::make_shared<CustomOp2>(cargo);
    p->setOpType(opType());
    return p;
  }
};
