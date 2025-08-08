// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <sstream>

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/OpTypes.hpp"
#include "mllm/nn/AbstractNnNode.hpp"
#include "mllm/core/ParameterFile.hpp"

namespace mllm::nn {

class LayerImpl : public AbstractNnNode {
 public:
  using ptr_t = std::shared_ptr<LayerImpl>;

  template<typename T>
  LayerImpl(OpTypes op_type, const T& option)
      : AbstractNnNode(AbstractNnNodeTypes::kLayer), op_type_(op_type), options_(option) {}

  void load(const ParameterFile::ptr_t& param_file);

  void to(DeviceTypes device_type);

  [[nodiscard]] OpTypes opType() const;

  BaseOpOptionsBase& refOptions();

  void __fmt_print(std::stringstream& ss);

  BaseOp::ptr_t getInstancedOp();

  void setInstancedOp(const BaseOp::ptr_t& op);

 private:
  OpTypes op_type_;
  BaseOpOptionsBase options_;
  BaseOp::ptr_t instanced_op_ = nullptr;
};

class Layer {
 public:
  explicit Layer(const LayerImpl::ptr_t& impl);

  template<typename T>
  Layer(OpTypes op_type, const T& cargo) {
    impl_ = std::make_shared<LayerImpl>(op_type, cargo);
  }

  [[nodiscard]] LayerImpl::ptr_t impl() const;

  template<typename... Args>
  Tensor operator()(Args&&... args) {
    auto inputs = std::vector<Tensor>{std::forward<decltype(args)>(args)...};
    return __main(inputs);
  }

  Tensor __main(const std::vector<Tensor>& inputs);

  [[nodiscard]] OpTypes opType() const;

  BaseOpOptionsBase& refOptions();

  Layer& to(DeviceTypes device_type);

  void __fmt_print(std::stringstream& ss);

 private:
  LayerImpl::ptr_t impl_ = nullptr;
};

}  // namespace mllm::nn
