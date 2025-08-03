// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include <type_traits>

#include "mllm/nn/AbstractNnNode.hpp"
#include "mllm/nn/Layer.hpp"
#include "mllm/core/ParameterFile.hpp"
#include "mllm/engine/Context.hpp"

namespace mllm::nn {

class ModuleImpl : public AbstractNnNode {
 public:
  using ptr_t = std::shared_ptr<ModuleImpl>;

  ModuleImpl();

  void load(const ParameterFile::ptr_t& param_file);

  [[nodiscard]] ParameterFile::ptr_t params() const;

  void to(DeviceTypes device_type);

  void __fmt_print(std::stringstream& ss);

 private:
  ParameterFile::ptr_t param_file_;
};

template<typename T>
class ModuleLists;

template<typename T>
class ModuleListsSuffix;

class Module {
 public:
  explicit Module(const std::string& name);

  [[nodiscard]] ModuleImpl::ptr_t impl() const;

  void to(DeviceTypes device_type);

  /**
   * @brief Register a module/layer into this module
   *
   * @tparam T
   * @tparam Args
   * @param name
   * @param args
   * @return auto
   */
  template<typename T, typename... Args>
  auto reg(const std::string& name, Args&&... args) {
    // Register a module
    if constexpr (std::is_base_of_v<Module, T>) {
      auto ret = T(impl_->getAbsoluteName() + "." + name, std::forward<Args>(args)...);
      impl_->regChildNode(ret.impl());
      return ret;
    }

    // Register to thisThread table
    if constexpr (std::is_base_of_v<Layer, T>) {
      auto ret = T(std::forward<Args>(args)...);
      impl_->regChildNode(ret.impl());
      ret.impl()->setAbsoluteName(impl_->getAbsoluteName() + "." + name);

      auto& ctx = Context::instance();

      // Create Op
      auto _op = ctx.getBackend(ret.impl()->getDevice())->createOp(ret.opType(), ret.refOptions());
      _op->setName(ret.impl()->getAbsoluteName());

      // Register Op
      ret.impl()->setInstancedOp(_op);
      return ret;
    }
  }

  /**
   * @brief This function is used to register a node to the module.
   *
   * @note This function is for python API binding specially. Do not use this function in C++ code.
   *
   * @param name
   * @param node
   */
  void reg_(const std::string& name, const AbstractNnNode::ptr_t& node);

  template<typename... Args>
  std::vector<Tensor> operator()(Args&&... args) {
    std::vector<Tensor> inputs = {std::forward<Args>(args)...};
    return __main(inputs);
  }

  void load(const ParameterFile::ptr_t& param_file);

  virtual std::vector<Tensor> forward(const std::vector<Tensor>& inputs) = 0;

  void __fmt_print(std::stringstream& ss) const;

  std::vector<Tensor> __main(const std::vector<Tensor>& inputs);

  std::vector<Tensor> __trace(const std::vector<Tensor>& inputs);

 private:
  ModuleImpl::ptr_t impl_ = nullptr;
};

template<typename T>
class ModuleList final : public Module {
  std::vector<T> layers_;

 public:
  ModuleList() = default;

  template<typename... Args>
  ModuleList(const std::string& name, int nums, Args&&... args) : Module(name) {
    for (int i = 0; i < nums; ++i) {
      layers_.emplace_back(reg<T>(/*name*/ std::to_string(i), /*args*/ std::forward<Args>(args)...));
    }
  };

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override {
    std::vector<Tensor> o = inputs;
    for (auto& layer : layers_) { o = layer.forward(o); }
    return o;
  }

  std::vector<T>& list() { return layers_; }
};

template<typename T>
class ModuleListSuffixed final : public Module {
  std::vector<T> layers_;

 public:
  ModuleListSuffixed() = default;

  template<typename... Args>
  ModuleListSuffixed(const std::string& name, int suffix_start, int nums, Args&&... args) : Module(name) {
    for (int i = 0; i < nums; ++i) {
      layers_.emplace_back(reg<T>(/*name*/ std::to_string(suffix_start + i), /*args*/ std::forward<Args>(args)...));
    }
  };

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override {
    std::vector<Tensor> o = inputs;
    for (auto& layer : layers_) { o = layer.forward(o); }
    return o;
  }

  std::vector<T>& list() { return layers_; }
};

}  // namespace mllm::nn
