// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include <type_traits>

#include "mllm/core/MappedFile.hpp"
#include "mllm/utils/AnyValue.hpp"
#include "mllm/nn/AbstractNnNode.hpp"
#include "mllm/nn/Layer.hpp"
#include "mllm/core/ParameterFile.hpp"
#include "mllm/engine/Context.hpp"
#include "mllm/utils/SymbolTable.hpp"
#include "mllm/utils/CompilerTraits.hpp"

namespace mllm::nn {

class ModuleImpl : public AbstractNnNode {
 public:
  using ptr_t = std::shared_ptr<ModuleImpl>;

  ModuleImpl();

  void load(const ParameterFile::ptr_t& param_file);

  [[nodiscard]] ParameterFile::ptr_t params(ModelFileVersion v);

  void to(DeviceTypes device_type);

  void __fmt_print(std::stringstream& ss);

  void registerBuffer(const std::string& name, const Tensor& tensor);

  Tensor getBuffer(const std::string& name);

 private:
  /// Buffer is tensors that will not shown in params. And will not be saved.
  SymbolTable<std::string, Tensor> buffer_;
  std::vector<MappedFile::ptr_t> resources_mapped_files_;
};

template<typename T>
class ModuleLists;

template<typename T>
class ModuleListsSuffix;

class Module {
 public:
  Module();

  explicit Module(const ModuleImpl::ptr_t& impl);

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
      auto ret =
          T((impl_->getAbsoluteName() == "" ? name : impl_->getAbsoluteName() + "." + name), std::forward<Args>(args)...);
      impl_->regChildNode(ret.impl());
      ret.impl()->setName(name);
      return ret;
    }

    // Register to thisThread table
    if constexpr (std::is_base_of_v<Layer, T>) {
      auto ret = T(std::forward<Args>(args)...);
      impl_->regChildNode(ret.impl());
      ret.impl()->setAbsoluteName((impl_->getAbsoluteName() == "" ? name : impl_->getAbsoluteName() + "." + name));
      ret.impl()->setName(name);

      auto& ctx = Context::instance();

      // Create Op
      auto _op = ctx.getBackend(ret.impl()->getDevice())->createOp(ret.opType(), ret.refOptions());
      _op->setName(ret.impl()->getAbsoluteName());

      // Register Op
      ret.impl()->setInstancedOp(_op);
      return ret;
    }
  }

  template<typename T, typename... Args>
  auto __reg_as_pointer(const std::string& name, Args&&... args) {
    // Register a module
    if constexpr (std::is_base_of_v<Module, T>) {
      auto ret = std::make_shared<T>((impl_->getAbsoluteName() == "" ? name : impl_->getAbsoluteName() + "." + name),
                                     std::forward<Args>(args)...);
      impl_->regChildNode(ret->impl());
      ret->impl()->setName(name);
      return ret;
    }

    // Register to thisThread table
    if constexpr (std::is_base_of_v<Layer, T>) {
      auto ret = std::make_shared<T>(std::forward<Args>(args)...);
      impl_->regChildNode(ret->impl());
      ret->impl()->setAbsoluteName((impl_->getAbsoluteName() == "" ? name : impl_->getAbsoluteName() + "." + name));
      ret->impl()->setName(name);

      auto& ctx = Context::instance();

      // Create Op
      auto _op = ctx.getBackend(ret->impl()->getDevice())->createOp(ret->opType(), ret->refOptions());
      _op->setName(ret->impl()->getAbsoluteName());

      // Register Op
      ret->impl()->setInstancedOp(_op);
      return ret;
    }
  }

  template<typename... Args>
  std::vector<Tensor> operator()(Args&&... args) {
    std::vector<Tensor> tensors;
    std::vector<AnyValue> others;

    (..., [&] {
      // The type must can be inference in compile time
      using CleanType = std::decay_t<decltype(args)>;
      if constexpr (std::is_convertible_v<CleanType, Tensor>) {
        tensors.push_back(std::forward<Args>(args));
      } else if constexpr (std::is_convertible_v<CleanType, AnyValue>) {
        others.push_back(std::forward<Args>(args));
      } else {
        static_assert(always_false<CleanType>::value, "Unsupported argument type!");
      }
    }());
    return __main(tensors, others);
  }

  void load(const ParameterFile::ptr_t& param_file);

  virtual std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args);

  void __fmt_print(std::stringstream& ss) const;

  std::vector<Tensor> __main(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args);

  // __send_graph_begin and __send_graph_end are used to anotate the begin and end of a module execution in trace mode.
  // During common execution, the module is called by submitting a TaskTypes::kExecuteModule task
  void __send_graph_begin(const std::vector<Tensor>& inputs);

  void __send_graph_end(const std::vector<Tensor>& inputs);

  std::vector<Tensor> __trace(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args);

  ParameterFile::ptr_t params(ModelFileVersion v);

  void registerBuffer(const std::string& name, const Tensor& tensor);

  Tensor getBuffer(const std::string& name);

  [[nodiscard]] inline std::string getModuleName() const { return impl_->getAbsoluteName(); }

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

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    std::vector<Tensor> o = inputs;
    for (auto& layer : layers_) { o = layer.forward(o, args); }
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

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    std::vector<Tensor> o = inputs;
    for (auto& layer : layers_) { o = layer.forward(o, args); }
    return o;
  }

  std::vector<T>& list() { return layers_; }
};

class Sequential : public Module {
 public:
  Sequential() = default;

  explicit Sequential(const std::string& name) : Module(name) {}

  template<typename T, typename... Args>
  Sequential& add(Args&&... args) {
    auto __IN_T = __reg_as_pointer<T>(std::to_string(ops_cnt_), std::forward<Args>(args)...);
    if constexpr (std::is_base_of_v<Module, T>) {
      ops_names_.emplace_back(std::to_string(ops_cnt_));
      module_holder_.emplace_back(__IN_T);
      op_mapping_[std::to_string(ops_cnt_)] = {AbstractNnNodeTypes::kModule, (void*)(__IN_T.get())};
      __IN_T->impl()->depthIncrease();
    } else if constexpr (std::is_base_of_v<Layer, T>) {
      ops_names_.emplace_back(std::to_string(ops_cnt_));
      layer_holder_.emplace_back(__IN_T);
      op_mapping_[std::to_string(ops_cnt_)] = {AbstractNnNodeTypes::kLayer, (void*)(__IN_T.get())};
      __IN_T->impl()->depthIncrease();
    }
    ops_cnt_++;
    return *this;
  }

  inline std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    std::vector<Tensor> inputs_ = inputs;
    for (auto& name : ops_names_) {
      auto payload = op_mapping_[name];
      switch (payload.first) {
        case AbstractNnNodeTypes::kLayer: {
          auto layer = (Layer*)payload.second;
          inputs_ = layer->__main(inputs_);
          break;
        }
        case AbstractNnNodeTypes::kModule: {
          auto op = (Module*)payload.second;
          inputs_ = op->__main(inputs_, args);
          break;
        }
      }
    }
    return inputs_;
  }

 private:
  int32_t ops_cnt_ = 0;
  std::vector<std::string> ops_names_;
  std::vector<std::shared_ptr<Layer>> layer_holder_;
  std::vector<std::shared_ptr<Module>> module_holder_;
  std::unordered_map<std::string, std::pair<AbstractNnNodeTypes, void*>> op_mapping_;
};

}  // namespace mllm::nn
