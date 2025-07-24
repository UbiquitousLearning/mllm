/**
 * @file Layer.hpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-24
 *
 */
#pragma once

#include <memory>

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/OpTypes.hpp"
#include "mllm/nn/AbstractNnNode.hpp"
#include "mllm/core/ParameterFile.hpp"

#include "mllm/engine/Context.hpp"

namespace mllm::nn {

class LayerImpl : public AbstractNnNode {
 public:
  using ptr_t = std::shared_ptr<LayerImpl>;

  template<typename T>
  LayerImpl(OpTypes op_type, const T& option)
      : AbstractNnNode(AbstractNnNodeTypes::kLayer), op_type_(op_type), options_(option) {}

  ParameterFile::ptr_t refParams();

  void load(const ParameterFile::ptr_t& param_file);

  void to(DeviceTypes device_type);

  [[nodiscard]] OpTypes opType() const;

  BaseOpOptionsBase& refOptions();

 private:
  OpTypes op_type_;
  BaseOpOptionsBase options_;
  ParameterFile::ptr_t parameter_loader_;
};

class Layer {
 public:
  template<typename T>
  Layer(OpTypes op_type, const T& cargo) {
    impl_ = std::make_shared<LayerImpl>(op_type, cargo);
  }

  [[nodiscard]] LayerImpl::ptr_t impl() const;

  template<typename... Args>
  Tensor operator()(Args&&... args) {
    auto inputs = std::vector<Tensor>{std::forward<decltype(args)>(args)...};

    // TODO Dispatch
  }

  [[nodiscard]] OpTypes opType() const;

  BaseOpOptionsBase& refOptions();

  Layer& to(DeviceTypes device_type);

 private:
  LayerImpl::ptr_t impl_;
};

}  // namespace mllm::nn
