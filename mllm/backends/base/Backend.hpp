// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <memory>
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/DeviceTypes.hpp"
#include "mllm/utils/SymbolTable.hpp"
#include "mllm/backends/base/Allocator.hpp"

namespace mllm {

class Backend {
 public:
  using ptr_t = std::shared_ptr<Backend>;

  Backend(DeviceTypes device, const Allocator::ptr_t& allocator);

  BaseOp::ptr_t createOp(OpTypes op_type, const BaseOpOptionsBase& base_options);

  template<typename... Args>
  void regOpFactory() {
    (..., (_reg_one_op_factory<Args>()));
  }

  void regOpFactory(const std::shared_ptr<BaseOpFactory>& factory);

  [[nodiscard]] inline DeviceTypes device() const { return device_; }

  [[nodiscard]] inline Allocator::ptr_t allocator() const { return allocator_; }

  // used when backend is not CPU, indicate whether to hold OP weights on CPU
  [[nodiscard]] virtual bool isWeightOnDevice() { return true; }

 protected:
  template<typename T>
  void _reg_one_op_factory() {
    auto ptr = std::make_shared<T>();
    op_factories_.reg(ptr->opType(), ptr);
  }

  DeviceTypes device_ = kCPU;
  Allocator::ptr_t allocator_ = nullptr;
  SymbolTable<OpTypes, std::shared_ptr<BaseOpFactory>, OpTypesSymbolTableFormatter> op_factories_;
};

}  // namespace mllm
