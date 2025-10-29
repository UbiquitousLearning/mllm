// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include "mllm/core/BaseOp.hpp"

#define MLLM_PLUGIN_OP_PACKAGE_DESCRIPTOR_VERSION 1
#define MLLM_PLUGIN_OP_PACKAGE_NAME_LEN 256
#define MLLM_PLUGIN_OP_PACKAGE_DESCRIPTOR_LEN 256

#define MLLM_PLUGIN_OP_INTERFACE_DEFINE_BEGIN \
  extern "C" {                                \
  void* opPackageDescriptor();
#define MLLM_PLUGIN_OP_INTERFACE_DEFINE_END }

namespace mllm::plugin::interface {

class CustomizedOp : public BaseOp {
  std::string op_type_name_;

 public:
  explicit CustomizedOp(const std::string& typeName) : BaseOp(OpTypes::kDynamicOp_Start) { op_type_name_ = typeName; }

  [[nodiscard]] std::string getCustomOpTypeName() const { return op_type_name_; }
};

template<typename CargoT>
class CustomizedOpFactory : protected TypedOpFactory<OpTypes::kDynamicOp_Start, CargoT> {
  inline std::shared_ptr<BaseOp> createOpImpl(const CargoT& cargo) override {
    MLLM_ERROR_EXIT(ExitCode::kCoreError, "CustomizedOpFactory::createOpImpl is not implemented");
    return nullptr;
  }
};

}  // namespace mllm::plugin::interface

extern "C" {

typedef void* (*OpFactoryCreateFunc)();    // NOLINT
typedef void (*OpFactoryFreeFunc)(void*);  // NOLINT

struct PluginOpPackageDescriptor {
  int32_t version = MLLM_PLUGIN_OP_PACKAGE_DESCRIPTOR_VERSION;
  char name[MLLM_PLUGIN_OP_PACKAGE_NAME_LEN];

  int32_t device_type;
  int32_t op_factories_count = 0;
  char op_factories_names[MLLM_PLUGIN_OP_PACKAGE_DESCRIPTOR_LEN][MLLM_PLUGIN_OP_PACKAGE_NAME_LEN];
  OpFactoryCreateFunc op_factory_create_funcs[MLLM_PLUGIN_OP_PACKAGE_DESCRIPTOR_LEN];
  OpFactoryFreeFunc op_factory_free_funcs[MLLM_PLUGIN_OP_PACKAGE_DESCRIPTOR_LEN];
};
}
