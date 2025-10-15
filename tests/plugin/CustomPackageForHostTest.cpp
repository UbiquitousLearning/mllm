#include "CustomPackageForHostTest.hpp"

MLLM_PLUGIN_OP_INTERFACE_DEFINE_BEGIN
void* createCustomOp1Factory() { return new CustomOp1Factory(); };

void freeCustomOp1Factory(void* factory) { delete static_cast<CustomOp1Factory*>(factory); };

void* createCustomOp2Factory() { return new CustomOp2Factory(); };

void freeCustomOp2Factory(void* factory) { delete static_cast<CustomOp2Factory*>(factory); };

void* opPackageDescriptor() {
  auto package = new PluginOpPackageDescriptor{
      .version = MLLM_PLUGIN_OP_PACKAGE_DESCRIPTOR_VERSION,
      .name = "CustomPackageForHostTest",
      .device_type = 1,
      .op_factories_count = 2,
      .op_factories_names =
          {
              "custom_op1",
              "custom_op2",
          },
      .op_factory_create_funcs =
          {
              createCustomOp1Factory,
              createCustomOp2Factory,
          },
      .op_factory_free_funcs =
          {
              freeCustomOp1Factory,
              freeCustomOp2Factory,
          },
  };
  return package;
}
MLLM_PLUGIN_OP_INTERFACE_DEFINE_END
