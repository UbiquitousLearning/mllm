#include "mllm/mllm.hpp"
#include "CustomPackageForHostTest.hpp"

MLLM_MAIN({
  mllm::loadOpPackage("./libCustomPackageForHostTest.so");
  std::vector<mllm::Tensor> inputs, outputs;
  auto op = mllm::Context::instance()
                .getBackend(mllm::kCPU)
                ->createOp((mllm::OpTypes)mllm::Context::instance().lookupCustomizedOpId(mllm::kCPU, "custom_op1"),
                           CustomOp1Options{.data = 42});
  op->forward(inputs, outputs);
})
