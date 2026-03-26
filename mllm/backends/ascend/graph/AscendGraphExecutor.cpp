// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "AscendGraphExecutor.hpp"
#include "mllm/backends/ascend/AscendCommon.hpp"
#include "mllm/utils/Common.hpp"
#include <acl/acl.h>

namespace mllm::ascend {

AscendGraphExecutor::AscendGraphExecutor(atb::Operation* graph_op, atb::Context* context)
    : graph_op_(graph_op),
      context_(context),
      workspace_(nullptr),
      workspace_size_(0) {

  if (graph_op_ == nullptr) {
    MLLM_ERROR_EXIT(ExitCode::kAscendError, "Graph operation is null");
  }

  if (context_ == nullptr) {
    MLLM_ERROR_EXIT(ExitCode::kAscendError, "ATB context is null");
  }
}

AscendGraphExecutor::~AscendGraphExecutor() {
  // Free workspace
  if (workspace_ != nullptr) {
    aclrtFree(workspace_);
    workspace_ = nullptr;
  }

  // Destroy graph operation
  if (graph_op_ != nullptr) {
    atb::DestroyOperation(graph_op_);
    graph_op_ = nullptr;
  }
}

void AscendGraphExecutor::execute(const std::vector<Tensor>& inputs,
                                   std::vector<Tensor>& outputs) {
  // 1. Build VariantPack
  atb::VariantPack variantPack;
  variantPack.inTensors.resize(inputs.size());
  variantPack.outTensors.resize(outputs.size());

  // Fill input tensors
  for (size_t i = 0; i < inputs.size(); ++i) {
    fillAtbTensor(inputs[i], variantPack.inTensors[i]);
  }

  // Fill output tensors
  for (size_t i = 0; i < outputs.size(); ++i) {
    fillAtbTensor(outputs[i], variantPack.outTensors[i]);
  }

  // 2. Setup: Compute required workspace size
  uint64_t required_workspace = 0;
  auto ret = graph_op_->Setup(variantPack, required_workspace, context_);
  if (ret != atb::NO_ERROR) {
    MLLM_ERROR_EXIT(ExitCode::kAscendError,
                    "Graph Setup failed, status={}", static_cast<int>(ret));
  }

  // 3. Allocate or resize workspace if needed
  if (required_workspace > workspace_size_) {
    // Free old workspace
    if (workspace_ != nullptr) {
      aclrtFree(workspace_);
      workspace_ = nullptr;
    }

    // Allocate new workspace
    if (required_workspace > 0) {
      auto acl_ret = aclrtMalloc(&workspace_, required_workspace,
                                 ACL_MEM_MALLOC_HUGE_FIRST);
      if (acl_ret != ACL_SUCCESS) {
        MLLM_ERROR_EXIT(ExitCode::kAscendError,
                        "Failed to allocate workspace of {} bytes, acl_ret={}",
                        required_workspace, static_cast<int>(acl_ret));
      }
      workspace_size_ = required_workspace;
    }
  }

  // 4. Execute graph
  ret = graph_op_->Execute(variantPack,
                           reinterpret_cast<uint8_t*>(workspace_),
                           workspace_size_,
                           context_);
  if (ret != atb::NO_ERROR) {
    MLLM_ERROR_EXIT(ExitCode::kAscendError,
                    "Graph Execute failed, status={}", static_cast<int>(ret));
  }

  // 5. Synchronize stream to ensure execution completes
  syncGlobalAtbStream();
}

}  // namespace mllm::ascend
