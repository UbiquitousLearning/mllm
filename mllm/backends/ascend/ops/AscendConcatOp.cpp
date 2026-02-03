// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/ascend/ops/AscendConcatOp.hpp"

#include <iostream>
#include <acl/acl.h>
#include <atb/atb_infer.h>
#include <atb/types.h>
#include <atb/utils.h>
#include <atb/infer_op_params.h>

#include "mllm/utils/Common.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/backends/ascend/memory/AscendMemoryManager.hpp"
#include "mllm/backends/ascend/AscendCommon.hpp"

namespace mllm::ascend {

AscendConcatOp::AscendConcatOp(const aops::ConcatOpOptions& options) : aops::ConcatOp(options) {}

void AscendConcatOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  BaseOp::setup(inputs, outputs);
}

void AscendConcatOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  MLLM_RT_ASSERT(inputs.size() >= 1);
  MLLM_RT_ASSERT_EQ(outputs.size(), 1);

  if (inputs.size() == 1) {
    const size_t data_size = inputs[0].bytes();
    const void* src_data = inputs[0].ptr<void>();
    void* dst_data = outputs[0].ptr<void>();

    if (src_data != dst_data) {
      auto ret = aclrtMemcpy(dst_data, data_size, src_data, data_size, ACL_MEMCPY_DEVICE_TO_DEVICE);
      if (ret != ACL_SUCCESS) {
        MLLM_ACL_CHECK(ret);
      }
      syncGlobalAtbStream();
    }
    return;
  }

  int32_t concat_dim = options().dim;
  if (concat_dim < 0) {
    concat_dim += static_cast<int32_t>(inputs[0].rank());
  }

  auto run_concat = [&](const Tensor& left, const Tensor& right, Tensor& out) {
    atb::infer::ConcatParam param;
    param.concatDim = concat_dim;

    atb::Operation* op = nullptr;
    auto st = atb::CreateOperation(param, &op);
    if (st != atb::NO_ERROR || op == nullptr) {
      MLLM_ERROR_EXIT(ExitCode::kAscendError, "ATB CreateOperation(Concat) failed, status={}", static_cast<int>(st));
    }

    atb::Context* atb_ctx = getGlobalAtbContext();

    atb::SVector<atb::Tensor> inTensors;
    atb::Tensor atb_left;
    atb::Tensor atb_right;
    fillAtbTensor(left, atb_left);
    fillAtbTensor(right, atb_right);
    inTensors.push_back(atb_left);
    inTensors.push_back(atb_right);

    atb::Tensor atb_out;
    fillAtbTensor(out, atb_out);
    atb::SVector<atb::Tensor> outTensors;
    outTensors.push_back(atb_out);

    atb::VariantPack vp;
    vp.inTensors = inTensors;
    vp.outTensors = outTensors;

    uint64_t workspaceSize = 0;
    st = op->Setup(vp, workspaceSize, atb_ctx);
    if (st != atb::NO_ERROR) {
      MLLM_ERROR_EXIT(ExitCode::kAscendError, "ATB ConcatOp Setup failed, status={}", static_cast<int>(st));
    }

    void* workspace = nullptr;
    int workspace_block_id = -1;
    if (workspaceSize > 0) {
      auto& mem_mgr = getAscendMemoryManager();
      mem_mgr.allocateBlock(static_cast<uint32_t>(workspaceSize), workspace_block_id);
      mem_mgr.getBlockPtr(workspace_block_id, workspace);
    }

    {
      ASCEND_TIME_SCOPE("AscendConcatOp::forward");
      st = op->Execute(vp, reinterpret_cast<uint8_t*>(workspace), workspaceSize, atb_ctx);
    }

    if (st != atb::NO_ERROR) {
      MLLM_ERROR_EXIT(ExitCode::kAscendError, "ATB ConcatOp Execute failed, status={}", static_cast<int>(st));
    }

    syncGlobalAtbStream();

    if (workspace_block_id != -1) {
      auto& mem_mgr = getAscendMemoryManager();
      mem_mgr.freeBlock(workspace_block_id);
    }

    atb::DestroyOperation(op);
  };

  std::vector<int32_t> current_shape = inputs[0].shape();
  Tensor current = inputs[0];

  for (size_t i = 1; i < inputs.size(); ++i) {
    current_shape[concat_dim] += inputs[i].shape()[concat_dim];

    if (i == inputs.size() - 1) {
      run_concat(current, inputs[i], outputs[0]);
    } else {
      Tensor temp = Tensor::empty(current_shape, outputs[0].dtype(), outputs[0].device()).alloc();
      run_concat(current, inputs[i], temp);
      current = temp;
    }
  }
}

}  // namespace mllm::ascend
