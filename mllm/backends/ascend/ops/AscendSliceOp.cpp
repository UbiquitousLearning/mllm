// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/ascend/ops/AscendSliceOp.hpp"

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

AscendSliceOp::AscendSliceOp(const aops::SliceOpOptions& options) : aops::SliceOp(options) {}

void AscendSliceOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  BaseOp::setup(inputs, outputs);
}

void AscendSliceOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& input = inputs[0];
  auto shape = input.shape();
  auto slice_index = options().indices_;

  MLLM_RT_ASSERT_EQ(slice_index.size(), shape.size());

  std::vector<int> out_shape;
  for (size_t i = 0; i < shape.size(); ++i) {
    const auto& pair = slice_index[i];
    int32_t start = pair.start_;
    int32_t end = pair.end_;

    if (start == kAll) { start = 0; }
    if (end == kAll) { end = shape[i]; }

    if (start < 0) { start = start + shape[i]; }
    if (end < 0) { end = end + shape[i]; }

    start = std::max(0, std::min(start, static_cast<int>(shape[i])));
    end = std::max(0, std::min(end, static_cast<int>(shape[i])));
    
    int len = std::max(0, end - start);
    out_shape.push_back(len);
  }
  
  outputs.emplace_back(Tensor::empty(out_shape, input.dtype(), input.device()));
}

void AscendSliceOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  atb::infer::SliceParam param;
  auto& input = inputs[0];
  auto shape = input.shape();
  auto slice_index = options().indices_;
  
  for(size_t i=0; i<shape.size(); ++i) {
      int32_t start = slice_index[i].start_;
      int32_t end = slice_index[i].end_;
      int32_t dim_size = shape[i];

      if (start == kAll) start = 0;
      if (end == kAll) end = dim_size;

      if (start < 0) start += dim_size;
      if (end < 0) end += dim_size;

      start = std::max(0, std::min(start, dim_size));
      end = std::max(0, std::min(end, dim_size));

      param.offsets.push_back(start);
      param.size.push_back(end - start);
  }

  atb::Operation* op = nullptr;
  auto st = atb::CreateOperation(param, &op);
  if (st != atb::NO_ERROR || op == nullptr) {
    MLLM_ERROR_EXIT(ExitCode::kAscendError, "ATB CreateOperation(Slice) failed, status={}", static_cast<int>(st));
  }

  atb::Context* atb_ctx = getGlobalAtbContext();
  
  atb::SVector<atb::Tensor> inTensors;
  std::vector<atb::Tensor> atb_inputs(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    fillAtbTensor(inputs[i], atb_inputs[i]);
    inTensors.push_back(atb_inputs[i]);
  }

  atb::Tensor atb_output;
  fillAtbTensor(outputs[0], atb_output);
  atb::SVector<atb::Tensor> outTensors;
  outTensors.push_back(atb_output);

  atb::VariantPack vp;
  vp.inTensors = inTensors;
  vp.outTensors = outTensors;
  
  uint64_t workspaceSize = 0;
  st = op->Setup(vp, workspaceSize, atb_ctx);
  if (st != atb::NO_ERROR) {
    MLLM_ERROR_EXIT(ExitCode::kAscendError, "ATB SliceOp Setup failed, status={}", static_cast<int>(st));
  }
  
  void* workspace = nullptr;
  int workspace_block_id = -1;
  if (workspaceSize > 0) {
    auto& mem_mgr = getAscendMemoryManager();
    mem_mgr.allocateBlock(static_cast<uint32_t>(workspaceSize), workspace_block_id);
    mem_mgr.getBlockPtr(workspace_block_id, workspace);
  }

  {
    ASCEND_TIME_SCOPE("AscendSliceOp::forward");
    st = op->Execute(vp, reinterpret_cast<uint8_t*>(workspace), workspaceSize, atb_ctx);
  }
  
  if (st != atb::NO_ERROR) {
    MLLM_ERROR_EXIT(ExitCode::kAscendError, "ATB SliceOp Execute failed, status={}", static_cast<int>(st));
  }

  syncGlobalAtbStream();

  if (workspace_block_id != -1) {
    auto& mem_mgr = getAscendMemoryManager();
    mem_mgr.freeBlock(workspace_block_id);
  }

  atb::DestroyOperation(op);
}

}  // namespace mllm::ascend
