// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/ascend/ops/AscendLinearOp.hpp"

#include <acl/acl.h>
#include <atb/atb_infer.h>
#include <atb/types.h>
#include <atb/utils.h>
#include <atb/infer_op_params.h>
#include <cstdlib>

#include "mllm/utils/Common.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/backends/ascend/memory/AscendMemoryManager.hpp"
#include "mllm/backends/ascend/AscendCommon.hpp"

namespace mllm::ascend {

namespace {
bool shouldDebugLinearStats() {
  const char* debug = std::getenv("MLLM_DEBUG_LINEAR_STATS");
  return debug != nullptr && debug[0] == '1';
}

void printTensorMeta(const char* tag, const Tensor& t) {
  fmt::print("[LinearDbg] {} dtype={} device={} rank={} bytes={} ptr={}\n", tag, static_cast<int>(t.dtype()),
             static_cast<int>(t.device()), t.shape().size(), t.bytes(), fmt::ptr(t.ptr<void>()));
}

void printAtbTensorMeta(const char* tag, const atb::Tensor& t) {
  fmt::print("[LinearDbg] {} atb_dtype={} dimNum={} shape=[", tag, static_cast<int>(t.desc.dtype), t.desc.shape.dimNum);
  for (size_t i = 0; i < t.desc.shape.dimNum; ++i) {
    fmt::print("{}{}", i == 0 ? "" : ",", t.desc.shape.dims[i]);
  }
  fmt::print("] dataSize={} ptr={}\n", t.dataSize, fmt::ptr(t.deviceData));
}
}  // namespace

AscendLinearOp::AscendLinearOp(const aops::LinearOpOptions& options) : aops::LinearOp(options) {}

void AscendLinearOp::load(const ParameterFile::ptr_t& ploader) {
  // First call parent's load to get weight/bias from file (on CPU)
  aops::LinearOp::load(ploader);

  // Convert weight to FP16 and move to Ascend NPU
  if (!weight_.isNil()) {
    weight_ = convertTensorToAscendFP16(weight_);
  }

  // Convert bias to FP16 and move to Ascend NPU (if exists)
  // ATB Linear requires bias to be 2D [1, out_channels]
  if (options_.bias && !bias_.isNil()) {
    // Reshape bias from [out_channels] to [1, out_channels] for ATB
    if (bias_.shape().size() == 1) {
      bias_ = bias_.view({1, bias_.shape()[0]});
    }
    bias_ = convertTensorToAscendFP16(bias_);
  }
}

void AscendLinearOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  if (options().isRedirect()) {
    MLLM_RT_ASSERT(inputs.size() >= 1);
    const auto& input = inputs[0];
    const auto& weight = inputs.size() >= 2 ? inputs[1] : this->weight();
    auto out_shape = input.shape();
    out_shape[out_shape.size() - 1] = weight.shape()[0];  // out_channels
    outputs.emplace_back(Tensor::empty(out_shape, input.dtype(), input.device()));
    return;
  }
  aops::LinearOp::reshape(inputs, outputs);
}

void AscendLinearOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  BaseOp::setup(inputs, outputs);
}

void AscendLinearOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  MLLM_RT_ASSERT(inputs.size() >= 1 && inputs.size() <= 3);
  MLLM_RT_ASSERT_EQ(outputs.size(), 1);

  const Tensor* weight_ptr = nullptr;
  const Tensor* bias_ptr = nullptr;

  if (inputs.size() == 1) {
    weight_ptr = &weight();
    if (options().bias) { bias_ptr = &bias(); }
  } else if (inputs.size() == 2) {
    weight_ptr = &inputs[1];
  } else if (inputs.size() == 3) {
    weight_ptr = &inputs[1];
    bias_ptr = &inputs[2];
  }

  const auto& x = inputs[0];
  auto& y = outputs[0];

  // Validate that input tensors are FP16
  if (x.dtype() != MLLM_TYPE_F16) {
    MLLM_ERROR_EXIT(ExitCode::kAscendError,
                    "AscendLinearOp: Input tensor must be FP16, but got dtype={}",
                    static_cast<int>(x.dtype()));
  }
  if (weight_ptr->dtype() != MLLM_TYPE_F16) {
    MLLM_ERROR_EXIT(ExitCode::kAscendError,
                    "AscendLinearOp: Weight tensor must be FP16, but got dtype={}",
                    static_cast<int>(weight_ptr->dtype()));
  }
  if (bias_ptr != nullptr && bias_ptr->dtype() != MLLM_TYPE_F16) {
    MLLM_ERROR_EXIT(ExitCode::kAscendError,
                    "AscendLinearOp: Bias tensor must be FP16, but got dtype={}",
                    static_cast<int>(bias_ptr->dtype()));
  }

  // Validate bias dimensions: ATB Linear requires bias to be 2D [1, out_channels]
  if (bias_ptr != nullptr) {
    const auto& bias_shape = bias_ptr->shape();
    if (bias_shape.size() == 1) {
      MLLM_ERROR_EXIT(ExitCode::kAscendError,
                      "AscendLinearOp: Bias tensor must be 2D [1, out_channels], but got 1D shape with size={}. "
                      "Please reshape the bias tensor before passing to AscendLinearOp.",
                      bias_shape[0]);
    }
    if (bias_shape.size() != 2 || bias_shape[0] != 1) {
      MLLM_ERROR_EXIT(ExitCode::kAscendError,
                      "AscendLinearOp: Bias tensor must be 2D with shape [1, out_channels], but got shape=[{}, {}]",
                      bias_shape.size() >= 1 ? bias_shape[0] : 0,
                      bias_shape.size() >= 2 ? bias_shape[1] : 0);
    }
  }


  atb::infer::LinearParam linearParam;
  linearParam.transposeA = false;
  linearParam.transposeB = true;  // Set to true because weight is [out_channels, in_channels]
  linearParam.hasBias = (bias_ptr != nullptr);
  linearParam.outDataType = ACL_DT_UNDEFINED;
  linearParam.enAccum = false;
  linearParam.matmulType = atb::infer::LinearParam::MATMUL_UNDEFINED;
  linearParam.quantMode = atb::infer::LinearParam::QUANT_UNDEFINED;

  atb::Operation* op = nullptr;
  auto st = atb::CreateOperation(linearParam, &op);
  if (st != atb::NO_ERROR || op == nullptr) {
    MLLM_ERROR_EXIT(ExitCode::kAscendError, "ATB CreateOperation(Linear) failed, status={}", static_cast<int>(st));
  }

  atb::Context* atb_ctx = getGlobalAtbContext();

  atb::Tensor atb_x;
  atb::Tensor atb_weight;
  atb::Tensor atb_y;
  atb::Tensor atb_bias;

  fillAtbTensor(x, atb_x);
  fillAtbTensor(*weight_ptr, atb_weight);
  fillAtbTensor(y, atb_y);
  if (shouldDebugLinearStats()) {
    static int linear_debug_count = 0;
    if (linear_debug_count < 6) {
      fmt::print("[LinearDbg] call={} hasBias={} transposeB={}\n", linear_debug_count, bias_ptr != nullptr ? 1 : 0,
                 linearParam.transposeB ? 1 : 0);
      printTensorMeta("x", x);
      printTensorMeta("weight", *weight_ptr);
      printTensorMeta("y", y);
      printAtbTensorMeta("atb_x", atb_x);
      printAtbTensorMeta("atb_weight", atb_weight);
      printAtbTensorMeta("atb_y", atb_y);
      if (bias_ptr != nullptr) { printTensorMeta("bias", *bias_ptr); }
    }
    ++linear_debug_count;
  }

  atb::SVector<atb::Tensor> inTensors;
  atb::SVector<atb::Tensor> outTensors;
  inTensors.push_back(atb_x);
  inTensors.push_back(atb_weight);

  if (bias_ptr != nullptr) {
    fillAtbTensor(*bias_ptr, atb_bias);
    inTensors.push_back(atb_bias);
  }

  outTensors.push_back(atb_y);

  atb::VariantPack vp;
  vp.inTensors = inTensors;
  vp.outTensors = outTensors;

  uint64_t workspaceSize = 0;
  st = op->Setup(vp, workspaceSize, atb_ctx);
  if (st != atb::NO_ERROR) {
    MLLM_ERROR_EXIT(ExitCode::kAscendError, "ATB LinearOp Setup failed, status={}", static_cast<int>(st));
  }
  if (shouldDebugLinearStats()) {
    static int workspace_debug_count = 0;
    if (workspace_debug_count < 6) { fmt::print("[LinearDbg] workspaceSize={} bytes\n", workspaceSize); }
    ++workspace_debug_count;
  }

  void* workspace = nullptr;
  int workspace_block_id = -1;
  if (workspaceSize > 0) {
    auto& mem_mgr = getAscendMemoryManager();
    mem_mgr.allocateBlock(static_cast<uint32_t>(workspaceSize), workspace_block_id);
    mem_mgr.getBlockPtr(workspace_block_id, workspace);
  }

  {
    //ASCEND_TIME_SCOPE("AscendLinearOp::forward");
    st = op->Execute(vp, reinterpret_cast<uint8_t*>(workspace), workspaceSize, atb_ctx);
  }

  if (st != atb::NO_ERROR) {
    MLLM_ERROR_EXIT(ExitCode::kAscendError, "ATB LinearOp Execute failed, status={}", static_cast<int>(st));
  }

  syncGlobalAtbStream();

  if (workspace_block_id != -1) {
    auto& mem_mgr = getAscendMemoryManager();
    mem_mgr.freeBlock(workspace_block_id);
  }

  atb::DestroyOperation(op);
}

}  // namespace mllm::ascend
