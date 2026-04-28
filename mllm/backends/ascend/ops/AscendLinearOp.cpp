// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/ascend/ops/AscendLinearOp.hpp"

#include <acl/acl.h>
#include <atb/atb_infer.h>
#include <atb/types.h>
#include <atb/utils.h>
#include <atb/infer_op_params.h>
#include <utility>
#include <vector>
#include <string>

#include "mllm/utils/Common.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/backends/ascend/memory/AscendMemoryManager.hpp"
#include "mllm/backends/ascend/AscendCommon.hpp"
#include "mllm/backends/ascend/ops/AscendLinearDynamicW8A8.hpp"
#include "mllm/backends/ascend/ops/AscendLinearQuant.hpp"

namespace mllm::ascend {

AscendLinearOp::AscendLinearOp(const aops::LinearOpOptions& options) : aops::LinearOp(options) {}

void AscendLinearOp::load(const ParameterFile::ptr_t& ploader) {
  // Guard: during LayerImpl::to() the temp ploader may be empty (no weight yet).
  // In that case, skip — model.load() will call us again with the real model file.
  if (!ploader->has(getName() + ".weight")) { return; }

  scale_x_ = 0.0f;
  scale_w_ = Tensor::nil();
  scale_x_tensor_ = Tensor::nil();
  deq_scale_npu_ = Tensor::nil();
  deq_scale_w_npu_ = Tensor::nil();
  bias_int32_npu_ = Tensor::nil();

  // Pull weight_ (and bias_ if any) onto CPU via parent implementation.
  aops::LinearOp::load(ploader);

  // The conversion script stores scales under the weight tensor name:
  //   {getName()}.weight          INT8  [N, K]
  //   {getName()}.weight.scale    FP32  [N]    per-channel weight scale
  //   {getName()}.weight.scale_x  FP32  [1]    per-tensor activation scale
  const std::string weight_key  = getName() + ".weight";
  const std::string scale_w_key = weight_key + ".scale";
  const std::string scale_x_key = weight_key + ".scale_x";

  if (weight_.dtype() == kInt8 && ploader->has(scale_x_key)) {
    // ----------------------------------------------------------------
    // W8A8 path (INT8 weight + per-channel scale + per-tensor scale_x)
    // ----------------------------------------------------------------
    if (!ploader->has(scale_w_key)) {
      MLLM_ERROR_EXIT(ExitCode::kAscendError,
                      "AscendLinearOp W8A8 load: layer {} has {} but no {}",
                      getName(),
                      scale_x_key,
                      scale_w_key);
    }

    auto artifacts = prepareLinearW8A8Artifacts(
        getName(), options_.out_channels, ploader->pull(scale_w_key), ploader->pull(scale_x_key));
    scale_x_ = artifacts.scale_x;
    scale_w_ = std::move(artifacts.scale_w_cpu);
    scale_x_tensor_ = std::move(artifacts.scale_x_cpu);
    deq_scale_npu_ = std::move(artifacts.deq_scale_npu);
    deq_scale_w_npu_ = std::move(artifacts.deq_scale_w_npu);
    bias_int32_npu_ = std::move(artifacts.bias_int32_npu);

    // Upload INT8 weight as-is to NPU (no dtype conversion)
    weight_ = weight_.to(kAscend);

  } else if (weight_.dtype() == kInt8 && ploader->has(scale_w_key)) {
    // ----------------------------------------------------------------
    // W8A16 path (INT8 weight + per-channel scale, no scale_x)
    //
    // Baseline strategy: dequantize INT8 -> FP16 once here, then let the
    // FP16 forward path handle matmul. No memory-bandwidth savings yet; this
    // exists to validate that the converter output + load wiring is correct
    // independently of any activation quantization complexity.
    // ----------------------------------------------------------------
    const int out_channels = options_.out_channels;
    const int in_channels = options_.in_channels;
    MLLM_RT_ASSERT_EQ(weight_.shape().size(), 2);
    MLLM_RT_ASSERT_EQ(weight_.shape()[0], out_channels);
    MLLM_RT_ASSERT_EQ(weight_.shape()[1], in_channels);

    Tensor scale_w_raw = ploader->pull(scale_w_key);
    Tensor sw_cpu = (scale_w_raw.device() == kAscend) ? scale_w_raw.to(kCPU) : scale_w_raw;
    MLLM_RT_ASSERT_EQ(sw_cpu.dtype(), kFloat32);
    MLLM_RT_ASSERT_EQ(sw_cpu.numel(), static_cast<size_t>(out_channels));

    const int8_t* w_int8 = weight_.ptr<int8_t>();
    const float*  sw     = sw_cpu.ptr<float>();

    Tensor w_fp16_cpu = Tensor::empty({out_channels, in_channels}, kFloat16, kCPU);
    w_fp16_cpu.alloc();
    auto* w_fp16 = w_fp16_cpu.ptr<half_float::half>();
    for (int n = 0; n < out_channels; ++n) {
      const float scale = sw[n];
      for (int k = 0; k < in_channels; ++k) {
        w_fp16[n * in_channels + k] = half_float::half(static_cast<float>(w_int8[n * in_channels + k]) * scale);
      }
    }

    weight_ = convertTensorToAscendFP16(w_fp16_cpu);

    if (options_.bias && !bias_.isNil()) {
      if (bias_.shape().size() == 1) { bias_ = bias_.view({1, bias_.shape()[0]}); }
      bias_ = convertTensorToAscendFP16(bias_);
    }
    // Intentionally do NOT retain scale_w_ — the weight is now FP16, so
    // getParams() returning a {FP16 weight, FP32 scale} pair would be
    // inconsistent. LayerImpl::to() will see a pure FP16 Linear.

  } else {
    // ----------------------------------------------------------------
    // FP16 path (original behaviour)
    // ----------------------------------------------------------------
    // Guard: an INT8 weight reaching this branch means the converter wrote
    // INT8 bytes but no scale metadata — reinterpreting them as FP16 would
    // silently produce garbage output. Fail loudly instead.
    if (weight_.dtype() == kInt8) {
      MLLM_ERROR_EXIT(ExitCode::kAscendError,
                      "AscendLinearOp::load: layer {} has INT8 weight but no {} "
                      "nor {} in the model file. Check the converter output.",
                      getName(), scale_w_key, scale_x_key);
    }
    if (!weight_.isNil()) {
      weight_ = convertTensorToAscendFP16(weight_);
    }

    // ATB Linear requires bias to be 2D [1, out_channels]
    if (options_.bias && !bias_.isNil()) {
      if (bias_.shape().size() == 1) {
        bias_ = bias_.view({1, bias_.shape()[0]});
      }
      bias_ = convertTensorToAscendFP16(bias_);
    }
  }
}

ParameterFile::ptr_t AscendLinearOp::getParams() {
  // Start from the base class (returns weight_ and optionally bias_).
  auto p = aops::LinearOp::getParams();
  // Append W8A8 scale tensors so LayerImpl::to() can pass them back to load().
  if (!scale_w_.isNil()) {
    const std::string weight_key = getName() + ".weight";
    p->push(weight_key + ".scale",   scale_w_);
    p->push(weight_key + ".scale_x", scale_x_tensor_);
  }
  return p;
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

  if (weight_ptr->dtype() == kInt8) {
    runLinearDynamicW8A8Eager(getName(), x, *weight_ptr, bias_int32_npu_, deq_scale_w_npu_, y);
  } else {
    // ----------------------------------------------------------------
    // FP16 path (original behaviour)
    // ----------------------------------------------------------------
    if (x.dtype() != kFloat16) {
      MLLM_ERROR_EXIT(ExitCode::kAscendError,
                      "AscendLinearOp: Input tensor must be FP16, but got dtype={}",
                      static_cast<int>(x.dtype()));
    }
    if (weight_ptr->dtype() != kFloat16) {
      MLLM_ERROR_EXIT(ExitCode::kAscendError,
                      "AscendLinearOp: Weight tensor must be FP16, but got dtype={}",
                      static_cast<int>(weight_ptr->dtype()));
    }
    if (bias_ptr != nullptr && bias_ptr->dtype() != kFloat16) {
      MLLM_ERROR_EXIT(ExitCode::kAscendError,
                      "AscendLinearOp: Bias tensor must be FP16, but got dtype={}",
                      static_cast<int>(bias_ptr->dtype()));
    }
    if (bias_ptr != nullptr) {
      const auto& bias_shape = bias_ptr->shape();
      if (bias_shape.size() != 2 || bias_shape[0] != 1) {
        MLLM_ERROR_EXIT(ExitCode::kAscendError,
                        "AscendLinearOp: Bias must be [1, out_channels], got shape=[{}, {}]",
                        bias_shape.size() >= 1 ? bias_shape[0] : 0,
                        bias_shape.size() >= 2 ? bias_shape[1] : 0);
      }
    }

    atb::infer::LinearParam lp;
    lp.transposeA  = false;
    lp.transposeB  = true;
    lp.hasBias     = (bias_ptr != nullptr);
    lp.outDataType = ACL_DT_UNDEFINED;
    lp.enAccum     = false;
    lp.matmulType  = atb::infer::LinearParam::MATMUL_UNDEFINED;
    lp.quantMode   = atb::infer::LinearParam::QUANT_UNDEFINED;

    atb::Operation* op = nullptr;
    auto st = atb::CreateOperation(lp, &op);
    if (st != atb::NO_ERROR || op == nullptr) {
      MLLM_ERROR_EXIT(ExitCode::kAscendError, "ATB CreateOperation(Linear FP16) failed, status={}", static_cast<int>(st));
    }

    auto* atb_ctx = getGlobalAtbContext();
    atb::Tensor atb_x, atb_w, atb_y, atb_bias;
    fillAtbTensor(x,           atb_x);
    fillAtbTensor(*weight_ptr, atb_w);
    fillAtbTensor(y,           atb_y);

    atb::SVector<atb::Tensor> inTensors, outTensors;
    inTensors.push_back(atb_x);
    inTensors.push_back(atb_w);
    if (bias_ptr != nullptr) {
      fillAtbTensor(*bias_ptr, atb_bias);
      inTensors.push_back(atb_bias);
    }
    outTensors.push_back(atb_y);

    atb::VariantPack vp;
    vp.inTensors  = inTensors;
    vp.outTensors = outTensors;

    uint64_t ws_size = 0;
    st = op->Setup(vp, ws_size, atb_ctx);
    if (st != atb::NO_ERROR) {
      MLLM_ERROR_EXIT(ExitCode::kAscendError, "ATB Linear FP16 Setup failed, status={}", static_cast<int>(st));
    }

    void* workspace = nullptr;
    int   ws_bid    = -1;
    if (ws_size > 0) {
      auto& mem_mgr = getAscendMemoryManager();
      mem_mgr.allocateBlock(static_cast<uint32_t>(ws_size), ws_bid);
      mem_mgr.getBlockPtr(ws_bid, workspace);
    }
    st = op->Execute(vp, reinterpret_cast<uint8_t*>(workspace), ws_size, atb_ctx);
    if (st != atb::NO_ERROR) {
      MLLM_ERROR_EXIT(ExitCode::kAscendError, "ATB Linear FP16 Execute failed, status={}", static_cast<int>(st));
    }
    syncGlobalAtbStream();

    if (ws_bid != -1) getAscendMemoryManager().freeBlock(ws_bid);
    atb::DestroyOperation(op);
  }
}

}  // namespace mllm::ascend
