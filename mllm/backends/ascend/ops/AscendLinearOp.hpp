// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/LinearOp.hpp"
#include "mllm/core/OpTypes.hpp"

namespace mllm::ascend {

class AscendLinearOp final : public aops::LinearOp {
 public:
  explicit AscendLinearOp(const aops::LinearOpOptions& options);

  void load(const ParameterFile::ptr_t& ploader) override;
  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
  ParameterFile::ptr_t getParams() override;

  // Static W8A8 graph artifacts, valid after load() when isW8A8() is true.
  // Eager dynamic W8A8 is an experimental accuracy/debug fallback and must be
  // explicitly enabled in AscendLinearOp::forward().
  [[nodiscard]] float scaleX() const { return scale_x_; }
  [[nodiscard]] const Tensor& biasInt32Npu() const { return bias_int32_npu_; }
  [[nodiscard]] const Tensor& deqScaleNpu() const { return deq_scale_npu_; }
  [[nodiscard]] const Tensor& deqScaleWNpu() const { return deq_scale_w_npu_; }
  [[nodiscard]] bool isW8A8() const { return !deq_scale_npu_.isNil(); }

 private:
  // W8A8 quantization artifacts — populated in load() when weight dtype is INT8.
  float scale_x_ = 0.0f;        // per-tensor activation scale scalar
  Tensor scale_w_;               // FP32 [N] on CPU — retained so getParams() can round-trip it
  Tensor scale_x_tensor_;        // FP32 [1] on CPU — retained so getParams() can round-trip it
  Tensor deq_scale_npu_;         // uint64 [N] on NPU: aclnnTransQuantParam(scale_x * scale_w)
  Tensor bias_int32_npu_;        // int32 [1, N] on NPU: mandatory zero bias for ATB W8A8 path

  // deq_scale with only scale_w (no scale_x baked in), used by the opt-in
  // dynamic eager W8A8 debug path.
  Tensor deq_scale_w_npu_;
};

class AscendLinearOpFactory final : public TypedOpFactory<OpTypes::kLinear, aops::LinearOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::LinearOpOptions& options) override {
    return std::make_shared<AscendLinearOp>(options);
  }
};

}  // namespace mllm::ascend
