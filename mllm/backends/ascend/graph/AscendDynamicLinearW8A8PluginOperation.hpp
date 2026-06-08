// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <string>

#include <atb/operation_infra.h>

namespace mllm::ascend {

// ATB plugin implementing W8A8 with DYNAMIC per-token activation quantization.
//
// Pipeline per Execute():
//   x_fp16 ──► aclnnAbs ──► aclnnMaxDim(dim=-1,keepdim=T) ──► MULS(1/127)
//          ──► scale_x[B,S,1]
//          ──► REALDIV(x, scale_x) ──► aclnnRound ──► aclnnClamp(-128,127)
//          ──► aclnnCast(FP16→INT8)
//          ──► ATB Linear W8A8 PER_CHANNEL(deq_scale_w only)
//          ──► ELEWISE_MUL(y_pre, scale_x) ──► y_fp16
//
// Inputs : [x_fp16, weight_int8, bias_i32, deq_scale_w]
//   NOTE: deq_scale_w = scale_w only (use AscendLinearOp::deqScaleWNpu()).
// Outputs: [y_fp16]
//
// ACL op executors (abs/maxDim/round/clamp/cast) are cached after first Setup
// and reused in Execute via pointer-update, avoiding repeated GetWorkspaceSize
// calls in the decode hot path.
//
// This graph plugin is kept as a spare/experimental path. It is not wired into
// the default Qwen Ascend graph; static calibrated W8A8 remains the production
// path.
class AscendDynamicLinearW8A8PluginOperation final : public atb::OperationInfra {
 public:
  explicit AscendDynamicLinearW8A8PluginOperation(std::string name_suffix = "");
  ~AscendDynamicLinearW8A8PluginOperation() override;

  std::string GetName() const override;
  atb::Status InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs,
                         atb::SVector<atb::TensorDesc>& outTensorDescs) const override;
  uint32_t GetInputNum() const override;
  uint32_t GetOutputNum() const override;
  atb::Status Setup(const atb::VariantPack& variantPack,
                    uint64_t& workspace_size,
                    atb::Context* context) override;
  atb::Status Execute(const atb::VariantPack& variantPack,
                      uint8_t* workspace,
                      uint64_t workspace_size,
                      atb::Context* context) override;

 private:
  std::string name_suffix_;

  // ATB operations — created once in constructor, destroyed in destructor.
  atb::Operation* muls_op_{nullptr};        // ELEWISE_MULS(1/127)
  atb::Operation* div_op_{nullptr};         // ELEWISE_REALDIV
  atb::Operation* linear_op_{nullptr};      // ATB Linear W8A8 PER_CHANNEL
  atb::Operation* mul_dequant_op_{nullptr}; // ELEWISE_MUL

  // Workspace layout cached by Setup(), consumed by Execute().
  //   buf_a reuse: abs_x (steps 1-2) → x_round (steps 5-6 input)
  //   buf_b reuse: x_scaled (step 4) → x_clamped (step 6 output / step 7 input)
  struct WorkspaceLayout {
    uint64_t buf_a_off{0},          buf_a_bytes{0};
    uint64_t max_abs_off{0},        max_abs_bytes{0};
    uint64_t idx_buf_off{0},        idx_buf_bytes{0};
    uint64_t scale_x_off{0},        scale_x_bytes{0};
    uint64_t buf_b_off{0},          buf_b_bytes{0};
    uint64_t x_int8_off{0},         x_int8_bytes{0};
    uint64_t y_pre_off{0},          y_pre_bytes{0};
    uint64_t abs_ws_off{0},         abs_ws_bytes{0};
    uint64_t maxdim_ws_off{0},      maxdim_ws_bytes{0};
    uint64_t round_ws_off{0},       round_ws_bytes{0};
    uint64_t clamp_ws_off{0},       clamp_ws_bytes{0};
    uint64_t cast_ws_off{0},        cast_ws_bytes{0};
    uint64_t muls_ws_off{0},        muls_ws_bytes{0};
    uint64_t div_ws_off{0},         div_ws_bytes{0};
    uint64_t linear_ws_off{0},      linear_ws_bytes{0};
    uint64_t mul_dequant_ws_off{0}, mul_dequant_ws_bytes{0};
    uint64_t total{0};
  } ws_layout_{};

  // Opaque ACL executor caches (concrete types defined in .cpp anonymous namespace).
  void* abs_cache_{nullptr};
  void* maxdim_cache_{nullptr};
  void* round_cache_{nullptr};
  void* clamp_cache_{nullptr};
  void* cast_cache_{nullptr};

  void buildAtbOps();
};

atb::Operation* createDynamicLinearW8A8PluginGraphOp(std::string name_suffix = "");

}  // namespace mllm::ascend
