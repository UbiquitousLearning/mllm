// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <string>

#include <atb/operation_infra.h>

namespace mllm::ascend {

// ATB plugin that implements the W8A8 activation-quantization + linear path:
//
//   x_fp16  ──► ELEWISE_MULS(*inv_scale) ──► aclnnRound ──► aclnnClamp(-128,127)
//           ──► [quant_subgraph, FP16 intermediate, output = x_clamped_fp16]
//           ──► aclnnCast(FP16→INT8)
//           ──► ATB Linear W8A8(PER_CHANNEL) ──► y_fp16
//
// Inputs : [x_fp16, weight_int8, bias_i32, deq_scale]
// Outputs: [y_fp16]
class AscendLinearW8A8PluginOperation final : public atb::OperationInfra {
 public:
  // inv_scale_x = 1 / scale_x  (pre-computed by caller)
  // name_suffix is appended to internal subgraph names to avoid ATB global-name collisions
  // when multiple instances coexist in the same outer graph.
  explicit AscendLinearW8A8PluginOperation(float inv_scale_x, std::string name_suffix = "");
  ~AscendLinearW8A8PluginOperation() override;

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
  float inv_scale_x_;
  std::string name_suffix_;

  // quant_subgraph_op_ owns muls/round/clamp child ops after build().
  // linear_op_ is a standalone ATB Linear W8A8 op.
  atb::Operation* quant_subgraph_op_{nullptr};
  atb::Operation* linear_op_{nullptr};

  // Workspace layout cached by Setup(), consumed by Execute().
  struct WorkspaceLayout {
    uint64_t x_clamped_off{0};
    uint64_t x_clamped_bytes{0};
    uint64_t x_int8_off{0};
    uint64_t x_int8_bytes{0};
    uint64_t quant_ws_off{0};
    uint64_t quant_ws_bytes{0};
    uint64_t cast_ws_off{0};
    uint64_t cast_ws_bytes{0};
    uint64_t linear_ws_off{0};
    uint64_t linear_ws_bytes{0};
    uint64_t total{0};
  } ws_layout_{};

  // Opaque cache state that owns aclnnCast tensors/executor.
  void* cast_cache_state_{nullptr};

  void buildOps();
};

atb::Operation* createLinearW8A8PluginGraphOp(float inv_scale_x, std::string name_suffix = "");

}  // namespace mllm::ascend
