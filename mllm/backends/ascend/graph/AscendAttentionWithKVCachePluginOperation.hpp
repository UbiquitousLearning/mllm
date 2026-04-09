// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <string>

#include <atb/operation_infra.h>

namespace mllm::ascend {

class AscendAttentionWithKVCachePluginOperation final : public atb::OperationInfra {
 public:
  AscendAttentionWithKVCachePluginOperation(int32_t num_attention_heads,
                                            int32_t num_key_value_heads,
                                            int32_t head_dim,
                                            int32_t max_cache_length,
                                            bool sliding_window = false,
                                            int32_t window_size = 0);
  ~AscendAttentionWithKVCachePluginOperation() override;

  std::string GetName() const override;
  atb::Status InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs,
                         atb::SVector<atb::TensorDesc>& outTensorDescs) const override;
  uint32_t GetInputNum() const override;
  uint32_t GetOutputNum() const override;
  atb::Status Setup(const atb::VariantPack& variantPack, uint64_t& workspaceSize, atb::Context* context) override;
  atb::Status Execute(const atb::VariantPack& variantPack,
                      uint8_t* workspace,
                      uint64_t workspaceSize,
                      atb::Context* context) override;

 private:
  int32_t num_attention_heads_;
  int32_t num_key_value_heads_;
  int32_t num_key_value_groups_;
  int32_t head_dim_;
  int32_t max_cache_length_;
  bool sliding_window_;
  int32_t window_size_;

  atb::Operation* prefill_subgraph_op_{nullptr};
  atb::Operation* decode_subgraph_op_{nullptr};
  atb::Operation* matmul_qk_op_{nullptr};
  atb::Operation* scale_mul_op_{nullptr};
  atb::Operation* mask_tensor_op_{nullptr};
  atb::Operation* mask_add_op_{nullptr};
  atb::Operation* softmax_op_{nullptr};
  atb::Operation* matmul_av_op_{nullptr};
  atb::Operation* decode_matmul_qk_op_{nullptr};
  atb::Operation* decode_scale_mul_op_{nullptr};
  atb::Operation* decode_softmax_op_{nullptr};
  atb::Operation* decode_matmul_av_op_{nullptr};
  uint64_t prefill_subgraph_workspace_bytes_{0};
  uint64_t decode_subgraph_workspace_bytes_{0};

  void buildAttentionSubgraph();
};

atb::Operation* createAttentionWithKVCachePluginGraphOp(int32_t num_attention_heads,
                                                        int32_t num_key_value_heads,
                                                        int32_t head_dim,
                                                        int32_t max_cache_length,
                                                        bool sliding_window = false,
                                                        int32_t window_size = 0);

}  // namespace mllm::ascend
