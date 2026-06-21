// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#include "mllm/mllm.hpp"
#include "mllm/backends/base/PluginInterface.hpp"

namespace mllm::ext_opset::cpu {

struct FlashAttention2SwaSinkOptions : public mllm::BaseOpOptions<FlashAttention2SwaSinkOptions> {
  int32_t B;
  int32_t q_head;
  int32_t kv_head;
  int32_t D_QK;
  int32_t D_V;
  int32_t cur_seq_len;
  int sliding_window = -1;
  bool s_aux_enable = false;
};

class FlashAttention2SwaSink final : public mllm::plugin::interface::CustomizedOp {
 public:
  explicit FlashAttention2SwaSink(const FlashAttention2SwaSinkOptions& options)
      : CustomizedOp("flash_attention_2_swa_sink"), options_(options) {}

  void load(const mllm::ParameterFile::ptr_t& ploader) override;

  void trace(void* trace_context, const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) override;

  void forward(const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) override;

  void reshape(const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) override;

  void setup(const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) override;

 protected:
  FlashAttention2SwaSinkOptions options_;
};

class FlashAttention2SwaSinkFactory final : public mllm::plugin::interface::CustomizedOpFactory<FlashAttention2SwaSinkOptions> {
 public:
  inline std::shared_ptr<mllm::BaseOp> createOpImpl(const FlashAttention2SwaSinkOptions& cargo) override {
    auto p = std::make_shared<FlashAttention2SwaSink>(cargo);
    p->setOpType(opType());
    return p;
  }
};

}  // namespace mllm::ext_opset::cpu
