// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#include "mllm/mllm.hpp"
#include "mllm/backends/base/PluginInterface.hpp"

namespace mllm::ext_opset::cpu {

enum class RadixAttnSwaSinkPattern : uint8_t {
  kPrefill = 0,
  kDecode = 1,
  kAppend = 2,
};

struct RadixAttnSwaSinkOptions : public mllm::BaseOpOptions<RadixAttnSwaSinkOptions> {
  int32_t B;
  int32_t q_head;
  int32_t kv_head;
  int32_t D_QK;
  int32_t D_V;
  int32_t cur_seq_len;
  int sliding_window = -1;
  bool s_aux_enable = false;
  RadixAttnSwaSinkPattern pattern = RadixAttnSwaSinkPattern::kDecode;
};

class RadixAttnSwaSink final : public mllm::plugin::interface::CustomizedOp {
 public:
  explicit RadixAttnSwaSink(const RadixAttnSwaSinkOptions& options) : CustomizedOp("radix_attn_swa_sink"), options_(options) {}

  void load(const mllm::ParameterFile::ptr_t& ploader) override;

  void trace(void* trace_context, const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) override;

  void forward(const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) override;

  void reshape(const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) override;

  void setup(const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) override;

 protected:
  RadixAttnSwaSinkOptions options_;
};

class RadixAttnSwaSinkFactory final : public mllm::plugin::interface::CustomizedOpFactory<RadixAttnSwaSinkOptions> {
 public:
  inline std::shared_ptr<mllm::BaseOp> createOpImpl(const RadixAttnSwaSinkOptions& cargo) override {
    auto p = std::make_shared<RadixAttnSwaSink>(cargo);
    p->setOpType(opType());
    return p;
  }
};

}  // namespace mllm::ext_opset::cpu
