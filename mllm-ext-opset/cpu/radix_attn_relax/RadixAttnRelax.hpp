// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#include "mllm/mllm.hpp"
#include "mllm/backends/base/PluginInterface.hpp"

namespace mllm::ext_opset::cpu {

struct RadixAttnRelaxOptions : public mllm::BaseOpOptions<RadixAttnRelaxOptions> {
  int32_t B;
  int32_t q_head;
  int32_t kv_head;
  int32_t D_QK;
  int32_t D_V;
};

class RadixAttnRelax final : public mllm::plugin::interface::CustomizedOp {
 public:
  explicit RadixAttnRelax(const RadixAttnRelaxOptions& options) : CustomizedOp("radix_attn_relax"), options_(options) {}

  void load(const mllm::ParameterFile::ptr_t& ploader) override;

  void trace(void* trace_context, const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) override;

  void forward(const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) override;

  void reshape(const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) override;

  void setup(const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) override;

 protected:
  RadixAttnRelaxOptions options_;
};

class RadixAttnRelaxFactory final : public mllm::plugin::interface::CustomizedOpFactory<RadixAttnRelaxOptions> {
 public:
  inline std::shared_ptr<mllm::BaseOp> createOpImpl(const RadixAttnRelaxOptions& cargo) override {
    auto p = std::make_shared<RadixAttnRelax>(cargo);
    p->setOpType(opType());
    return p;
  }
};

}  // namespace mllm::ext_opset::cpu
