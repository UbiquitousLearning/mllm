// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/backends/base/PluginInterface.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/nn/Layer.hpp"

// -------------------- Custom QNN Layers --------------------
namespace mllm::nn::qnn {

struct DequantizeAddOpOptions : public BaseOpOptions<DequantizeAddOpOptions> {
  DataTypes dtype;
  int32_t out_channels;
};

/**
 * @brief QNN Custom Layer: DequantizeAdd
 *
 * This layer performs dequantization of the input tensor followed by an element-wise addition with a bias tensor.
 * The bias is the previous linear layer's bias. This layer MUST be named with the name of the previous Linear plus
 * ".dequantize" to correctly load the bias during model loading.
 *
 */
class DequantizeAdd : public Layer {
 public:
  DequantizeAdd();

  explicit DequantizeAdd(DataTypes dtype, int32_t out_channels);

  MLLM_LAYER_ANY_INPUTS_1_OUTPUTS_FORWARD
};

}  // namespace mllm::nn::qnn

// -------------------- Custom QNN Ops --------------------
namespace mllm::qnn {

class DequantizeAddOp final : public mllm::plugin::interface::CustomizedOp {
 public:
  explicit DequantizeAddOp(const nn::qnn::DequantizeAddOpOptions& options) : CustomizedOp("DequantizeAdd"), options_(options) {}

  void load(const mllm::ParameterFile::ptr_t& ploader) override;

  void trace(void* trace_context, const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) override;

  void forward(const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) override {}

  void reshape(const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) override;

  void setup(const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) override {}

  // Public accessors for QNN pattern matching
  const Tensor& getWeightTensor() const { return weight_; }
  const nn::qnn::DequantizeAddOpOptions& getOptions() const { return options_; }

 protected:
  nn::qnn::DequantizeAddOpOptions options_;
  Tensor weight_;
};

class DequantizeAddFactory final : public mllm::plugin::interface::CustomizedOpFactory<nn::qnn::DequantizeAddOpOptions> {
 public:
  inline std::shared_ptr<mllm::BaseOp> createOpImpl(const nn::qnn::DequantizeAddOpOptions& cargo) override {
    auto p = std::make_shared<DequantizeAddOp>(cargo);
    p->setOpType(opType());
    return p;
  }
};

}  // namespace mllm::qnn
