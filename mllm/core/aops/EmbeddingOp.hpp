// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/ParameterFile.hpp"

namespace mllm::aops {

struct EmbeddingOpOptions : public BaseOpOptions<EmbeddingOpOptions> {
  int vocab_size = 0;
  int hidden_size = 0;
};

class EmbeddingOp : public BaseOp {
 public:
  explicit EmbeddingOp(const EmbeddingOpOptions& options);

  void load(const ParameterFile::ptr_t& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  inline Tensor& weight() { return weight_; }

  ParameterFile::ptr_t getParams() override;

  inline const EmbeddingOpOptions& options() const { return options_; }

 protected:
  Tensor weight_;
  EmbeddingOpOptions options_;
};

}  // namespace mllm::aops
