// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/ParameterFile.hpp"
#include "mllm/core/Tensor.hpp"

namespace mllm::aops {

struct IndexOpOptions : public BaseOpOptions<IndexOpOptions> {
  ComplexIndexingList indices_;
};

class IndexOp : public BaseOp {
 public:
  explicit IndexOp(const IndexOpOptions& options);

  void load(const ParameterFile::ptr_t& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  inline const IndexOpOptions& options() const { return options_; }

 protected:
  void calculateOutputShape(const Tensor& input, Tensor::shape_t& o_shape) const;

  void processTensorIndices(const Tensor& input, const std::vector<int>& out_indices, std::vector<int>& in_indices,
                            bool& valid_index) const;

  IndexOpOptions options_;
};

}  // namespace mllm::aops