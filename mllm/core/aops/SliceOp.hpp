// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/ParameterFile.hpp"
#include "mllm/core/SlicePrimitives.hpp"

namespace mllm::aops {

struct SliceOpOptions : public BaseOpOptions<SliceOpOptions> {
  SliceIndices indices_;
};

class SliceOp : public BaseOp {
 public:
  explicit SliceOp(const SliceOpOptions& options);

  void load(const ParameterFile::ptr_t& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  inline const SliceOpOptions& options() const { return options_; }

 protected:
  SliceOpOptions options_;
};

}  // namespace mllm::aops