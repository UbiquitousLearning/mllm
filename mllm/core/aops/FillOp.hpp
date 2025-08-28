// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/ParameterFile.hpp"

namespace mllm::aops {

enum class FillOpTypes : int32_t {
  kFillOpTypes_Start = 0,
  kZeros,
  kOnes,
  kSpecific,
  kRandom,
  kArange,
  kFillOpTypes_End,
};

struct FillOpOptions : public BaseOpOptions<FillOpOptions> {
  FillOpTypes type = FillOpTypes::kFillOpTypes_Start;
  float value = 0.f;
  float start = 0.f;
  float end = 0.f;
  float step = 0.f;
  uint64_t seed = 42;
};

class FillOp : public BaseOp {
 public:
  explicit FillOp(const FillOpOptions& cargo);

  void load(const ParameterFile::ptr_t& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  inline const FillOpOptions& options() const { return options_; }

 protected:
  FillOpOptions options_;
};

}  // namespace mllm::aops
