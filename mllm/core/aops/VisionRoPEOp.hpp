// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/ParameterFile.hpp"

namespace mllm::aops {

enum class VisionRoPEOpOptionsType : uint8_t {
  kStart = 0,
  kQwen2VL,
  kEnd,
};

struct Qwen2VLRoPEOpOptions {
  int32_t dims;
  int32_t spatial_merge_size = 2;
  float theta;
};

struct VisionRoPEOpOptions : public BaseOpOptions<VisionRoPEOpOptions> {
  VisionRoPEOpOptionsType type;
  union {
    Qwen2VLRoPEOpOptions qwen2vl_rope_op_options;
  };
};

class VisionRoPEOp : public BaseOp {
 public:
  explicit VisionRoPEOp(const VisionRoPEOpOptions& options);

  void load(const ParameterFile::ptr_t& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  inline const VisionRoPEOpOptions& options() const { return options_; }

 protected:
  VisionRoPEOpOptions options_;
};

}  // namespace mllm::aops