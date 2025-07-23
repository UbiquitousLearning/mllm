/**
 * @file LinearOp.hpp
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2025-07-23
 *
 */
#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/ParameterFile.hpp"

namespace mllm::aops {

enum class LinearImplTypes {
  kLinearImplTypes_Start = 0,
  kDefault,

  kKleidiai_Start,
  // Add KAI quantized linear
  kKleidiai_End,

  kGGUF_Start,
  // Add GGUF quantized linear
  kGGUF_End,

  kLinearImplTypes_End,
};

struct LinearOptions : public BaseOpOptions<LinearOptions> {
  int32_t in_channels;
  int32_t out_channels;
  bool bias;
  LinearImplTypes impl_type;
};

class LinearOp : public BaseOp {
 public:
  explicit LinearOp(const LinearOptions& cargo);

  void load(const ParameterFile::ptr_t& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  ParameterFile::ptr_t getParams() override;

  inline Tensor& weight() { return weight_; }

  inline Tensor& bias() { return bias_; }

  inline const LinearOptions& cargo() const { return cargo_; }

 protected:
  Tensor weight_;
  Tensor bias_;
  LinearOptions cargo_;
};

}  // namespace mllm::aops
