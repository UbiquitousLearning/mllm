// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/ParameterFile.hpp"

namespace mllm::aops {

/**
 * @brief Options for the Inverse Short-Time Fourier Transform (ISTFT) operation.
 */
struct ISTFTOpOptions : public BaseOpOptions<ISTFTOpOptions> {
  int n_fft = 0;
  int hop_length = 0;
  int win_length = 0;
  bool onesided = true;
  bool center = false;
  std::string pad_mode = "reflect";  // [constant, reflect, same(when center==false)]
  bool normalized = false;
  int length = 0;
};

class ISTFTOp : public BaseOp {
 public:
  explicit ISTFTOp(const ISTFTOpOptions& options);

  void load(const ParameterFile::ptr_t& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

 protected:
  ISTFTOpOptions options_;
};

}  // namespace mllm::aops