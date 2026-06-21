// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/ParameterFile.hpp"

namespace mllm::aops {

/**
 * @brief Options for the Short-Time Fourier Transform (STFT) operation.
 * It mainly configures following pytorch definition, but without normalized argument
 */
struct STFTOpOptions : public BaseOpOptions<STFTOpOptions> {
  int n_fft = 0;
  int hop_length = 0;
  int win_length = 0;
  bool onesided = true;
  bool center = false;
  std::string pad_mode = "reflect";  // [constant, reflect]
  bool return_complex = false;       // Whether to return a complex tensor
};

class STFTOp : public BaseOp {
 public:
  explicit STFTOp(const STFTOpOptions& options);

  void load(const ParameterFile::ptr_t& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  inline STFTOpOptions& options() { return options_; }

 protected:
  STFTOpOptions options_;
};

}  // namespace mllm::aops
