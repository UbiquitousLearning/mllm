// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/nn/Layer.hpp"
#include "mllm/core/aops/STFTOp.hpp"

namespace mllm::nn {

/**
 * @brief Short-Time Fourier Transform (STFT) layer.
 *
 * This layer computes the Short-Time Fourier Transform of input signals,
 * which is commonly used in audio signal processing and spectral analysis.
 *
 * @param n_fft The size of Fourier transform, which creates n_fft // 2 + 1 frequency bins.
 * @param hop_length The distance between neighboring sliding window frames.
 * @param win_length The size of window frame and STFT window. If 0, it will be set to n_fft.
 * @param onesided Whether to return only the non-negative frequency bins.
 *
 * @input_0 Input tensor with shape (batch_size, signal_length)
 * @input_1 Window tensor with shape (win_length), should always be provided.
 *          Pass a tensor of ones for no windowing.
 * @output Output tensor with shape (batch_size, freq_bins, n_frames, 2) where the last dimension
 *         represents real and imaginary parts of the complex STFT coefficients.
 */
class STFT : public Layer {
 public:
  STFT();

  explicit STFT(const aops::STFTOpOptions& options);

  STFT(int n_fft, int hop_length, int win_length, bool onesided = true, bool center = false,
       const std::string& pad_mode = "constant");

  MLLM_LAYER_ANY_INPUTS_1_OUTPUTS_FORWARD
};

}  // namespace mllm::nn
