// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/STFTOp.hpp"
#include "mllm/nn/layers/STFT.hpp"

namespace mllm::nn {
STFT::STFT() : Layer(OpTypes::kSTFT, aops::STFTOpOptions{}) {}

STFT::STFT(const aops::STFTOpOptions& options) : Layer(OpTypes::kSTFT, options) {}

STFT::STFT(int n_fft, int hop_length, int win_length, bool onesided, bool center, const std::string& pad_mode,
           bool return_complex)
    : Layer(OpTypes::kSTFT, aops::STFTOpOptions{.n_fft = n_fft,
                                                .hop_length = hop_length,
                                                .win_length = win_length,
                                                .onesided = onesided,
                                                .center = center,
                                                .pad_mode = pad_mode,
                                                .return_complex = return_complex}) {}

}  // namespace mllm::nn
