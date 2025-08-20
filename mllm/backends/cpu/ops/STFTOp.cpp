// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cstring>
#include <cmath>
#include <vector>
#include <complex>
#include "mllm/mllm.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/utils/Log.hpp"
#include "mllm/backends/cpu/ops/STFTOp.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace mllm::cpu {

CPUSTFTOp::CPUSTFTOp(const aops::STFTOpOptions& options) : aops::STFTOp(options) {}

template<typename T>
static T computeAngularVelocity(size_t number_of_samples, bool inverse = false) {
  // Calculate fundamental angular velocity
  static const T pi = static_cast<T>(M_PI);
  static const T tau = 2 * pi;
  T inverse_switch = inverse ? 1.f : -1.f;
  T angular_velocity = inverse_switch * tau / number_of_samples;
  return angular_velocity;
}

static bool is_power_of_2(size_t size) {
  size_t n_bits = 0;
  while (size != 0) {
    n_bits += size & 1;
    size = size >> 1;
  }
  return n_bits == 1;
}

template<typename T>
T nextPowerOf2(T in) {
  in--;
  T out = 1;
  while (out <= in) { out <<= 1; }
  return out;
}

// Bluestein's algorithm for FFT of arbitrary size
template<typename T>
static void fftBluestein(const std::vector<float>& input, std::vector<std::complex<T>>& output, int n_fft) {
  // Next power of 2 for efficient FFT
  size_t N = static_cast<size_t>(n_fft);
  size_t M = nextPowerOf2(2 * N - 1);

  // Precompute chirp signal
  std::vector<std::complex<T>> chirp(M, std::complex<T>(0, 0));
  std::vector<std::complex<T>> b(M, std::complex<T>(0, 0));
  std::vector<std::complex<T>> b_fft(M, std::complex<T>(0, 0));

  static const T pi = static_cast<T>(M_PI);
  for (size_t n = 0; n < N; n++) {
    T exponent = -pi * n * n / N;
    chirp[n] = std::complex<T>(std::cos(exponent), std::sin(exponent));
    b[n] = std::conj(chirp[n]);
  }

  // Fill the rest of b
  for (size_t n = M - N + 1; n < M; n++) { b[n] = std::conj(b[M - n]); }

  // Compute FFT of b
  std::vector<std::complex<T>> temp_fft(M);
  fftRadix2(b, b_fft, static_cast<int>(M));

  // Prepare a signal
  std::vector<std::complex<T>> a(M, std::complex<T>(0, 0));
  for (size_t n = 0; n < input.size() && n < N; n++) { a[n] = std::complex<T>(input[n], 0) * chirp[n]; }

  // Compute FFT of a
  fftRadix2(a, temp_fft, static_cast<int>(M));

  // Convolution in frequency domain
  for (size_t i = 0; i < M; i++) { temp_fft[i] *= b_fft[i]; }

  // Inverse FFT
  // For inverse FFT, we conjugate the twiddle factors
  std::vector<std::complex<T>> temp_ifft(M);
  fftRadix2(temp_fft, temp_ifft, static_cast<int>(M), true);

  // Final scaling and chirp multiplication
  for (size_t i = 0; i < N; i++) {
    T scale = static_cast<T>(1.0) / M;
    if (i == 0) {
      output[i] = temp_ifft[i] * chirp[i] * scale;
    } else {
      output[i] = temp_ifft[M - i] * chirp[i] * scale;
    }
  }
}

// 位反转查找表
static const unsigned char BitReverseTable256[] = {
    0x00, 0x80, 0x40, 0xC0, 0x20, 0xA0, 0x60, 0xE0, 0x10, 0x90, 0x50, 0xD0, 0x30, 0xB0, 0x70, 0xF0, 0x08, 0x88, 0x48, 0xC8,
    0x28, 0xA8, 0x68, 0xE8, 0x18, 0x98, 0x58, 0xD8, 0x38, 0xB8, 0x78, 0xF8, 0x04, 0x84, 0x44, 0xC4, 0x24, 0xA4, 0x64, 0xE4,
    0x14, 0x94, 0x54, 0xD4, 0x34, 0xB4, 0x74, 0xF4, 0x0C, 0x8C, 0x4C, 0xCC, 0x2C, 0xAC, 0x6C, 0xEC, 0x1C, 0x9C, 0x5C, 0xDC,
    0x3C, 0xBC, 0x7C, 0xFC, 0x02, 0x82, 0x42, 0xC2, 0x22, 0xA2, 0x62, 0xE2, 0x12, 0x92, 0x52, 0xD2, 0x32, 0xB2, 0x72, 0xF2,
    0x0A, 0x8A, 0x4A, 0xCA, 0x2A, 0xAA, 0x6A, 0xEA, 0x1A, 0x9A, 0x5A, 0xDA, 0x3A, 0xBA, 0x7A, 0xFA, 0x06, 0x86, 0x46, 0xC6,
    0x26, 0xA6, 0x66, 0xE6, 0x16, 0x96, 0x56, 0xD6, 0x36, 0xB6, 0x76, 0xF6, 0x0E, 0x8E, 0x4E, 0xCE, 0x2E, 0xAE, 0x6E, 0xEE,
    0x1E, 0x9E, 0x5E, 0xDE, 0x3E, 0xBE, 0x7E, 0xFE, 0x01, 0x81, 0x41, 0xC1, 0x21, 0xA1, 0x61, 0xE1, 0x11, 0x91, 0x51, 0xD1,
    0x31, 0xB1, 0x71, 0xF1, 0x09, 0x89, 0x49, 0xC9, 0x29, 0xA9, 0x69, 0xE9, 0x19, 0x99, 0x59, 0xD9, 0x39, 0xB9, 0x79, 0xF9,
    0x05, 0x85, 0x45, 0xC5, 0x25, 0xA5, 0x65, 0xE5, 0x15, 0x95, 0x55, 0xD5, 0x35, 0xB5, 0x75, 0xF5, 0x0D, 0x8D, 0x4D, 0xCD,
    0x2D, 0xAD, 0x6D, 0xED, 0x1D, 0x9D, 0x5D, 0xDD, 0x3D, 0xBD, 0x7D, 0xFD, 0x03, 0x83, 0x43, 0xC3, 0x23, 0xA3, 0x63, 0xE3,
    0x13, 0x93, 0x53, 0xD3, 0x33, 0xB3, 0x73, 0xF3, 0x0B, 0x8B, 0x4B, 0xCB, 0x2B, 0xAB, 0x6B, 0xEB, 0x1B, 0x9B, 0x5B, 0xDB,
    0x3B, 0xBB, 0x7B, 0xFB, 0x07, 0x87, 0x47, 0xC7, 0x27, 0xA7, 0x67, 0xE7, 0x17, 0x97, 0x57, 0xD7, 0x37, 0xB7, 0x77, 0xF7,
    0x0F, 0x8F, 0x4F, 0xCF, 0x2F, 0xAF, 0x6F, 0xEF, 0x1F, 0x9F, 0x5F, 0xDF, 0x3F, 0xBF, 0x7F, 0xFF};

// 位反转函数
template<typename T>
static inline T bitReverse(T num, unsigned significant_bits) {
  if (significant_bits > 32) {
    MLLM_ERROR("Unsupported bit size.");
    return 0;
  }
  uint32_t num_32 = static_cast<uint32_t>(num);
  uint32_t rev = (BitReverseTable256[num_32 & 0xff] << 24) | (BitReverseTable256[(num_32 >> 8) & 0xff] << 16)
                 | (BitReverseTable256[(num_32 >> 16) & 0xff] << 8) | (BitReverseTable256[(num_32 >> 24) & 0xff]);
  return static_cast<T>(((uint64_t)rev) >> (32 - significant_bits));
}

static void fftRadix2(const std::vector<float>& input, std::vector<std::complex<float>>& output, int n_fft,
                      bool inverse = false) {
  unsigned log2_n = static_cast<unsigned>(std::log2(n_fft));

  // 使用位反转表进行位反转排列
  for (int i = 0; i < n_fft; ++i) {
    int reversed = bitReverse(i, log2_n);
    if (reversed < input.size()) {
      output[i] = std::complex<float>(input[reversed], 0.0f);
    } else {
      output[i] = std::complex<float>(0.0f, 0.0f);
    }
  }

  auto angular_velocity = computeAngularVelocity<float>(n_fft, inverse);

  // Cooley-Tukey FFT算法
  for (size_t stride = 2; stride <= n_fft; stride <<= 1) {
    size_t midpoint = stride >> 1;

    float angle = angular_velocity * static_cast<float>(n_fft / stride);
    std::complex<float> w_m(std::cos(angle), std::sin(angle));  // Note: sin is positive for inverse FFT

    for (size_t j = 0; j < n_fft; j += stride) {
      std::complex<float> w(1.0f, 0.0f);
      for (size_t k = 0; k < midpoint; k++) {
        std::complex<float> even = output[j + k];
        std::complex<float> odd = output[j + k + midpoint];

        std::complex<float> t = w * odd;
        output[j + k] = even + t;
        output[j + k + midpoint] = even - t;

        w = w * w_m;
      }
    }
  }

  // For inverse FFT, apply scaling
  if (inverse) {
    for (int i = 0; i < n_fft; ++i) { output[i] /= static_cast<float>(n_fft); }
  }
}

void CPUSTFTOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& input = inputs[0];
  auto& window = inputs[1];
  auto& output = outputs[0];

  // Get input dimensions
  auto input_shape = input.shape();
  auto output_shape = output.shape();
  int batch_size = output_shape[0];
  int signal_length = input_shape.size() == 1 ? input_shape[0] : input_shape[1];

  // Get output dimensions
  int freq_bins = output_shape[1];
  int n_frames = output_shape[2];

  // STFT parameters
  int n_fft = options_.n_fft;
  int hop_length = options_.hop_length;
  int win_length = options_.win_length;
  // If win_length is not specified, use n_fft
  if (win_length == 0) { win_length = n_fft; }

  // Temporary buffers
  std::vector<float> windowed_input(n_fft);
  std::vector<std::complex<float>> fft_output(n_fft);

  // Process each batch
  for (int b = 0; b < batch_size; ++b) {
    // Process each frame
    for (int f = 0; f < n_frames; ++f) {
      int start_idx = f * hop_length;

      // Get windowed input and apply window function
      // do the same padding as `center=False` in pytorch
      // TODO: support center=True and pad_mode
      for (int i = 0; i < win_length; ++i) {
        if (start_idx + i < signal_length) {
          windowed_input[i] = input.ptr<float>()[b * signal_length + start_idx + i] * window.ptr<float>()[i];
        } else {
          windowed_input[i] = 0.0f;
        }
      }
      // Zero-pad if needed
      for (int i = win_length; i < n_fft; ++i) { windowed_input[i] = 0.0f; }

      // Compute FFT
      if (is_power_of_2(n_fft)) {
        fftRadix2(windowed_input, fft_output, n_fft);
      } else {
        NYI("FFT for non-power-of-2 sizes is not implemented yet.");
      }

      // Store onesided output (real and imaginary parts)
      for (int freq = 0; freq < freq_bins; ++freq) {
        output.at<float>({b, freq, f, 0}) = fft_output[freq].real();
        output.at<float>({b, freq, f, 1}) = fft_output[freq].imag();
      }
    }
  }
}

}  // namespace mllm::cpu
