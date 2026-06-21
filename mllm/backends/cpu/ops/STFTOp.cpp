// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cstring>
#include <cmath>
#include <vector>
#include <complex>
#include "mllm/core/DataTypes.hpp"
// #include "mllm/mllm.hpp"
#include "mllm/utils/Log.hpp"
#include "mllm/backends/cpu/ops/STFTOp.hpp"
#include "mllm/backends/cpu/ops/ISTFTOp.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace mllm::cpu {

CPUSTFTOp::CPUSTFTOp(const aops::STFTOpOptions& options) : aops::STFTOp(options) {}

CPUISTFTOp::CPUISTFTOp(const aops::ISTFTOpOptions& options) : aops::ISTFTOp(options) {}

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

template<typename T>
static void fftRadix2(const std::vector<T>& input, std::vector<std::complex<T>>& output, int n_fft, bool inverse = false) {
  const unsigned log2_n = static_cast<unsigned>(std::log2(n_fft));

  // 使用位反转表进行位反转排列
  for (int i = 0; i < n_fft; ++i) {
    int reversed = bitReverse(i, log2_n);
    if (reversed < input.size()) {
      output[i] = std::complex<T>(input[reversed], 0.0f);
    } else {
      output[i] = std::complex<T>(0.0f, 0.0f);
    }
  }

  auto angular_velocity = computeAngularVelocity<T>(n_fft, inverse);

  // Cooley-Tukey FFT算法
  for (size_t stride = 2; stride <= n_fft; stride <<= 1) {
    size_t midpoint = stride >> 1;

    T angle = angular_velocity * static_cast<T>(n_fft / stride);
    std::complex<T> w_m(std::cos(angle), std::sin(angle));  // Note: inverse is handled by computeAngularVelocity()

    for (size_t j = 0; j < n_fft; j += stride) {
      std::complex<T> w(1.0f, 0.0f);
      for (size_t k = 0; k < midpoint; k++) {
        std::complex<T> even = output[j + k];
        std::complex<T> odd = output[j + k + midpoint];

        std::complex<T> t = w * odd;
        output[j + k] = even + t;
        output[j + k + midpoint] = even - t;

        w = w * w_m;
      }
    }
  }

  // For inverse FFT, apply scaling
  if (inverse) {
    for (int i = 0; i < n_fft; ++i) { output[i] /= static_cast<T>(n_fft); }
  }
}

template<typename T>
static void fftRadix2Complex(std::vector<std::complex<T>>& data, const int n_fft, bool inverse = false) {
  if (n_fft == 0) return;

  const unsigned log2_n = static_cast<unsigned>(std::log2(n_fft));

  for (int i = 0; i < n_fft; ++i) {
    int reversed = bitReverse(i, log2_n);
    if (reversed < data.size() && i < reversed) { std::swap(data[i], data[reversed]); }
  }

  auto angular_velocity = computeAngularVelocity<T>(n_fft, inverse);

  // 2. Cooley-Tukey FFT 蝶形运算
  for (size_t stride = 2; stride <= n_fft; stride <<= 1) {
    size_t midpoint = stride >> 1;

    T angle = angular_velocity * static_cast<T>(n_fft / stride);
    std::complex<T> w_m(std::cos(angle), std::sin(angle));  // Note: inverse is handled by computeAngularVelocity()

    for (size_t j = 0; j < n_fft; j += stride) {
      std::complex<T> w(1.0, 0.0);
      for (size_t k = 0; k < midpoint; k++) {
        std::complex<T> even = data[j + k];
        std::complex<T> odd = data[j + k + midpoint];

        std::complex<T> t = w * odd;
        data[j + k] = even + t;
        data[j + k + midpoint] = even - t;

        w *= w_m;  // 更新旋转因子
      }
    }
  }

  // For inverse FFT, apply scaling
  if (inverse) {
    for (size_t i = 0; i < n_fft; ++i) { data[i] /= static_cast<T>(n_fft); }
  }
}

template<typename T>
static void fftBluestein(const std::vector<T>& input, std::vector<std::complex<T>>& output, int n_fft, bool inverse = false) {
  const size_t N = n_fft;
  if (N == 0) {
    output.clear();
    return;
  }

  // 1. 确定用于卷积的FFT尺寸M，M >= 2*N-1 且为2的幂
  size_t M = nextPowerOf2(2 * N - 1);

  const T pi = static_cast<T>(M_PI);
  const T direction = inverse ? 1.f : -1.f;

  // 2. 创建序列a(n): 输入信号与预处理chirp相乘
  std::vector<std::complex<T>> a(M, {0, 0});
  for (size_t n = 0; n < N; n++) {
    if (n < input.size()) {
      T exponent = direction * pi * n * n / N;
      std::complex<T> chirp(cos(exponent), sin(exponent));
      a[n] = input[n] * chirp;
    }
  }

  // 3. 创建序列b(n): 用于卷积的chirp核
  std::vector<std::complex<T>> b(M, {0, 0});
  for (size_t n = 0; n < N; n++) {
    T exponent = -direction * pi * n * n / N;  // 注意这里是反号的
    b[n] = {static_cast<float>(cos(exponent)), static_cast<float>(sin(exponent))};
  }
  // 创建循环卷积核
  for (size_t n = 1; n < N; n++) { b[M - n] = b[n]; }

  // 4. 执行卷积: IFFT(FFT(a) * FFT(b))
  fftRadix2Complex<T>(a, M, false);  // FFT of a
  fftRadix2Complex<T>(b, M, false);  // FFT of b

  for (size_t i = 0; i < M; i++) {
    a[i] *= b[i];  // 频域相乘
  }

  fftRadix2Complex<T>(a, M, true);  // IFFT of the product

  // 5. 后处理与最终输出
  output.resize(N);
  for (size_t n = 0; n < N; n++) {
    T exponent = direction * pi * n * n / N;
    std::complex<T> chirp(cos(exponent), sin(exponent));
    output[n] = a[n] * chirp;
  }

  // 如果是逆变换，应用1/N缩放
  if (inverse) {
    for (size_t i = 0; i < N; ++i) { output[i] /= static_cast<T>(N); }
  }
}

template<typename T>
static void fftBluesteinComplex(std::vector<std::complex<T>>& input, std::vector<std::complex<T>>& output, int n_fft,
                                bool inverse = false) {
  const size_t N = n_fft;
  if (N == 0) {
    output.clear();
    return;
  }

  // 1. 确定用于卷积的FFT尺寸M，M >= 2*N-1 且为2的幂
  size_t M = nextPowerOf2(2 * N - 1);

  const T pi = static_cast<T>(M_PI);
  const T direction = inverse ? 1.f : -1.f;

  // 2. 创建序列a(n): 输入信号与预处理chirp相乘
  std::vector<std::complex<T>> a(M, {0, 0});
  for (size_t n = 0; n < N; n++) {
    if (n < input.size()) {
      T exponent = direction * pi * n * n / N;
      std::complex<T> chirp(cos(exponent), sin(exponent));
      a[n] = input[n] * chirp;
    }
  }

  // 3. 创建序列b(n): 用于卷积的chirp核
  std::vector<std::complex<T>> b(M, {0, 0});
  for (size_t n = 0; n < N; n++) {
    T exponent = -direction * pi * n * n / N;  // 注意这里是反号的
    b[n] = {static_cast<float>(cos(exponent)), static_cast<float>(sin(exponent))};
  }
  // 创建循环卷积核
  for (size_t n = 1; n < N; n++) { b[M - n] = b[n]; }

  // 4. 执行卷积: IFFT(FFT(a) * FFT(b))
  fftRadix2Complex<T>(a, M, false);  // FFT of a
  fftRadix2Complex<T>(b, M, false);  // FFT of b

  for (size_t i = 0; i < M; i++) {
    a[i] *= b[i];  // 频域相乘
  }

  fftRadix2Complex<T>(a, M, true);  // IFFT of the product

  // 5. 后处理与最终输出
  output.resize(N);
  for (size_t n = 0; n < N; n++) {
    T exponent = direction * pi * n * n / N;
    std::complex<T> chirp(cos(exponent), sin(exponent));
    output[n] = a[n] * chirp;
  }

  // 如果是逆变换，应用1/N缩放
  if (inverse) {
    for (size_t i = 0; i < N; ++i) { output[i] /= static_cast<T>(N); }
  }
}

static void padSignal(const Tensor& input, std::vector<float>& padded_signal, int n_fft, bool center,
                      const std::string& pad_mode) {
  auto input_shape = input.shape();
  int signal_length = input_shape.size() == 1 ? input_shape[0] : input_shape[1];

  if (center) {
    int pad_length = n_fft / 2;
    int padded_length = signal_length + 2 * pad_length;
    padded_signal.resize(padded_length);

    // 复制原始信号
    for (int i = 0; i < signal_length; ++i) { padded_signal[pad_length + i] = input.ptr<float>()[i]; }

    if (pad_mode == "reflect") {
      // reflect：mirror padding
      for (int i = 0; i < pad_length; ++i) {
        // left padding: reverse copy from the beginning, not including the boundary
        padded_signal[i] = input.ptr<float>()[pad_length - i];  // 1...pad_length
        // right padding: reverse copy from the end, not including the boundary
        padded_signal[pad_length + signal_length + i] = input.ptr<float>()[signal_length - 2 - i];
      }
    } else {
      // constant：0 padding (default)
      for (int i = 0; i < pad_length; ++i) {
        padded_signal[i] = 0.0f;
        padded_signal[pad_length + signal_length + i] = 0.0f;
      }
    }
  } else {
    // center=false，no padding
    padded_signal.resize(signal_length);
    for (int i = 0; i < signal_length; ++i) { padded_signal[i] = input.ptr<float>()[i]; }
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
  int win_length = options_.win_length;  // win_length <= n_fft has been verified in aops
  bool center = options_.center;
  std::string pad_mode = options_.pad_mode;
  bool return_complex = options_.return_complex;

  // 对输入信号进行填充处理
  std::vector<float> padded_signal;
  padSignal(input, padded_signal, n_fft, center, pad_mode);
  int padded_length = static_cast<int>(padded_signal.size());

  // Recalculate signal length after padding
  signal_length = padded_length;

  // Temporary buffers
  std::vector<float> windowed_input(n_fft);
  std::vector<std::complex<float>> fft_output(n_fft);

  // Process each batch
  for (int b = 0; b < batch_size; ++b) {
    // Process each frame
    for (int f = 0; f < n_frames; ++f) {
      int start_idx = f * hop_length;

      // Get windowed input and apply window function
      // Apply padding based on center and pad_mode parameters
      for (int i = 0; i < win_length; ++i) {
        if (start_idx + i < signal_length) {
          windowed_input[i] = padded_signal[start_idx + i] * window.ptr<float>()[i];
        } else {
          windowed_input[i] = 0.0f;
        }
      }

      // Compute FFT
      if (is_power_of_2(n_fft)) {
        fftRadix2<float>(windowed_input, fft_output, n_fft);
      } else {
        fftBluestein<float>(windowed_input, fft_output, n_fft);
      }

      // Store output based on return_complex option
      if (return_complex) {
        // Store as complex values
        for (int freq = 0; freq < freq_bins; ++freq) { output.at<std::complex<float>>({b, freq, f}) = fft_output[freq]; }
      } else {
        // Store as real and imaginary parts in the last dimension
        for (int freq = 0; freq < freq_bins; ++freq) {
          output.at<float>({b, freq, f, 0}) = fft_output[freq].real();
          output.at<float>({b, freq, f, 1}) = fft_output[freq].imag();
        }
      }
    }
  }
}

void CPUISTFTOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& input = inputs[0];
  auto& window = inputs[1];
  auto& output = outputs[0];

  // Shapes
  auto input_shape = input.shape();
  auto output_shape = output.shape();
  int batch_size = output_shape[0];
  // signal_length 现在代表了最终裁剪后的目标长度
  int signal_length = output_shape[1];

  int freq_bins = input_shape[1];
  int n_frames = input_shape[2];

  // ISTFT params
  int n_fft = options_.n_fft;
  int hop_length = options_.hop_length;
  int win_length = options_.win_length;
  bool center = options_.center;
  std::string pad_mode = options_.pad_mode;
  bool normalized = options_.normalized;

  // Buffers for IFFT
  std::vector<std::complex<float>> fft_input(n_fft);
  std::vector<std::complex<float>> fft_output(n_fft);
  std::vector<float> ifft_output(n_fft);

  const float* window_ptr = window.ptr<float>();

  // Process each batch
  for (int b = 0; b < batch_size; ++b) {
    // 获取当前 batch 的输出指针
    float* output_ptr = output.ptrAt<float>({b, 0});

    // python vocos library use this pad_mode, just to keep consistent with it
    if (pad_mode == "same" && !center) {
      // 'same' padding 逻辑
      int padding = (win_length - hop_length) / 2;
      int final_length = n_frames * hop_length;

      // 预分配的输出Tensor长度应与期望的最终长度一致
      // 在某些情况下，由于STFT的ceil操作，可能存在1个点的差异，这里放宽检查
      MLLM_RT_ASSERT(std::abs(signal_length - final_length) <= 1);

      // 1. 分配用于重叠相加的临时、全长缓冲区
      int full_length = (n_frames - 1) * hop_length + win_length;
      std::vector<float> temp_output(full_length, 0.0f);
      std::vector<float> temp_envelope(full_length, 0.0f);

      // 2. 在临时缓冲区中执行重叠相加
      for (int f = 0; f < n_frames; ++f) {
        // load complex freq bins
        for (int freq = 0; freq < freq_bins; ++freq) {
          if (input.dtype() == kComplexFloat32) {
            fft_input[freq] = input.constAt<mllm_complex_fp32_t>({b, freq, f});
          } else {
            float real = input.constAt<float>({b, freq, f, 0});
            float imag = input.constAt<float>({b, freq, f, 1});
            fft_input[freq] = std::complex<float>(real, imag);
          }
        }

        // reconstruct onesided spectrum if needed
        if (options_.onesided) {
          for (int freq = freq_bins; freq < n_fft; ++freq) {
            int conj_freq = n_fft - freq;
            fft_input[freq] = std::conj(fft_input[conj_freq]);
          }
        }

        // IFFT
        if (is_power_of_2(n_fft)) {
          fftRadix2Complex<float>(fft_input, n_fft, true);  // In-place FFT
        } else {
          fftBluesteinComplex<float>(fft_input, fft_output, n_fft, true);
          fft_input = fft_output;
        }

        // IFFT output (real part) and normalization
        for (int i = 0; i < n_fft; ++i) {
          ifft_output[i] = fft_input[i].real();
          if (normalized) ifft_output[i] /= n_fft;
        }

        // overlap-add
        int start_idx = f * hop_length;
        for (int i = 0; i < win_length; ++i) {
          int out_idx = start_idx + i;
          if (out_idx < full_length) {
            temp_output[out_idx] += ifft_output[i] * window_ptr[i];
            temp_envelope[out_idx] += window_ptr[i] * window_ptr[i];
          }
        }
      }

      // 3. 在临时缓冲区中归一化
      for (int i = 0; i < full_length; ++i) {
        if (temp_envelope[i] > 1e-10f) { temp_output[i] /= temp_envelope[i]; }
      }

      // 4. 将裁剪后的有效数据从临时缓冲拷贝到最终的 output tensor
      // 源地址：temp_output 的 padding 之后
      // 目标地址：output_ptr 的开头
      // 拷贝长度：final_length
      std::memcpy(output_ptr, temp_output.data() + padding, final_length * sizeof(float));

    } else {
      // 标准 'center' 或其他 padding 逻辑 (与您之前的原始逻辑类似)
      std::vector<float> window_envelope(signal_length, 0.0f);
      std::memset(output_ptr, 0, signal_length * sizeof(float));

      for (int f = 0; f < n_frames; ++f) {
        // load complex freq bins
        for (int freq = 0; freq < freq_bins; ++freq) {
          if (input.dtype() == kComplexFloat32) {
            fft_input[freq] = input.constAt<std::complex<float>>({b, freq, f});
          } else {
            float real = input.constAt<float>({b, freq, f, 0});
            float imag = input.constAt<float>({b, freq, f, 1});
            fft_input[freq] = std::complex<float>(real, imag);
          }
        }

        // reconstruct onesided spectrum if needed
        if (options_.onesided) {
          for (int freq = freq_bins; freq < n_fft; ++freq) {
            int conj_freq = n_fft - freq;
            fft_input[freq] = std::conj(fft_input[conj_freq]);
          }
        }

        // IFFT
        if (is_power_of_2(n_fft)) {
          fftRadix2Complex<float>(fft_input, n_fft, true);  // In-place FFT
        } else {
          fftBluesteinComplex<float>(fft_input, fft_output, n_fft, true);
          fft_input = fft_output;
        }

        // IFFT output (real part) and normalization
        for (int i = 0; i < n_fft; ++i) {
          ifft_output[i] = fft_input[i].real();
          if (normalized) ifft_output[i] /= n_fft;
        }

        // overlap-add
        int start_idx = f * hop_length;
        for (int i = 0; i < win_length; ++i) {
          int out_idx = start_idx + i;
          if (out_idx < signal_length) {
            output_ptr[out_idx] += ifft_output[i] * window_ptr[i];
            window_envelope[out_idx] += window_ptr[i] * window_ptr[i];
          }
        }
      }

      for (int i = 0; i < signal_length; ++i) {
        if (window_envelope[i] > 1e-10f) { output_ptr[i] /= window_envelope[i]; }
      }
    }
  }
}
}  // namespace mllm::cpu