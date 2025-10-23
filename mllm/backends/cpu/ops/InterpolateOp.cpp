// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cstring>
#include <cmath>
#include <algorithm>
#include "mllm/backends/cpu/ops/InterpolateOp.hpp"

namespace mllm::cpu {

CPUInterpolateOp::CPUInterpolateOp(const aops::InterpolateOpOptions& options) : aops::InterpolateOp(options) {}

// Helper function to compute scale factors
static void compute_scale_factors(const std::vector<int>& input_size, const std::vector<int>& output_size,
                                  std::vector<float>& scale_factors, bool align_corners) {
  scale_factors.resize(input_size.size());
  for (size_t i = 0; i < input_size.size(); ++i) {
    if (output_size[i] == 1) {
      scale_factors[i] = 0.0f;
    } else if (align_corners) {
      scale_factors[i] = static_cast<float>(input_size[i] - 1) / (output_size[i] - 1);
    } else {
      scale_factors[i] = static_cast<float>(input_size[i]) / output_size[i];
    }
  }
}

// Nearest neighbor interpolation for 2D data (4D tensor with NCHW layout)
template<typename T>
static void nearest_interpolate_2d(const T* input_data, T* output_data, const std::vector<int>& input_shape,
                                   const std::vector<int>& output_shape, bool align_corners) {
  const int batch_size = input_shape[0];
  const int channels = input_shape[1];
  const int input_height = input_shape[2];
  const int input_width = input_shape[3];
  const int output_height = output_shape[2];
  const int output_width = output_shape[3];

  std::vector<float> scale_factors;
  compute_scale_factors({input_height, input_width}, {output_height, output_width}, scale_factors, align_corners);

  const float height_scale = scale_factors[0];
  const float width_scale = scale_factors[1];

  for (int n = 0; n < batch_size; ++n) {
    for (int c = 0; c < channels; ++c) {
      for (int oh = 0; oh < output_height; ++oh) {
        for (int ow = 0; ow < output_width; ++ow) {
          // Compute source indices
          float ih_f = align_corners ? oh * height_scale : (oh + 0.5f) * height_scale - 0.5f;
          float iw_f = align_corners ? ow * width_scale : (ow + 0.5f) * width_scale - 0.5f;

          // Round to nearest neighbor
          int ih = std::min(static_cast<int>(std::round(ih_f)), input_height - 1);
          int iw = std::min(static_cast<int>(std::round(iw_f)), input_width - 1);

          // Handle edge cases
          ih = std::max(0, ih);
          iw = std::max(0, iw);

          // Copy value
          const int input_idx = ((n * channels + c) * input_height + ih) * input_width + iw;
          const int output_idx = ((n * channels + c) * output_height + oh) * output_width + ow;
          output_data[output_idx] = input_data[input_idx];
        }
      }
    }
  }
}

// Linear interpolation for 1D data (3D tensor with NCL layout)
template<typename T>
static void linear_interpolate_1d(const T* input_data, T* output_data, const std::vector<int>& input_shape,
                                  const std::vector<int>& output_shape, bool align_corners) {
  const int batch_size = input_shape[0];
  const int channels = input_shape[1];
  const int input_length = input_shape[2];
  const int output_length = output_shape[2];

  std::vector<float> scale_factors;
  compute_scale_factors({input_length}, {output_length}, scale_factors, align_corners);

  const float length_scale = scale_factors[0];

  for (int n = 0; n < batch_size; ++n) {
    for (int c = 0; c < channels; ++c) {
      for (int ol = 0; ol < output_length; ++ol) {
        // Compute source position
        float il_f = align_corners ? ol * length_scale : (ol + 0.5f) * length_scale - 0.5f;

        // Get the neighboring indices
        int il_low = static_cast<int>(std::floor(il_f));
        int il_high = il_low + 1;

        // Compute weights
        float w_high = il_f - il_low;
        float w_low = 1.0f - w_high;

        // Handle boundary conditions
        il_low = std::max(0, std::min(il_low, input_length - 1));
        il_high = std::max(0, std::min(il_high, input_length - 1));

        // Compute indices
        const int input_idx_low = (n * channels + c) * input_length + il_low;
        const int input_idx_high = (n * channels + c) * input_length + il_high;
        const int output_idx = (n * channels + c) * output_length + ol;

        // Linear interpolation
        output_data[output_idx] = static_cast<T>(w_low * input_data[input_idx_low] + w_high * input_data[input_idx_high]);
      }
    }
  }
}

// Bilinear interpolation for 2D data (4D tensor with NCHW layout)
template<typename T>
static void bilinear_interpolate_2d(const T* input_data, T* output_data, const std::vector<int>& input_shape,
                                    const std::vector<int>& output_shape, bool align_corners) {
  const int batch_size = input_shape[0];
  const int channels = input_shape[1];
  const int input_height = input_shape[2];
  const int input_width = input_shape[3];
  const int output_height = output_shape[2];
  const int output_width = output_shape[3];

  std::vector<float> scale_factors;
  compute_scale_factors({input_height, input_width}, {output_height, output_width}, scale_factors, align_corners);

  const float height_scale = scale_factors[0];
  const float width_scale = scale_factors[1];

  for (int n = 0; n < batch_size; ++n) {
    for (int c = 0; c < channels; ++c) {
      for (int oh = 0; oh < output_height; ++oh) {
        for (int ow = 0; ow < output_width; ++ow) {
          // Compute source position
          float ih_f = align_corners ? oh * height_scale : (oh + 0.5f) * height_scale - 0.5f;
          float iw_f = align_corners ? ow * width_scale : (ow + 0.5f) * width_scale - 0.5f;

          // Get the four neighboring pixels
          int ih_low = static_cast<int>(std::floor(ih_f));
          int iw_low = static_cast<int>(std::floor(iw_f));
          int ih_high = ih_low + 1;
          int iw_high = iw_low + 1;

          // Compute weights
          float h_weight_high = ih_f - ih_low;
          float w_weight_high = iw_f - iw_low;
          float h_weight_low = 1.0f - h_weight_high;
          float w_weight_low = 1.0f - w_weight_high;

          // Handle boundary conditions
          ih_low = std::max(0, std::min(ih_low, input_height - 1));
          ih_high = std::max(0, std::min(ih_high, input_height - 1));
          iw_low = std::max(0, std::min(iw_low, input_width - 1));
          iw_high = std::max(0, std::min(iw_high, input_width - 1));

          // Compute indices for the four corners
          const int idx_top_left = ((n * channels + c) * input_height + ih_low) * input_width + iw_low;
          const int idx_top_right = ((n * channels + c) * input_height + ih_low) * input_width + iw_high;
          const int idx_bottom_left = ((n * channels + c) * input_height + ih_high) * input_width + iw_low;
          const int idx_bottom_right = ((n * channels + c) * input_height + ih_high) * input_width + iw_high;

          // Compute output index
          const int output_idx = ((n * channels + c) * output_height + oh) * output_width + ow;

          // Bilinear interpolation
          output_data[output_idx] = static_cast<T>(h_weight_low * w_weight_low * input_data[idx_top_left]
                                                   + h_weight_low * w_weight_high * input_data[idx_top_right]
                                                   + h_weight_high * w_weight_low * input_data[idx_bottom_left]
                                                   + h_weight_high * w_weight_high * input_data[idx_bottom_right]);
        }
      }
    }
  }
}

// Bicubic interpolation helper function
static float cubic_interp1d(float x0, float x1, float x2, float x3, float t) {
  float a = -0.5f * x0 + 1.5f * x1 - 1.5f * x2 + 0.5f * x3;
  float b = x0 - 2.5f * x1 + 2.0f * x2 - 0.5f * x3;
  float c = -0.5f * x0 + 0.5f * x2;
  float d = x1;

  return ((a * t + b) * t + c) * t + d;
}

// Bicubic interpolation for 2D data (4D tensor with NCHW layout)
template<typename T>
static void bicubic_interpolate_2d(const T* input_data, T* output_data, const std::vector<int>& input_shape,
                                   const std::vector<int>& output_shape, bool align_corners) {
  const int batch_size = input_shape[0];
  const int channels = input_shape[1];
  const int input_height = input_shape[2];
  const int input_width = input_shape[3];
  const int output_height = output_shape[2];
  const int output_width = output_shape[3];

  std::vector<float> scale_factors;
  compute_scale_factors({input_height, input_width}, {output_height, output_width}, scale_factors, align_corners);

  const float height_scale = scale_factors[0];
  const float width_scale = scale_factors[1];

  auto get_value_bounded = [&](int n, int c, int h, int w) -> T {
    h = std::max(0, std::min(h, input_height - 1));
    w = std::max(0, std::min(w, input_width - 1));
    return input_data[((n * channels + c) * input_height + h) * input_width + w];
  };

  for (int n = 0; n < batch_size; ++n) {
    for (int c = 0; c < channels; ++c) {
      for (int oh = 0; oh < output_height; ++oh) {
        for (int ow = 0; ow < output_width; ++ow) {
          // Compute source position
          float ih_f = align_corners ? oh * height_scale : (oh + 0.5f) * height_scale - 0.5f;
          float iw_f = align_corners ? ow * width_scale : (ow + 0.5f) * width_scale - 0.5f;

          // Get the integer part
          int ih = static_cast<int>(std::floor(ih_f));
          int iw = static_cast<int>(std::floor(iw_f));

          // Get fractional part
          float h_frac = ih_f - ih;
          float w_frac = iw_f - iw;

          // Compute output index
          const int output_idx = ((n * channels + c) * output_height + oh) * output_width + ow;

          // Perform bicubic interpolation
          float coeffs[4];

          // Interpolate along each row
          for (int i = 0; i < 4; ++i) {
            float row_values[4];
            for (int j = 0; j < 4; ++j) { row_values[j] = static_cast<float>(get_value_bounded(n, c, ih + i - 1, iw + j - 1)); }
            coeffs[i] = cubic_interp1d(row_values[0], row_values[1], row_values[2], row_values[3], w_frac);
          }

          // Interpolate along column
          float result = cubic_interp1d(coeffs[0], coeffs[1], coeffs[2], coeffs[3], h_frac);

          // Clamp result to avoid overshoot/undershoot
          output_data[output_idx] = static_cast<T>(result);
        }
      }
    }
  }
}

// Trilinear interpolation for 3D data (5D tensor with NCDHW layout)
template<typename T>
static void trilinear_interpolate_3d(const T* input_data, T* output_data, const std::vector<int>& input_shape,
                                     const std::vector<int>& output_shape, bool align_corners) {
  const int batch_size = input_shape[0];
  const int channels = input_shape[1];
  const int input_depth = input_shape[2];
  const int input_height = input_shape[3];
  const int input_width = input_shape[4];
  const int output_depth = output_shape[2];
  const int output_height = output_shape[3];
  const int output_width = output_shape[4];

  std::vector<float> scale_factors;
  compute_scale_factors({input_depth, input_height, input_width}, {output_depth, output_height, output_width}, scale_factors,
                        align_corners);

  const float depth_scale = scale_factors[0];
  const float height_scale = scale_factors[1];
  const float width_scale = scale_factors[2];

  for (int n = 0; n < batch_size; ++n) {
    for (int c = 0; c < channels; ++c) {
      for (int od = 0; od < output_depth; ++od) {
        for (int oh = 0; oh < output_height; ++oh) {
          for (int ow = 0; ow < output_width; ++ow) {
            // Compute source position
            float id_f = align_corners ? od * depth_scale : (od + 0.5f) * depth_scale - 0.5f;
            float ih_f = align_corners ? oh * height_scale : (oh + 0.5f) * height_scale - 0.5f;
            float iw_f = align_corners ? ow * width_scale : (ow + 0.5f) * width_scale - 0.5f;

            // Get the eight neighboring voxels
            int id_low = static_cast<int>(std::floor(id_f));
            int ih_low = static_cast<int>(std::floor(ih_f));
            int iw_low = static_cast<int>(std::floor(iw_f));
            int id_high = id_low + 1;
            int ih_high = ih_low + 1;
            int iw_high = iw_low + 1;

            // Compute weights
            float d_weight_high = id_f - id_low;
            float h_weight_high = ih_f - ih_low;
            float w_weight_high = iw_f - iw_low;
            float d_weight_low = 1.0f - d_weight_high;
            float h_weight_low = 1.0f - h_weight_high;
            float w_weight_low = 1.0f - w_weight_high;

            // Handle boundary conditions
            id_low = std::max(0, std::min(id_low, input_depth - 1));
            id_high = std::max(0, std::min(id_high, input_depth - 1));
            ih_low = std::max(0, std::min(ih_low, input_height - 1));
            ih_high = std::max(0, std::min(ih_high, input_height - 1));
            iw_low = std::max(0, std::min(iw_low, input_width - 1));
            iw_high = std::max(0, std::min(iw_high, input_width - 1));

            // Compute indices for the eight corners
            const int idx_d0_h0_w0 =
                (((n * channels + c) * input_depth + id_low) * input_height + ih_low) * input_width + iw_low;
            const int idx_d0_h0_w1 =
                (((n * channels + c) * input_depth + id_low) * input_height + ih_low) * input_width + iw_high;
            const int idx_d0_h1_w0 =
                (((n * channels + c) * input_depth + id_low) * input_height + ih_high) * input_width + iw_low;
            const int idx_d0_h1_w1 =
                (((n * channels + c) * input_depth + id_low) * input_height + ih_high) * input_width + iw_high;
            const int idx_d1_h0_w0 =
                (((n * channels + c) * input_depth + id_high) * input_height + ih_low) * input_width + iw_low;
            const int idx_d1_h0_w1 =
                (((n * channels + c) * input_depth + id_high) * input_height + ih_low) * input_width + iw_high;
            const int idx_d1_h1_w0 =
                (((n * channels + c) * input_depth + id_high) * input_height + ih_high) * input_width + iw_low;
            const int idx_d1_h1_w1 =
                (((n * channels + c) * input_depth + id_high) * input_height + ih_high) * input_width + iw_high;

            // Compute output index
            const int output_idx = (((n * channels + c) * output_depth + od) * output_height + oh) * output_width + ow;

            // Trilinear interpolation
            output_data[output_idx] =
                static_cast<T>(d_weight_low * h_weight_low * w_weight_low * input_data[idx_d0_h0_w0]
                               + d_weight_low * h_weight_low * w_weight_high * input_data[idx_d0_h0_w1]
                               + d_weight_low * h_weight_high * w_weight_low * input_data[idx_d0_h1_w0]
                               + d_weight_low * h_weight_high * w_weight_high * input_data[idx_d0_h1_w1]
                               + d_weight_high * h_weight_low * w_weight_low * input_data[idx_d1_h0_w0]
                               + d_weight_high * h_weight_low * w_weight_high * input_data[idx_d1_h0_w1]
                               + d_weight_high * h_weight_high * w_weight_low * input_data[idx_d1_h1_w0]
                               + d_weight_high * h_weight_high * w_weight_high * input_data[idx_d1_h1_w1]);
          }
        }
      }
    }
  }
}

void CPUInterpolateOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  const auto& X = inputs[0];
  auto& Y = outputs[0];

  // Get shapes
  const auto& input_shape = X.shape();
  const auto& output_shape = Y.shape();
  const int input_dim = static_cast<int>(input_shape.size());

  // Get options
  const auto& mode = options().mode;
  const bool align_corners = options().align_corners;
  const bool antialias = options().antialias;

  // Allocate output tensor if not already allocated
  if (Y.isNil() || Y.numel() == 0) { Y = Tensor::empty(output_shape, X.dtype(), X.device()).alloc(); }

  switch (X.dtype()) {
    case kFloat32: {
      const float* input_data = X.ptr<float>();
      float* output_data = Y.ptr<float>();

      // Gaussian blur utility (separable) for NCHW
      auto gaussian_blur_nchw = [&](const float* src, int N, int C, int H, int W, float sigma) {
        int radius = std::max(1, static_cast<int>(std::ceil(3.f * sigma)));
        std::vector<float> kernel(2 * radius + 1);
        float sumw = 0.f;
        for (int i = -radius; i <= radius; ++i) {
          float w = std::exp(-(i * i) / (2.f * sigma * sigma));
          kernel[i + radius] = w;
          sumw += w;
        }
        for (float& w : kernel) { w /= sumw; }

        size_t numel = static_cast<size_t>(N) * C * H * W;
        std::vector<float> tmp(numel, 0.f);
        std::vector<float> dst(numel, 0.f);

        // Horizontal pass
        for (int n = 0; n < N; ++n) {
          for (int c = 0; c < C; ++c) {
            for (int h = 0; h < H; ++h) {
              for (int w = 0; w < W; ++w) {
                float acc = 0.f;
                for (int k = -radius; k <= radius; ++k) {
                  int x = std::min(std::max(w + k, 0), W - 1);
                  size_t idx = ((static_cast<size_t>(n) * C + c) * H + h) * W + x;
                  acc += kernel[k + radius] * src[idx];
                }
                size_t oidx = ((static_cast<size_t>(n) * C + c) * H + h) * W + w;
                tmp[oidx] = acc;
              }
            }
          }
        }
        // Vertical pass
        for (int n = 0; n < N; ++n) {
          for (int c = 0; c < C; ++c) {
            for (int h = 0; h < H; ++h) {
              for (int w = 0; w < W; ++w) {
                float acc = 0.f;
                for (int k = -radius; k <= radius; ++k) {
                  int y = std::min(std::max(h + k, 0), H - 1);
                  size_t idx = ((static_cast<size_t>(n) * C + c) * H + y) * W + w;
                  acc += kernel[k + radius] * tmp[idx];
                }
                size_t oidx = ((static_cast<size_t>(n) * C + c) * H + h) * W + w;
                dst[oidx] = acc;
              }
            }
          }
        }
        return dst;
      };

      // Choose interpolation method based on mode and input dimensions
      if (mode == aops::InterpolateOpMode::kNearest) {
        if (input_dim == 3) {  // NCL format
          nearest_interpolate_2d<float>(input_data, output_data, {input_shape[0], input_shape[1], 1, input_shape[2]},
                                        {output_shape[0], output_shape[1], 1, output_shape[2]}, align_corners);
        } else if (input_dim == 4) {  // NCHW format
          nearest_interpolate_2d<float>(input_data, output_data, input_shape, output_shape, align_corners);
        } else if (input_dim == 5) {  // NCDHW format
          // For 3D data, we handle each depth slice separately using 2D nearest neighbor
          const int batch_size = input_shape[0];
          const int channels = input_shape[1];
          const int input_depth = input_shape[2];
          const int output_depth = output_shape[2];

          std::vector<float> scale_factors;
          compute_scale_factors({input_depth}, {output_depth}, scale_factors, align_corners);
          const float depth_scale = scale_factors[0];

          for (int od = 0; od < output_depth; ++od) {
            // Compute source depth index
            float id_f = align_corners ? od * depth_scale : (od + 0.5f) * depth_scale - 0.5f;
            int id = std::min(static_cast<int>(std::round(id_f)), input_depth - 1);
            id = std::max(0, id);

            // Process each 2D slice
            for (int n = 0; n < batch_size; ++n) {
              for (int c = 0; c < channels; ++c) {
                const float* input_slice =
                    input_data + (((n * channels + c) * input_depth + id) * input_shape[3] * input_shape[4]);
                float* output_slice =
                    output_data + (((n * channels + c) * output_depth + od) * output_shape[3] * output_shape[4]);

                nearest_interpolate_2d<float>(input_slice, output_slice, {1, 1, input_shape[3], input_shape[4]},
                                              {1, 1, output_shape[3], output_shape[4]}, align_corners);
              }
            }
          }
        } else {
          NYI("CPUInterpolateOp::forward nearest mode not support input dim {}", input_dim);
        }
      } else if (mode == aops::InterpolateOpMode::kLinear) {
        if (input_dim == 3) {  // NCL format
          linear_interpolate_1d<float>(input_data, output_data, input_shape, output_shape, align_corners);
        } else {
          NYI("CPUInterpolateOp::forward linear mode only supports 3D input (NCL format)");
        }
      } else if (mode == aops::InterpolateOpMode::kBilinear) {
        if (input_dim == 4) {  // NCHW format
          // Antialias for downsampling
          std::vector<float> scale_factors;
          compute_scale_factors({input_shape[2], input_shape[3]}, {output_shape[2], output_shape[3]}, scale_factors,
                                align_corners);
          const float h_scale = scale_factors[0];
          const float w_scale = scale_factors[1];
          const float* src_ptr = input_data;
          std::vector<float> blurred;
          if (antialias && (h_scale > 1.f || w_scale > 1.f)) {
            float sigma = 0.5f * std::max(h_scale, w_scale);
            blurred = gaussian_blur_nchw(input_data, input_shape[0], input_shape[1], input_shape[2], input_shape[3], sigma);
            src_ptr = blurred.data();
          }
          bilinear_interpolate_2d<float>(src_ptr, output_data, input_shape, output_shape, align_corners);
        } else {
          NYI("CPUInterpolateOp::forward bilinear mode only supports 4D input (NCHW format)");
        }
      } else if (mode == aops::InterpolateOpMode::kBicubic) {
        if (input_dim == 4) {  // NCHW format
          // Antialias for downsampling
          std::vector<float> scale_factors;
          compute_scale_factors({input_shape[2], input_shape[3]}, {output_shape[2], output_shape[3]}, scale_factors,
                                align_corners);
          const float h_scale = scale_factors[0];
          const float w_scale = scale_factors[1];
          const float* src_ptr = input_data;
          std::vector<float> blurred;
          if (antialias && (h_scale > 1.f || w_scale > 1.f)) {
            float sigma = 0.5f * std::max(h_scale, w_scale);
            blurred = gaussian_blur_nchw(input_data, input_shape[0], input_shape[1], input_shape[2], input_shape[3], sigma);
            src_ptr = blurred.data();
          }
          bicubic_interpolate_2d<float>(src_ptr, output_data, input_shape, output_shape, align_corners);
        } else {
          NYI("CPUInterpolateOp::forward bicubic mode only supports 4D input (NCHW format)");
        }
      } else if (mode == aops::InterpolateOpMode::kTrilinear) {
        if (input_dim == 5) {  // NCDHW format
          trilinear_interpolate_3d<float>(input_data, output_data, input_shape, output_shape, align_corners);
        } else {
          NYI("CPUInterpolateOp::forward trilinear mode only supports 5D input (NCDHW format)");
        }
      } else {
        NYI("CPUInterpolateOp::forward unknown interpolation mode");
      }
      break;
    }
    default: NYI("CPUInterpolateOp::forward not support dtype {}", nameOfType(X.dtype())); break;
  }
}

}  // namespace mllm::cpu
