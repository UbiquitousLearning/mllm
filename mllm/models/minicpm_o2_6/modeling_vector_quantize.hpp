// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/mllm.hpp"
#include "mllm/nn/Functional.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/nn/Nn.hpp"
#include "mllm/utils/AnyValue.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/utils/Log.hpp"
#include <cstdint>
#include <string>

// Finite Scalar Quantization: VQ-VAE Made Simple - https://arxiv.org/abs/2309.15505
// follows vector_quantize_pytorch in python

namespace mllm::models::vq {
// Round with straight-through estimator
Tensor round_ste(const Tensor& z) {
  // Simple implementation: just round the values
  // In a full implementation, this would preserve gradients
  auto result = Tensor::zeros(z.shape(), z.dtype(), z.device());
  auto z_ptr = z.ptr<float>();
  auto result_ptr = result.ptr<float>();
  size_t numel = z.numel();

  for (size_t i = 0; i < numel; ++i) { result_ptr[i] = std::round(z_ptr[i]); }

  return result;
}

// Create tensor from vector values
Tensor create_levels_tensor(const std::vector<int32_t>& levels) {
  auto tensor = Tensor::zeros({static_cast<int32_t>(levels.size())});
  auto ptr = tensor.ptr<float>();
  for (size_t i = 0; i < levels.size(); ++i) { ptr[i] = static_cast<float>(levels[i]); }
  return tensor;
}

// Create basis tensor: cumulative product of [1] + levels[:-1]
Tensor create_basis_tensor(const std::vector<int32_t>& levels) {
  std::vector<int32_t> basis_vec = {1};
  for (size_t i = 0; i < levels.size() - 1; ++i) { basis_vec.push_back(levels[i]); }

  // Compute cumulative product
  for (size_t i = 1; i < basis_vec.size(); ++i) { basis_vec[i] *= basis_vec[i - 1]; }

  auto tensor = Tensor::zeros({static_cast<int32_t>(basis_vec.size())});
  auto ptr = tensor.ptr<float>();
  for (size_t i = 0; i < basis_vec.size(); ++i) { ptr[i] = static_cast<float>(basis_vec[i]); }
  return tensor;
}

/**
 * FSQ: Finite Scalar Quantization
 * Main implementation based on "Finite Scalar Quantization: VQ-VAE Made Simple"
 */
class FSQ final : public nn::Module {
  std::vector<int32_t> levels_;
  Tensor _levels;  // Tensor version of levels
  Tensor _basis;   // Cumulative product basis for indexing

  int32_t codebook_dim_;
  int32_t effective_codebook_dim_;
  int32_t num_codebooks_;
  int32_t dim_;

  bool keep_num_codebooks_dim_;
  bool channel_first_;
  bool return_indices_;
  bool force_quantization_f32_;

  float scale_;

  // Projection layers
  nn::Linear project_in_;
  nn::Linear project_out_;

 public:
  // Make these public for ResidualFSQ access
  bool has_projections_;
  int32_t codebook_size_;
  Tensor implicit_codebook_;

  FSQ() = default;

  /**
   * @brief Construct FSQ module
   * @param name Module name
   * @param levels List of quantization levels for each dimension
   * @param dim Input/output dimension (if None, uses len(levels) * num_codebooks)
   * @param num_codebooks Number of codebooks (for multi-codebook quantization)
   * @param keep_num_codebooks_dim Whether to keep codebook dimension in output
   * @param scale Scaling factor for quantization
   * @param channel_first Whether input is channel-first format
   * @param projection_has_bias Whether projection layers have bias
   * @param return_indices Whether to return quantized indices
   * @param force_quantization_f32 Whether to force f32 precision for quantization
   */
  FSQ(const std::string& name, const std::vector<int32_t>& levels, int32_t dim = -1, int32_t num_codebooks = 1,
      bool keep_num_codebooks_dim = false, float scale = 1.0f, bool channel_first = false, bool projection_has_bias = true,
      bool return_indices = true, bool force_quantization_f32 = true)
      : nn::Module(name),
        levels_(levels),
        num_codebooks_(num_codebooks),
        scale_(scale),
        channel_first_(channel_first),
        return_indices_(return_indices),
        force_quantization_f32_(force_quantization_f32) {
    // Create levels and basis tensors
    _levels = create_levels_tensor(levels);
    _basis = create_basis_tensor(levels);

    codebook_dim_ = static_cast<int32_t>(levels.size());
    effective_codebook_dim_ = codebook_dim_ * num_codebooks;

    // Set keep_num_codebooks_dim default
    if (keep_num_codebooks_dim == false && num_codebooks > 1) {
      keep_num_codebooks_dim_ = true;
    } else {
      keep_num_codebooks_dim_ = keep_num_codebooks_dim;
    }

    // Set default dim
    dim_ = (dim == -1) ? effective_codebook_dim_ : dim;

    has_projections_ = (dim_ != effective_codebook_dim_);

    // Initialize projection layers
    if (has_projections_) {
      project_in_ = reg<nn::Linear>("project_in", dim_, effective_codebook_dim_, projection_has_bias);
      project_out_ = reg<nn::Linear>("project_out", effective_codebook_dim_, dim_, projection_has_bias);
    }

    // Calculate codebook size and create implicit codebook
    if (return_indices_) {
      codebook_size_ = 1;
      for (int32_t level : levels_) { codebook_size_ *= level; }

      // Create implicit codebook: indices_to_codes for all possible indices
      implicit_codebook_ = createImplicitCodebook();
    }
  }

 private:
  /**
   * Create implicit codebook by mapping all possible indices to codes
   */
  Tensor createImplicitCodebook() {
    auto indices = Tensor::arange(0, static_cast<float>(codebook_size_), 1, kFloat32, kCPU);
    return indicesToCodes(indices);
  }

  /**
   * Bound input tensor to quantization range
   * @param z Input tensor [..., d]
   * @param eps Small epsilon for numerical stability
   * @return Bounded tensor
   */
  [[nodiscard]] Tensor bound(const Tensor& z, float eps = 1e-3f) const {
    // Simplified version without tanh/atanh - just clip to valid range
    auto result = Tensor::zeros(z.shape(), z.dtype(), z.device());
    auto z_ptr = z.ptr<float>();
    auto result_ptr = result.ptr<float>();

    size_t numel = z.numel();
    auto levels_ptr = _levels.ptr<float>();

    for (size_t i = 0; i < numel; ++i) {
      size_t level_idx = i % codebook_dim_;  // Cycle through levels for each dimension
      float level = levels_ptr[level_idx];
      float half_l = (level - 1) * (1 + eps) / 2;
      float offset = (static_cast<int>(level) % 2 == 0) ? 0.5f : 0.0f;

      // Simple clipping instead of tanh/atanh
      float val = std::max(-half_l + offset, std::min(half_l + offset, z_ptr[i]));
      result_ptr[i] = val;
    }

    return result;
  }

 public:
  /**
   * Quantize input tensor using FSQ
   * @param z Input tensor to quantize
   * @return Quantized tensor (normalized to [-1, 1])
   */
  [[nodiscard]] Tensor quantize(const Tensor& z) const {
    auto quantized = round_ste(bound(z));

    // Renormalize to [-1, 1]: quantized / half_width
    auto result = Tensor::zeros(quantized.shape(), quantized.dtype(), quantized.device());
    auto quantized_ptr = quantized.ptr<float>();
    auto result_ptr = result.ptr<float>();
    auto levels_ptr = _levels.ptr<float>();

    size_t numel = quantized.numel();
    for (size_t i = 0; i < numel; ++i) {
      size_t level_idx = i % codebook_dim_;
      float half_width = levels_ptr[level_idx] / 2.0f;
      result_ptr[i] = quantized_ptr[i] / half_width;
    }

    return result;
  }

  /**
   * Scale and shift normalized codes to level indices
   */
  [[nodiscard]] Tensor scaleAndShift(const Tensor& zhat_normalized) const {
    auto result = Tensor::zeros(zhat_normalized.shape(), zhat_normalized.dtype(), zhat_normalized.device());
    auto input_ptr = zhat_normalized.ptr<float>();
    auto result_ptr = result.ptr<float>();
    auto levels_ptr = _levels.ptr<float>();

    size_t numel = zhat_normalized.numel();
    for (size_t i = 0; i < numel; ++i) {
      size_t level_idx = i % codebook_dim_;
      int32_t half_width = static_cast<int32_t>(levels_ptr[level_idx]) / 2;
      result_ptr[i] = (input_ptr[i] * half_width) + half_width;
    }

    return result;
  }

  /**
   * Inverse of scale and shift
   */
  [[nodiscard]] Tensor scaleAndShiftInverse(const Tensor& zhat) const {
    auto result = Tensor::zeros(zhat.shape(), zhat.dtype(), zhat.device());
    auto input_ptr = zhat.ptr<float>();
    auto result_ptr = result.ptr<float>();
    auto levels_ptr = _levels.ptr<float>();

    size_t numel = zhat.numel();
    for (size_t i = 0; i < numel; ++i) {
      size_t level_idx = i % codebook_dim_;
      int32_t half_width = static_cast<int32_t>(levels_ptr[level_idx]) / 2;
      result_ptr[i] = (input_ptr[i] - half_width) / half_width;
    }

    return result;
  }

  /**
   * Convert indices to level indices at each quantization level
   */
  [[nodiscard]] Tensor indicesToLevelIndices(const Tensor& indices) const {
    auto shape = indices.shape();
    std::vector<int32_t> new_shape = shape;
    new_shape.push_back(codebook_dim_);

    auto result = Tensor::zeros(new_shape, kFloat32, indices.device());
    auto indices_ptr = indices.ptr<float>();
    auto result_ptr = result.ptr<float>();
    auto basis_ptr = _basis.ptr<float>();
    auto levels_ptr = _levels.ptr<float>();

    size_t input_numel = indices.numel();
    for (size_t i = 0; i < input_numel; ++i) {
      int32_t idx = static_cast<int32_t>(indices_ptr[i]);
      for (int32_t d = 0; d < codebook_dim_; ++d) {
        int32_t basis = static_cast<int32_t>(basis_ptr[d]);
        int32_t level = static_cast<int32_t>(levels_ptr[d]);
        int32_t code = (idx / basis) % level;
        result_ptr[i * codebook_dim_ + d] = static_cast<float>(code);
      }
    }

    return result;
  }

  /**
   * Convert indices to codes (inverse of codes_to_indices)
   */
  [[nodiscard]] Tensor indicesToCodes(const Tensor& indices) const {
    auto level_indices = indicesToLevelIndices(indices);
    return scaleAndShiftInverse(level_indices);
  }

  /**
   * Convert codes to indices in codebook
   */
  [[nodiscard]] Tensor codesToIndices(const Tensor& zhat) const {
    // Ensure last dimension matches codebook_dim
    auto shape = zhat.shape();
    if (shape.back() != codebook_dim_) { MLLM_ERROR("Expected last dimension {}, got {}", codebook_dim_, shape.back()); }

    auto zhat_shifted = scaleAndShift(zhat);

    // Calculate indices = (zhat * basis).sum(dim=-1)
    auto input_ptr = zhat_shifted.ptr<float>();
    auto basis_ptr = _basis.ptr<float>();

    std::vector<int32_t> result_shape = shape;
    result_shape.pop_back();  // Remove last dimension

    auto result = Tensor::zeros(result_shape, kFloat32, zhat.device());
    auto result_ptr = result.ptr<float>();

    size_t result_numel = result.numel();
    for (size_t i = 0; i < result_numel; ++i) {
      int32_t sum = 0;
      for (int32_t d = 0; d < codebook_dim_; ++d) {
        sum += static_cast<int32_t>(input_ptr[i * codebook_dim_ + d] * basis_ptr[d]);
      }
      result_ptr[i] = static_cast<float>(sum);
    }

    return result;
  }

  /**
   * Forward pass
   */
  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    NYI("NOT VERIFIED");
    auto z = inputs[0];

    bool is_img_or_video = z.shape().size() >= 4;
    bool need_move_channel_last = is_img_or_video || channel_first_;

    std::vector<int32_t> original_shape;

    // Standardize to (batch, seq, dimension) format
    if (need_move_channel_last) {
      original_shape = z.shape();
      // Rearrange from 'b d ...' to 'b ... d'
      z = z.transpose(1, -1);
      // Flatten spatial dimensions: 'b ... d' -> 'b * d'
      auto shape = z.shape();
      int32_t batch = shape[0];
      int32_t total_spatial = 1;
      for (size_t i = 1; i < shape.size() - 1; ++i) { total_spatial *= shape[i]; }
      z = z.view({batch, total_spatial, shape.back()});
    }

    // Verify input dimension
    auto z_shape = z.shape();
    if (z_shape.back() != dim_) { MLLM_ERROR("Expected input dimension {}, got {}", dim_, z_shape.back()); }

    // Project input if needed
    if (has_projections_) { z = project_in_(z); }

    // Reshape for multiple codebooks: 'b n (c d)' -> 'b n c d'
    if (num_codebooks_ > 1) {
      auto shape = z.shape();
      z = z.view({shape[0], shape[1], num_codebooks_, codebook_dim_});
    }

    // Quantization (potentially in f32)
    Tensor codes;
    if (force_quantization_f32_ && z.dtype() != kFloat32) {
      auto z_f32 = z.to(kFloat32);
      codes = quantize(z_f32);
      codes = codes.to(z.dtype());  // Convert back to original dtype
    } else {
      codes = quantize(z);
    }

    // Get indices if needed
    Tensor indices;
    if (return_indices_) {
      if (num_codebooks_ > 1) {
        // Handle multiple codebooks: compute indices for each codebook
        std::vector<Tensor> indices_list;
        for (int c = 0; c < num_codebooks_; ++c) {
          // Extract codes for codebook c - simplified slicing
          auto codes_shape = codes.shape();
          auto codes_c = Tensor::zeros({codes_shape[0], codes_shape[1], codebook_dim_}, codes.dtype(), codes.device());

          // Copy data for codebook c
          auto codes_ptr = codes.ptr<float>();
          auto codes_c_ptr = codes_c.ptr<float>();
          size_t batch_seq = codes_shape[0] * codes_shape[1];
          for (size_t i = 0; i < batch_seq; ++i) {
            for (int32_t d = 0; d < codebook_dim_; ++d) {
              codes_c_ptr[i * codebook_dim_ + d] = codes_ptr[i * (num_codebooks_ * codebook_dim_) + c * codebook_dim_ + d];
            }
          }

          indices_list.push_back(codesToIndices(codes_c));
        }
        indices = nn::functional::concat(indices_list, 2);  // Concat along last dim
      } else {
        indices = codesToIndices(codes);
      }
    }

    // Reshape codes back: 'b n c d' -> 'b n (c d)'
    if (num_codebooks_ > 1) {
      auto shape = codes.shape();
      codes = codes.view({shape[0], shape[1], shape[2] * shape[3]});
    }

    // Project output if needed
    Tensor out = has_projections_ ? project_out_(codes) : codes;

    // Restore original shape if needed
    if (need_move_channel_last) {
      // Reshape back to original spatial dimensions
      auto out_shape = out.shape();
      std::vector<int32_t> target_shape = {out_shape[0]};
      for (size_t i = 1; i < original_shape.size() - 1; ++i) { target_shape.push_back(original_shape[i]); }
      target_shape.push_back(out_shape[2]);
      out = out.view(target_shape);

      // Move channel back to front: 'b ... d' -> 'b d ...'
      out = out.transpose(1, -1);

      // Handle indices too if they exist
      if (return_indices_) {
        auto indices_shape = indices.shape();
        std::vector<int32_t> indices_target_shape = {indices_shape[0]};
        for (size_t i = 1; i < original_shape.size() - 1; ++i) { indices_target_shape.push_back(original_shape[i]); }
        if (keep_num_codebooks_dim_) { indices_target_shape.push_back(num_codebooks_); }
        indices = indices.view(indices_target_shape);
      }
    }

    // Remove codebook dimension if not keeping it
    if (return_indices_ && !keep_num_codebooks_dim_ && num_codebooks_ == 1) {
      auto indices_shape = indices.shape();
      if (!indices_shape.empty()) {
        indices_shape.pop_back();
        indices = indices.view(indices_shape);
      }
    }

    if (return_indices_) {
      return {out, indices};
    } else {
      return {out};
    }
  }
};

/**
 * ResidualFSQ: Residual Finite Scalar Quantization
 * Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf
 * A quantization method that uses residual quantization with finite scalar quantization
 */
class ResidualFSQ final : public nn::Module {
  std::vector<int32_t> levels_;
  int32_t num_quantizers_;
  int32_t dim_;
  int32_t codebook_dim_;
  bool is_channel_first_;
  bool has_projections_;
  bool quantize_dropout_;
  int32_t quantize_dropout_cutoff_index_;
  int32_t quantize_dropout_multiple_of_;

  // Projection layers
  nn::Linear project_in_;
  nn::Linear project_out_;

  // FSQ layers - one for each quantizer
  nn::ModuleList<FSQ> layers_;

  // Scales buffer - stored as parameter for each quantizer: (levels - 1) ** -ind
  Tensor scales_;

 public:
  int32_t codebook_size_;

  ResidualFSQ() = default;

  /**
   * @brief Construct a ResidualFSQ module
   * @param name Module name
   * @param levels List of quantization levels for each dimension
   * @param num_quantizers Number of quantizers in the residual chain
   * @param dim Input/output dimension (if None, uses codebook_dim)
   * @param is_channel_first Whether input is channel-first format
   * @param quantize_dropout Whether to enable quantize dropout during training
   * @param quantize_dropout_cutoff_index Minimum index for quantize dropout
   * @param quantize_dropout_multiple_of Multiple for structured dropout
   */
  ResidualFSQ(const std::string& name, const std::vector<int32_t>& levels, int32_t num_quantizers, int32_t dim = -1,
              bool is_channel_first = false, bool quantize_dropout = false, int32_t quantize_dropout_cutoff_index = 0,
              int32_t quantize_dropout_multiple_of = 1)
      : nn::Module(name),
        levels_(levels),
        num_quantizers_(num_quantizers),
        is_channel_first_(is_channel_first),
        quantize_dropout_(quantize_dropout && num_quantizers > 1),
        quantize_dropout_cutoff_index_(quantize_dropout_cutoff_index),
        quantize_dropout_multiple_of_(quantize_dropout_multiple_of) {
    codebook_dim_ = static_cast<int32_t>(levels_.size());
    dim_ = (dim == -1) ? codebook_dim_ : dim;

    has_projections_ = (codebook_dim_ != dim_);

    // Initialize projection layers
    if (has_projections_) {
      project_in_ = reg<nn::Linear>("project_in", dim_, codebook_dim_);
      project_out_ = reg<nn::Linear>("project_out", codebook_dim_, dim_);
    }

    // Initialize FSQ layers - each one should NOT have projections (codebook_dim -> codebook_dim)
    layers_ = reg<nn::ModuleList<FSQ>>("fsq_", num_quantizers_, levels, codebook_dim_, 1, /*num_codebooks*/
                                       false,                                             /*keep_num_codebooks_fim*/
                                       1.0f,                                              /*scale*/
                                       false,                                             /*channel_first*/
                                       true,                                              /*projection_has_bias*/
                                       true,                                              /*return_indices*/
                                       true);                                             /*force_quantization_f32*/

    // Verify no FSQ layer has projections
    for (const auto& fsq : layers_.list()) {
      if (fsq.has_projections_) { MLLM_ERROR("FSQ layer should not have projections in ResidualFSQ"); }
    }

    // Get codebook size from first layer
    codebook_size_ = layers_.list()[0].codebook_size_;

    // Calculate scales: (levels - 1) ** -ind and stack them
    std::vector<Tensor> scales_list;
    scales_list.reserve(num_quantizers_);

    for (int ind = 0; ind < num_quantizers_; ++ind) {
      auto scale_tensor = Tensor::zeros({codebook_dim_}, kFloat32, kCPU);
      auto scale_data = scale_tensor.ptr<float>();

      for (int d = 0; d < codebook_dim_; ++d) {
        float level = static_cast<float>(levels_[d]);
        scale_data[d] = std::pow(level - 1.0f, -static_cast<float>(ind));
      }

      scales_list.push_back(scale_tensor);
    }

    // Stack scales: [num_quantizers, codebook_dim]
    scales_ = nn::functional::concat(scales_list, 0).view({num_quantizers_, codebook_dim_});
  }

  /**
   * Get codebooks from all FSQ layers
   * @return Tensor of shape [num_quantizers, codebook_size, codebook_dim]
   */
  [[nodiscard]] Tensor getCodebooks() {
    std::vector<Tensor> codebooks;
    codebooks.reserve(num_quantizers_);

    for (const auto& layer : layers_.list()) { codebooks.push_back(layer.implicit_codebook_); }

    return nn::functional::concat(codebooks, 0).view({num_quantizers_, codebook_size_, codebook_dim_});
  }

  /**
   * Get codes from quantized indices
   * @param indices Quantized indices [B, N, Q] where Q is num_quantizers
   * @return Tensor of shape [Q, B, N, D] containing codes for each quantizer
   */
  [[nodiscard]] Tensor getCodesFromIndices(Tensor& indices) {
    auto batch = indices.shape()[0];
    auto quantize_dim = indices.shape()[indices.shape().size() - 1];

    // Get codebooks from all FSQ layers
    auto codebooks = getCodebooks();  // [num_quantizers_, codebook_size_, codebook_dim_]

    // Reshape indices to 'b * q' format
    auto indices_reshaped = indices.view({batch, -1, num_quantizers_});

    if (quantize_dim < num_quantizers_) {
      // Check if quantize_dropout > 0
      if (!quantize_dropout_) {
        MLLM_ERROR(
            "quantize dropout must be greater than 0 if you wish to reconstruct from a signal with less fine quantizations");
      }

      // Pad indices with -1 values
      std::vector<int> padded_shape = {batch, indices_reshaped.shape()[1], num_quantizers_};
      Tensor padded_indices = Tensor::ones(padded_shape, indices.dtype(), indices.device());
      padded_indices = padded_indices * (-1.0f);  // Fill with -1

      // Copy original indices to padded tensor
      auto padded_ptr = padded_indices.ptr<float>();
      auto indices_ptr = indices_reshaped.ptr<float>();
      size_t copy_elements = batch * indices_reshaped.shape()[1] * quantize_dim;

      for (size_t i = 0; i < copy_elements; ++i) { padded_ptr[i] = indices_ptr[i]; }

      indices_reshaped = padded_indices;
    }

    // Handle dropout masking
    // Replace -1 indices with 0 (to fetch dummy code that will be masked out)
    auto indices_ptr = indices_reshaped.ptr<float>();
    size_t numel = indices_reshaped.numel();
    for (size_t i = 0; i < numel; ++i) {
      if (indices_ptr[i] == -1.0f) { indices_ptr[i] = 0; }
    }

    // Gather codes from codebooks based on indices
    // Using gather operation: get_at('q [c] d, b n q -> q b n d', codebooks, indices_for_gather)
    // This means we gather from codebooks[q] using indices_for_gather[b,n,q] to get codes[q,b,n,d]
    Tensor all_codes = Tensor::zeros({num_quantizers_, batch, indices_reshaped.shape()[1], codebooks.shape()[2]},
                                     codebooks.dtype(), codebooks.device());

    auto codebooks_ptr = codebooks.ptr<float>();
    auto indices_ptr_gather = indices_reshaped.ptr<float>();
    auto codes_ptr = all_codes.ptr<float>();

    const int codebook_size = codebooks.shape()[1];
    const int codebook_dim = codebooks.shape()[2];
    const int seq_len = indices_reshaped.shape()[1];

    // Perform the gather operation manually
    for (int q = 0; q < num_quantizers_; ++q) {
      for (int b = 0; b < batch; ++b) {
        for (int n = 0; n < seq_len; ++n) {
          // Get index for this position
          int idx = static_cast<int>(indices_ptr_gather[b * seq_len * num_quantizers_ + n * num_quantizers_ + q]);

          // Ensure index is within bounds
          idx = std::max(0, std::min(idx, codebook_size - 1));

          // Copy codebook entry to output
          for (int d = 0; d < codebook_dim; ++d) {
            int codebook_offset = q * codebook_size * codebook_dim + idx * codebook_dim + d;
            int codes_offset = q * batch * seq_len * codebook_dim + b * seq_len * codebook_dim + n * codebook_dim + d;
            codes_ptr[codes_offset] = codebooks_ptr[codebook_offset];
          }

          // If this was a masked entry, zero it out
          if (indices_ptr[b * seq_len * num_quantizers_ + n * num_quantizers_ + q] == 0.0f) {
            for (int d = 0; d < codebook_dim; ++d) {
              int codes_offset = q * batch * seq_len * codebook_dim + b * seq_len * codebook_dim + n * codebook_dim + d;
              codes_ptr[codes_offset] = 0.0f;
            }
          }
        }
      }
    }

    // Apply scales
    auto temp_scales = Tensor(scales_);
    auto scales_reshaped = temp_scales.view({num_quantizers_, 1, 1, -1});  // q 1 1 d

    // Broadcast multiply: all_codes * scales
    all_codes = all_codes * scales_reshaped;

    return all_codes;
  }

  /**
   * Get output from indices by summing across quantizers
   * @param indices Quantized indices
   * @return Reconstructed output tensor
   */
  [[nodiscard]] Tensor getOutputFromIndices(Tensor& indices) {
    auto codes = getCodesFromIndices(indices);  // [Q, B, N, D]

    // Sum across quantizers: reduce(codes, 'q ... -> ...', 'sum')
    auto codes_summed = nn::functional::sum(codes, 0, false);  // Sum along quantizer dimension [B, N, D]

    return project_out_(codes_summed);
  }

  /**
   * Forward pass with residual quantization
   */
  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    NYI("NOT VERIFIED");

    auto x = inputs[0];
    bool return_all_codes = args.empty() ? false : std::any_cast<bool>(args[0]);

    std::vector<int32_t> original_shape;
    std::vector<int32_t> pack_shape;

    // Handle channel first format
    if (is_channel_first_) {
      original_shape = x.shape();
      // Rearrange from 'b d ...' to 'b ... d'
      x = x.transpose(1, -1);
      // Flatten spatial dimensions: 'b ... d' -> 'b * d'
      auto shape = x.shape();
      int32_t batch = shape[0];
      int32_t total_spatial = 1;
      for (size_t i = 1; i < shape.size() - 1; ++i) { total_spatial *= shape[i]; }
      pack_shape = {batch, total_spatial, shape.back()};
      x = x.view(pack_shape);
    }

    // Apply input projection if needed
    if (has_projections_) { x = project_in_(x); }

    // Initialize quantization variables
    auto quantized_out = Tensor::zeros(x.shape(), x.dtype(), x.device());
    auto residual = Tensor(x);  // Copy input

    std::vector<Tensor> all_indices;
    all_indices.reserve(num_quantizers_);

    // Quantize dropout logic (simplified - no random for now)
    bool should_quantize_dropout = false;                   // In training mode would check this.training && quantize_dropout_
    int32_t rand_quantize_dropout_index = num_quantizers_;  // No dropout by default

    if (should_quantize_dropout) {
      // In real implementation, would sample random index
      rand_quantize_dropout_index = quantize_dropout_cutoff_index_;

      if (quantize_dropout_multiple_of_ != 1) {
        // Round up to multiple
        rand_quantize_dropout_index =
            ((rand_quantize_dropout_index + quantize_dropout_multiple_of_) / quantize_dropout_multiple_of_)
                * quantize_dropout_multiple_of_
            - 1;
      }
    }

    // Null indices for dropout
    Tensor null_indices;
    if (should_quantize_dropout) {
      auto x_shape = x.shape();
      null_indices = Tensor::zeros({x_shape[0], x_shape[1]}, kFloat32, x.device()) - 1.0f;  // Fill with -1
    }

    // Go through the layers
    for (int quantizer_index = 0; quantizer_index < num_quantizers_; ++quantizer_index) {
      // Check quantize dropout
      if (should_quantize_dropout && quantizer_index > rand_quantize_dropout_index) {
        all_indices.push_back(null_indices);
        continue;
      }

      // Get scale for this quantizer
      auto scale_slice = scales_.view({num_quantizers_, codebook_dim_});
      auto scale_data = scale_slice.ptr<float>();
      auto scale_tensor = Tensor::zeros({codebook_dim_}, kFloat32, x.device());
      auto scale_ptr = scale_tensor.ptr<float>();

      // Copy scale values for this quantizer
      for (int32_t d = 0; d < codebook_dim_; ++d) { scale_ptr[d] = scale_data[quantizer_index * codebook_dim_ + d]; }

      // Scale residual: residual / scale (broadcast division)
      auto scaled_residual = residual / scale_tensor;

      // Quantize using FSQ layer
      auto layer_output = const_cast<FSQ&>(layers_.list()[quantizer_index]).forward({scaled_residual}, {});
      auto quantized = layer_output[0];
      const auto& indices = layer_output[1];

      // Scale back: quantized * scale
      quantized = quantized * scale_tensor;

      // Update residual (no detach in C++)
      residual = residual - quantized;
      quantized_out = quantized_out + quantized;

      all_indices.push_back(indices);
    }

    // Project output if needed
    if (has_projections_) { quantized_out = project_out_(quantized_out); }

    // Stack all indices along last dimension
    auto stacked_indices = nn::functional::concat(all_indices, -1);

    // Handle channel first output format
    if (is_channel_first_) {
      // Restore original shape
      quantized_out = quantized_out.view({original_shape[0], original_shape.back()});
      for (size_t i = 1; i < original_shape.size() - 1; ++i) {
        quantized_out = quantized_out.view({original_shape[0], -1, original_shape.back()});
      }

      // Move channel back to front: 'b ... d' -> 'b d ...'
      quantized_out = quantized_out.transpose(1, -1);

      // Handle indices too
      stacked_indices = stacked_indices.view({original_shape[0], -1, num_quantizers_});
      stacked_indices = stacked_indices.transpose(1, -1);
    }

    // Prepare return values
    std::vector<Tensor> ret = {quantized_out, stacked_indices};

    if (return_all_codes) {
      auto all_codes = getCodesFromIndices(stacked_indices);
      ret.push_back(all_codes);
    }

    return ret;
  }
};

/**
 * GroupedResidualFSQ: Grouped Residual Finite Scalar Quantization
 * Groups the input channels and applies ResidualFSQ to each group independently
 */
class GroupedResidualFSQ final : public nn::Module {
  int32_t dim_;
  int32_t groups_;
  bool accept_image_fmap_;
  int32_t dim_per_group_;
  int32_t codebook_size_;

  // List of ResidualFSQ modules for each group
  nn::ModuleList<ResidualFSQ> rvqs_;

 public:
  GroupedResidualFSQ() = default;

  GroupedResidualFSQ(const std::string& name, int32_t dim, const std::vector<int32_t>& levels, int32_t num_quantizers,
                     int32_t groups = 1, bool accept_image_fmap = false)
      : nn::Module(name), dim_(dim), groups_(groups), accept_image_fmap_(accept_image_fmap) {
    // Ensure dim is divisible by groups
    if (dim % groups != 0) { MLLM_ERROR("dim must be divisible by groups"); }

    dim_per_group_ = dim / groups;

    // Create ResidualFSQ modules for each group
    rvqs_ = reg<nn::ModuleList<ResidualFSQ>>("rvqs", groups, levels, num_quantizers, dim_per_group_);

    // All groups should have same codebook size
    codebook_size_ = rvqs_.list()[0].codebook_size_;
  }

  /**
   * Get the split dimension for tensor operations
   */
  [[nodiscard]] int32_t getSplitDim() const {
    return accept_image_fmap_ ? 1 : -1;  // For image: split on channel dim, for others: last dim
  }

  /**
   * Get output from indices by reconstructing from all groups
   */
  Tensor getOutputFromIndices(const Tensor& indices) {
    std::vector<Tensor> codes;
    codes.reserve(groups_);
    auto chunk_indices = nn::functional::split(groups_, indices, 0);

    for (int i = 0; i < groups_; ++i) {
      // Each RVQ processes its corresponding chunk of indices
      codes.emplace_back(rvqs_.list()[i].getOutputFromIndices(chunk_indices[i]));  // Remove second argument
    }

    return nn::functional::concat(codes, (getSplitDim() == -1 ? codes[0].shape().size() - 1 : 1));
  }

  /**
   * Forward pass with grouped residual quantization
   */
  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    NYI("NOT VERIFIED");
    const auto& x = inputs[0];
    bool return_all_codes = args.empty() ? false : std::any_cast<bool>(args[0]);

    auto shape = x.shape();
    auto split_dim = getSplitDim();

    // Verify input dimension matches expected
    if (shape[split_dim] != dim_) { MLLM_ERROR("Input dimension mismatch: expected {}, got {}", dim_, shape[split_dim]); }

    auto x_chunks = nn::functional::chunk(groups_, x, split_dim);

    // Process each chunk with its corresponding RVQ
    std::vector<Tensor> quantized_chunks;
    std::vector<Tensor> all_indices;
    std::vector<Tensor> maybe_all_codes;

    for (int i = 0; i < groups_; ++i) {
      // Forward through RVQ
      auto out = rvqs_.list()[i](x_chunks[i], AnyValue(return_all_codes));

      quantized_chunks.push_back(out[0]);  // quantized output
      all_indices.push_back(out[1]);       // indices

      if (return_all_codes && out.size() > 2) {
        maybe_all_codes.push_back(out[2]);  // all codes
      }
    }

    // Combine results from all groups
    auto quantized = nn::functional::concat(quantized_chunks, split_dim);
    auto stacked_indices = nn::functional::concat(all_indices, split_dim);

    if (return_all_codes) {
      auto all_codes = nn::functional::concat(maybe_all_codes, split_dim);
      return {quantized, stacked_indices, all_codes};
    } else {
      return {quantized, stacked_indices};
    }
  }
};

}  // namespace mllm::models::vq