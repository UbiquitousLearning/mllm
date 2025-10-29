// Copyright (c) MLLM Team.
// Licensed under the MIT License.

/**
 * ImageTransform.hpp
 *
 * This header provides a small, extensible transform system tailored for mllm::Image,
 * modeled after the design philosophy of torchvision.transforms. It enables users to
 * compose image processing operations and common post-processing steps used in CV/ML pipelines.
 *
 * Key Concepts (parallels to torchvision):
 * - Transform Operators: Small classes that implement a single, well-defined transformation.
 *   For example, Resize, CenterCrop operate on Image; Normalize operates on Tensor.
 * - Composition: Compose objects allow chaining multiple transforms in order.
 * - Type Flow: Similar to torchvision, some image transforms keep data as image-like objects,
 *   then a "ToTensor" step converts the image to a tensor. Later steps like "Normalize" operate on Tensor.
 *
 * Design Notes:
 * - Image-level transforms implement IImageTransform and return a new Image.
 * - Tensor-level transforms implement ITensorTransform and return a new Tensor.
 * - ToTensor is a bridging transform converting Image(H,W,C, uint8-like) to Tensor(C,H,W, float32)
 *   with values scaled to [0, 1], matching torchvision.transforms.ToTensor semantics.
 * - Normalize implements channel-wise normalization: (x - mean) / std for each channel,
 *   where x is a float32 Tensor in CHW layout.
 * - BasicImageTransform is a convenience pipeline assembling common steps: resize -> optional crop -> to_tensor -> optional
 * normalize.
 *
 * Example (usage, mirrors torchvision style):
 *   // Build a preprocessing pipeline
 *   // mllm::BasicImageTransform tf(
 *   //     std::optional<int>(512),                // resize shorter side to 512
 *   //     std::optional<int>(448),                // center-crop square 448x448
 *   //     std::vector<float>{0.485f, 0.456f, 0.406f}, // mean
 *   //     std::vector<float>{0.229f, 0.224f, 0.225f}  // std
 *   // );
 *
 *   // Apply to an image
 *   // mllm::Image img = mllm::Image::open("/path/to/img.jpg");
 *   // mllm::Tensor input = tf(img); // CHW, float32, normalized
 */

#pragma once

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "mllm/preprocessor/visual/Image.hpp"
#include "mllm/core/Tensor.hpp"

namespace mllm {

// Interface for transforms that take an Image and return an Image.
class IImageTransform {
 public:
  virtual ~IImageTransform() = default;
  [[nodiscard]] virtual Image apply(const Image& input) const = 0;
};

// Interface for transforms that take a Tensor and return a Tensor.
class ITensorTransform {
 public:
  virtual ~ITensorTransform() = default;
  [[nodiscard]] virtual Tensor apply(const Tensor& input) const = 0;
};

// Compose multiple image transforms (executed in order). Returns the final Image.
class ComposeImageTransforms {
 public:
  ComposeImageTransforms() = default;

  explicit ComposeImageTransforms(const std::vector<std::shared_ptr<IImageTransform>>& transforms) : transforms_(transforms) {}

  ComposeImageTransforms& add(const std::shared_ptr<IImageTransform>& t) {
    transforms_.push_back(t);
    return *this;
  }

  [[nodiscard]] Image operator()(const Image& img) const;

 private:
  std::vector<std::shared_ptr<IImageTransform>> transforms_;
};

// Compose multiple tensor transforms (executed in order). Returns the final Tensor.
class ComposeTensorTransforms {
 public:
  ComposeTensorTransforms() = default;

  explicit ComposeTensorTransforms(const std::vector<std::shared_ptr<ITensorTransform>>& transforms)
      : transforms_(transforms) {}

  ComposeTensorTransforms& add(const std::shared_ptr<ITensorTransform>& t) {
    transforms_.push_back(t);
    return *this;
  }

  [[nodiscard]] Tensor operator()(const Tensor& t) const;

 private:
  std::vector<std::shared_ptr<ITensorTransform>> transforms_;
};

// Resize transform.
// TorchVision semantics:
// - If constructed with a single integer `size`, resize so that the shorter side == size,
//   preserving aspect ratio. The longer side is scaled accordingly.
// - If constructed with (height, width), resize to exactly that spatial size.
class Resize : public IImageTransform {
 public:
  // Preserve aspect ratio: shorter side == size.
  explicit Resize(int size_shorter);
  // Explicit target (height, width).
  Resize(int target_h, int target_w);

  [[nodiscard]] Image apply(const Image& input) const override;

 private:
  std::optional<int> size_shorter_;
  std::optional<std::pair<int, int>> size_hw_;
};

// CenterCrop transform.
// TorchVision semantics:
// - Crop a Region of size (crop_h, crop_w) at the image center.
// - If the crop extends beyond boundaries, out-of-bounds is zero-padded (PIL-style); our Image::crop already supports this.
class CenterCrop : public IImageTransform {
 public:
  explicit CenterCrop(int crop_size);
  CenterCrop(int crop_h, int crop_w);

  [[nodiscard]] Image apply(const Image& input) const override;

 private:
  int crop_h_;
  int crop_w_;
};

// ToTensor bridging transform: Image -> Tensor.
// TorchVision semantics:
// - Convert PIL-like image to a float32 tensor with shape (C, H, W).
// - Scale values from [0, 255] to [0, 1].
class ToTensor {
 public:
  [[nodiscard]] Tensor apply(const Image& input) const;
};

// Normalize transform: Tensor -> Tensor.
// TorchVision semantics:
// - Input tensor expected to be float32 in CHW layout.
// - For each channel c: out[c, :, :] = (in[c, :, :] - mean[c]) / std[c].
class Normalize : public ITensorTransform {
 public:
  Normalize(const std::vector<float>& mean, const std::vector<float>& std);

  [[nodiscard]] Tensor apply(const Tensor& input) const override;

 private:
  std::vector<float> mean_;
  std::vector<float> std_;
};

// Convenience pipeline resembling common torchvision usage:
//   transforms = Compose([Resize, (optional) CenterCrop, ToTensor, (optional) Normalize])
// - Users provide parameters commonly used in OCR/vision preprocessing.
// - Returns a Tensor ready for model input.
class BasicImageTransform {
 public:
  // Build a pipeline:
  // - If `resize_shorter` is set, resize by shorter side; otherwise if `resize_hw` is set, resize to (h, w).
  // - If `center_crop` is set, apply center crop.
  // - Always apply ToTensor.
  // - If `norm_mean` and `norm_std` are provided, apply Normalize.
  BasicImageTransform(std::optional<int> resize_shorter, std::optional<std::pair<int, int>> resize_hw,
                      std::optional<std::pair<int, int>> center_crop, std::optional<std::vector<float>> norm_mean,
                      std::optional<std::vector<float>> norm_std);

  // Convenience ctor: shorter-side resize, optional square crop, optional normalize.
  BasicImageTransform(std::optional<int> resize_shorter, std::optional<int> center_crop_square,
                      std::optional<std::vector<float>> norm_mean, std::optional<std::vector<float>> norm_std);

  // Apply the pipeline to an input image and return the final tensor.
  [[nodiscard]] Tensor operator()(const Image& img) const;

 private:
  ComposeImageTransforms image_pipeline_;
  ToTensor to_tensor_;
  ComposeTensorTransforms tensor_pipeline_;
};

}  // namespace mllm
