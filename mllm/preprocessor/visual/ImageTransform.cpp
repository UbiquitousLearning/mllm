// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <algorithm>
#include <cmath>

#include "mllm/preprocessor/visual/ImageTransform.hpp"
#include "mllm/utils/Common.hpp"

namespace mllm {

// ========================= ComposeImageTransforms =========================
Image ComposeImageTransforms::operator()(const Image& img) const {
  Image current = img;  // make a mutable copy for non-const member calls
  for (const auto& t : transforms_) { current = t->apply(current); }
  return current;
}

// ========================= ComposeTensorTransforms =========================
Tensor ComposeTensorTransforms::operator()(const Tensor& t) const {
  Tensor current = t;  // make a mutable copy for non-const member calls
  for (const auto& tr : transforms_) { current = tr->apply(current); }
  return current;
}

// ========================= Resize =========================
Resize::Resize(int size_shorter) : size_shorter_(size_shorter) {}
Resize::Resize(int target_h, int target_w) : size_hw_(std::make_pair(target_h, target_w)) {}

Image Resize::apply(const Image& input) const {
  Image src = input;  // mutable copy
  if (size_hw_.has_value()) {
    // Explicit resize to (H, W)
    const int new_h = size_hw_->first;
    const int new_w = size_hw_->second;
    return src.resize(new_w, new_h);  // Image::resize expects (w, h)
  }

  // Shorter-side resize with aspect ratio preserved
  MLLM_RT_ASSERT(size_shorter_.has_value());
  const int target_shorter = *size_shorter_;
  const int h = src.h();
  const int w = src.w();

  const int shorter = std::min(h, w);
  const double scale = static_cast<double>(target_shorter) / static_cast<double>(shorter);
  const int new_h = static_cast<int>(std::round(h * scale));
  const int new_w = static_cast<int>(std::round(w * scale));
  return src.resize(new_w, new_h);
}

// ========================= CenterCrop =========================
CenterCrop::CenterCrop(int crop_size) : crop_h_(crop_size), crop_w_(crop_size) {}
CenterCrop::CenterCrop(int crop_h, int crop_w) : crop_h_(crop_h), crop_w_(crop_w) {}

Image CenterCrop::apply(const Image& input) const {
  Image src = input;  // mutable copy
  const int h = src.h();
  const int w = src.w();

  // Compute centered crop box, PIL-style
  const int left = (w - crop_w_) / 2;
  const int upper = (h - crop_h_) / 2;
  const int right = left + crop_w_;
  const int lower = upper + crop_h_;

  return src.crop(left, upper, right, lower);
}

// ========================= ToTensor =========================
Tensor ToTensor::apply(const Image& input) const {
  Image src = input;  // mutable copy
  // Image::tensor returns HWC float32 tensor with values in [0, 255]
  Tensor t = src.tensor();

  // Reorder to CHW to match torchvision semantics
  t = t.permute({2, 0, 1});

  // Scale to [0, 1]
  t = t / 255.0f;
  return t;
}

// ========================= Normalize =========================
Normalize::Normalize(const std::vector<float>& mean, const std::vector<float>& std) : mean_(mean), std_(std) {
  MLLM_RT_ASSERT_EQ(mean_.size(), std_.size());
}

Tensor Normalize::apply(const Tensor& input) const {
  const Tensor& src = input;
  // Expect src in CHW layout
  MLLM_RT_ASSERT(src.rank() == 3);
  const int c = src.size(0);
  const int h = src.size(1);
  const int w = src.size(2);
  MLLM_RT_ASSERT_EQ(static_cast<int>(mean_.size()), c);
  MLLM_RT_ASSERT_EQ(static_cast<int>(std_.size()), c);

  // Asuming Work on a contiguous clone to simplify indexing
  float* ptr = input.ptr<float>();
  const size_t plane = static_cast<size_t>(h) * static_cast<size_t>(w);

  for (int ch = 0; ch < c; ++ch) {
    const float m = mean_[ch];
    const float s = std_[ch];

    float* base = ptr + static_cast<size_t>(ch) * plane;
    for (size_t i = 0; i < plane; ++i) { base[i] = (base[i] - m) / s; }
  }

  return input;
}

// ========================= BasicImageTransform =========================
BasicImageTransform::BasicImageTransform(std::optional<int> resize_shorter, std::optional<std::pair<int, int>> resize_hw,
                                         std::optional<std::pair<int, int>> center_crop,
                                         std::optional<std::vector<float>> norm_mean,
                                         std::optional<std::vector<float>> norm_std) {
  // Build image pipeline
  if (resize_shorter.has_value()) {
    image_pipeline_.add(std::make_shared<Resize>(*resize_shorter));
  } else if (resize_hw.has_value()) {
    image_pipeline_.add(std::make_shared<Resize>(resize_hw->first, resize_hw->second));
  }

  if (center_crop.has_value()) { image_pipeline_.add(std::make_shared<CenterCrop>(center_crop->first, center_crop->second)); }

  // Build tensor pipeline (Normalize optional)
  if (norm_mean.has_value() && norm_std.has_value()) {
    tensor_pipeline_.add(std::make_shared<Normalize>(*norm_mean, *norm_std));
  }
}

BasicImageTransform::BasicImageTransform(std::optional<int> resize_shorter, std::optional<int> center_crop_square,
                                         std::optional<std::vector<float>> norm_mean,
                                         std::optional<std::vector<float>> norm_std) {
  if (resize_shorter.has_value()) { image_pipeline_.add(std::make_shared<Resize>(*resize_shorter)); }
  if (center_crop_square.has_value()) { image_pipeline_.add(std::make_shared<CenterCrop>(*center_crop_square)); }
  if (norm_mean.has_value() && norm_std.has_value()) {
    tensor_pipeline_.add(std::make_shared<Normalize>(*norm_mean, *norm_std));
  }
}

Tensor BasicImageTransform::operator()(const Image& img) const {
  // 1) Run image-level transforms
  Image processed = image_pipeline_(img);
  // 2) Convert to tensor (CHW, [0,1])
  Tensor t = to_tensor_.apply(processed);
  // 3) Run tensor-level transforms
  t = tensor_pipeline_(t);
  return t;
}

}  // namespace mllm
