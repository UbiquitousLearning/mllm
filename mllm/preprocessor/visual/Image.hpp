// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

#define STB_IMAGE_RESIZE_STATIC
#define STB_IMAGE_RESIZE2_IMPLEMENTATION
#include <stb/stb_image_resize2.h>

#include "mllm/core/Tensor.hpp"

namespace mllm {

struct _ImagePtr {
  ~_ImagePtr();
  void* ptr_;
};

class Image {
 public:
  static Image open(const std::string& fp);

  Image resize(int new_w, int new_h, const std::string& method = "bilinear");

  // Crop the image with PIL-style box (left, upper, right, lower).
  // Out-of-bounds areas are padded with zeros. Returns a new Image.
  Image crop(int left, int upper, int right, int lower);

  // Pad the image to target size (target_w, target_h) with RGB color.
  // Mirrors PIL ImageOps.pad: scale to fit, then center-pad with color.
  Image pad(int target_w, int target_h, unsigned char r, unsigned char g, unsigned char b);

  void save(const std::string& fp);

  Tensor tensor();

  unsigned char* ptr();

  int w();

  int h();

  int c();

 private:
  int w_;
  int h_;
  int c_;
  std::shared_ptr<_ImagePtr> image_ptr_ = nullptr;
};

}  // namespace mllm
