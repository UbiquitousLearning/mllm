/**
 * @file Image.hpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-29
 *
 */
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

  Image resize(int new_w, int new_h);

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