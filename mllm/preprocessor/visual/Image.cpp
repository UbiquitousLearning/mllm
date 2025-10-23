// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <algorithm>
#include <filesystem>

#include "mllm/utils/Common.hpp"
#include "mllm/preprocessor/visual/Image.hpp"

namespace mllm {

_ImagePtr::~_ImagePtr() { stbi_image_free(ptr_); }

Image Image::open(const std::string& fp) {
  Image ret_image;

  if (!stbi_info(fp.c_str(), &ret_image.w_, &ret_image.h_, &ret_image.c_)) {
    MLLM_ERROR_EXIT(ExitCode::kIOError, "Can't get infomation of image: {}", fp);
  }

  if (ret_image.c_ >= 4) {
    MLLM_WARN("Found image: {} has {} channel. MLLM Image::open method will force convert this "
              "image to RGB.",
              fp, ret_image.c_);
  }

  auto _ptr = stbi_load(fp.c_str(), &ret_image.w_, &ret_image.h_, &ret_image.c_, 3);

  if (_ptr) {
    ret_image.image_ptr_ = std::make_shared<_ImagePtr>();
    ret_image.image_ptr_->ptr_ = _ptr;
    ret_image.c_ = 3;
    MLLM_INFO("Load image: {} success.", fp);
    MLLM_INFO("Image size: {}x{} in RGB format.", ret_image.h_, ret_image.w_);
  } else {
    MLLM_ERROR_EXIT(ExitCode::kIOError, "stbi_load load error. Found i_data is empty pointer!");
  }

  return ret_image;
}

Image Image::resize(int new_w, int new_h) {
  Image new_img;
  new_img.w_ = new_w;
  new_img.h_ = new_h;
  new_img.c_ = 3;

  unsigned char* output_data = nullptr;

  // stb will alloc memory for us
  output_data = stbir_resize_uint8_linear(static_cast<unsigned char*>(image_ptr_->ptr_), w_, h_, 0, output_data, new_w, new_h,
                                          0, STBIR_RGB);

  new_img.image_ptr_ = std::make_shared<_ImagePtr>();
  new_img.image_ptr_->ptr_ = output_data;
  MLLM_RT_ASSERT(output_data != nullptr);
  return new_img;
}

void Image::save(const std::string& fp) {
  namespace fs = std::filesystem;

  fs::path file_path(fp);

  std::string ext = file_path.extension().string();

  if (!ext.empty() && ext[0] == '.') { ext = ext.substr(1); }
  std::ranges::transform(ext, ext.begin(), ::tolower);

  if (ext == "png") {
    stbi_write_png(fp.c_str(), w_, h_, c_, image_ptr_->ptr_, w_ * c_);
  } else if (ext == "jpg" || ext == "jpeg") {
    stbi_write_jpg(fp.c_str(), w_, h_, c_, image_ptr_->ptr_, 95);
  } else if (ext == "bmp") {
    stbi_write_bmp(fp.c_str(), w_, h_, c_, image_ptr_->ptr_);
  } else {
    MLLM_ERROR_EXIT(ExitCode::kIOError, "Unsupported image format: {}", ext);
  }
}

Tensor Image::tensor() {
  MLLM_RT_ASSERT_EQ(c_, 3);
  auto ret_tensor = Tensor::empty({h_, w_, c_}, kFloat32, kCPU).alloc();

  auto bare_tensor_ptr = ret_tensor.ptr<float>();
  auto bare_stb_ptr = static_cast<unsigned char*>(image_ptr_->ptr_);

  std::copy(bare_stb_ptr, bare_stb_ptr + ret_tensor.numel(), bare_tensor_ptr);

  return ret_tensor;
}

unsigned char* Image::ptr() {
  if (!image_ptr_) return nullptr;
  return static_cast<unsigned char*>(image_ptr_->ptr_);
}

int Image::w() { return w_; }

int Image::h() { return h_; }

int Image::c() { return c_; }

Image Image::crop(int left, int upper, int right, int lower) {
  // Validate input dimensions and ensure source image is loaded
  MLLM_RT_ASSERT(image_ptr_ != nullptr);
  MLLM_RT_ASSERT(right > left && lower > upper);

  const int crop_w = right - left;
  const int crop_h = lower - upper;

  Image new_img;
  new_img.w_ = crop_w;
  new_img.h_ = crop_h;
  new_img.c_ = 3;  // Force RGB, consistent with Image::open

  // Allocate output buffer; stbi_image_free uses free, so malloc is compatible
  unsigned char* output = static_cast<unsigned char*>(malloc(static_cast<size_t>(crop_w) * crop_h * new_img.c_));
  MLLM_RT_ASSERT(output != nullptr);

  const unsigned char* src = static_cast<const unsigned char*>(image_ptr_->ptr_);

  // PIL-style crop: pad out-of-bounds with zeros
  for (int y = 0; y < crop_h; ++y) {
    const int sy = upper + y;
    for (int x = 0; x < crop_w; ++x) {
      const int sx = left + x;
      unsigned char* dst_px = output + (static_cast<size_t>(y) * crop_w + x) * new_img.c_;
      if (sx >= 0 && sx < w_ && sy >= 0 && sy < h_) {
        const unsigned char* src_px = src + (static_cast<size_t>(sy) * w_ + sx) * c_;
        dst_px[0] = src_px[0];
        dst_px[1] = src_px[1];
        dst_px[2] = src_px[2];
      } else {
        dst_px[0] = 0;
        dst_px[1] = 0;
        dst_px[2] = 0;
      }
    }
  }

  new_img.image_ptr_ = std::make_shared<_ImagePtr>();
  new_img.image_ptr_->ptr_ = output;

  return new_img;
}

}  // namespace mllm