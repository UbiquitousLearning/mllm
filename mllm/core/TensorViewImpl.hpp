// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#define MLLM_TENSOR_SHAPE_MAX_LEN 16

#include <memory>

#include "mllm/utils/Common.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/core/DeviceTypes.hpp"
#include "mllm/core/TensorStorage.hpp"

namespace mllm {

class TensorViewImpl : public std::enable_shared_from_this<TensorViewImpl> {
 public:
  using ptr_t = std::shared_ptr<TensorViewImpl>;
  using storage_t = std::shared_ptr<TensorStorage>;
  using shape_t = std::vector<int32_t>;
  using stride_t = std::vector<int32_t>;
  using dtype_t = DataTypes;
  using device_t = DeviceTypes;

  TensorViewImpl() = default;
  TensorViewImpl(TensorViewImpl&) = delete;
  TensorViewImpl(const TensorViewImpl&) = delete;
  TensorViewImpl(const TensorViewImpl&&) = delete;

  DataTypes dtype() const;

  DeviceTypes device() const;

  storage_t storage() const;

  [[nodiscard]] std::string name() const;

  [[nodiscard]] TensorMemTypes memType() const;

  [[nodiscard]] uint32_t uuid() const;

  uint64_t address() const;

  [[nodiscard]] void* barePtr() const;

  // How many bytes. Not Aligned
  size_t size() const;

  size_t numel() const;

  [[nodiscard]] shape_t shape() const;

  [[nodiscard]] stride_t stride() const;

  bool isContiguous() const;

  bool isContiguousN(int n) const;

  ptr_t clone() const;

  int32_t storageOffset() const;

  // create empty TensorViewImpl
  static ptr_t create();

  // Will automatic calculate stride for you.
  static ptr_t create(const shape_t& shape, const storage_t& storage);

  static ptr_t create(int32_t storage_offset, const shape_t& shape, const storage_t& storage);

  static ptr_t create(int32_t storage_offset, const shape_t& shape, const stride_t& stride, const storage_t& storage);

  template<typename T>
  T* ptr() {
    return (T*)(((char*)(storage_->ptr_))
                + (size_t)((storage_offset_ / lanesOfType(storage_->dtype_)) * bytesOfType(storage_->dtype_)));
  }

  template<typename T>
  T* offsettedPtr(const shape_t& offsets) {
    MLLM_RT_ASSERT_EQ(offsets.size(), shape_len_);

    int32_t _offset = 0;
    for (int i = 0; i < shape_len_; ++i) { _offset += offsets[i] * stride_[i]; }

    return (T*)(ptr<char>() + (_offset / lanesOfType(storage_->dtype_)) * bytesOfType(storage_->dtype_));
  }

  inline void dropStorage() { storage_ = nullptr; }

 private:
  int32_t shape_len_ = 0;
  int32_t storage_offset_ = 0;
  int32_t shape_[MLLM_TENSOR_SHAPE_MAX_LEN];
  int32_t stride_[MLLM_TENSOR_SHAPE_MAX_LEN];
  std::shared_ptr<TensorStorage> storage_ = nullptr;
};

}  // namespace mllm
