// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/TensorViewImpl.hpp"

namespace mllm {

DataTypes TensorViewImpl::dtype() const { return storage_->dtype_; }

DeviceTypes TensorViewImpl::device() const { return storage_->device_; }

TensorViewImpl::storage_t TensorViewImpl::storage() const { return storage_; }

std::string TensorViewImpl::name() const { return storage_->name_; }

TensorMemTypes TensorViewImpl::memType() const { return storage_->mem_type_; }

uint32_t TensorViewImpl::uuid() const { return storage_->custom_32bit_uuid_; }

uint64_t TensorViewImpl::address() const { return (uint64_t)(storage_->ptr_); }

void* TensorViewImpl::barePtr() const {
  return (void*)(((char*)storage_->ptr_)
                 + (size_t)((storage_offset_ / lanesOfType(storage_->dtype_)) * bytesOfType(storage_->dtype_)));
}

size_t TensorViewImpl::size() const { return storage_->size_; }

size_t TensorViewImpl::numel() const {
  size_t cnt = 1;
  for (int i = 0; i < shape_len_; ++i) cnt *= shape_[i];
  return cnt;
}

TensorViewImpl::shape_t TensorViewImpl::shape() const { return {shape_, shape_ + shape_len_}; }

TensorViewImpl::stride_t TensorViewImpl::stride() const { return {stride_, stride_ + shape_len_}; }

bool TensorViewImpl::isContiguous() const {
  if (shape_len_ == 0) return true;

  // Check stride
  int expected_stride = 1;
  for (int i = shape_len_ - 1; i >= 0; --i) {
    if (stride_[i] != expected_stride) { return false; }
    expected_stride *= shape_[i];
  }

  return true;
}

bool TensorViewImpl::isContiguousN(int n) const {
  if (n <= 0) return true;
  if (shape_len_ == 0) return true;
  if (n > shape_len_) n = shape_len_;

  // Check stride for the last n dimensions
  int expected_stride = 1;
  for (int i = shape_len_ - 1; i >= shape_len_ - n; --i) {
    if (stride_[i] != expected_stride) { return false; }
    expected_stride *= shape_[i];
  }

  return true;
}

TensorViewImpl::ptr_t TensorViewImpl::clone() const {
  auto ret = TensorViewImpl::create();

  ret->shape_len_ = shape_len_;
  for (int i = 0; i < shape_len_; ++i) {
    ret->shape_[i] = shape_[i];
    ret->stride_[i] = stride_[i];
  }
  ret->storage_offset_ = storage_offset_;
  ret->storage_ = storage_;

  return ret;
}

int32_t TensorViewImpl::storageOffset() const { return storage_offset_; }

// Create empty TensorViewImpl
TensorViewImpl::ptr_t TensorViewImpl::create() { return std::make_shared<TensorViewImpl>(); }

// Will automatic calculate stride for you.
TensorViewImpl::ptr_t TensorViewImpl::create(const TensorViewImpl::shape_t& shape, const TensorViewImpl::storage_t& storage) {
  auto ret = std::make_shared<TensorViewImpl>();
  ret->shape_len_ = shape.size();
  ret->storage_offset_ = 0;

  int _cnt = 0;
  for (unsigned int it : shape) { ret->shape_[_cnt++] = (int32_t)it; }
  int _acc = 1;
  ret->stride_[ret->shape_len_ - 1] = 1;
  for (int i = ret->shape_len_ - 1; i > 0; i--) {
    ret->stride_[i - 1] = _acc * ret->shape_[i];
    _acc *= ret->shape_[i];
  }
  ret->storage_ = storage;

  return ret;
}

TensorViewImpl::ptr_t TensorViewImpl::create(int32_t storage_offset, const TensorViewImpl::shape_t& shape,
                                             const TensorViewImpl::storage_t& storage) {
  auto ret = std::make_shared<TensorViewImpl>();
  ret->shape_len_ = shape.size();
  ret->storage_offset_ = storage_offset;

  int _cnt = 0;
  for (unsigned int it : shape) { ret->shape_[_cnt++] = (int32_t)it; }
  int _acc = 1;
  ret->stride_[ret->shape_len_ - 1] = 1;
  for (int i = ret->shape_len_ - 1; i > 0; i--) {
    ret->stride_[i - 1] = _acc * ret->shape_[i];
    _acc *= ret->shape_[i];
  }
  ret->storage_ = storage;

  return ret;
}

TensorViewImpl::ptr_t TensorViewImpl::create(int32_t storage_offset, const TensorViewImpl::shape_t& shape,
                                             const TensorViewImpl::stride_t& stride, const TensorViewImpl::storage_t& storage) {
  auto ret = std::make_shared<TensorViewImpl>();
  ret->shape_len_ = shape.size();
  ret->storage_offset_ = storage_offset;

  int _cnt = 0;
  for (unsigned int it : shape) { ret->shape_[_cnt++] = (int32_t)it; }
  _cnt = 0;
  for (unsigned int it : stride) { ret->stride_[_cnt++] = (int32_t)it; }
  ret->storage_ = storage;

  return ret;
}

}  // namespace mllm
