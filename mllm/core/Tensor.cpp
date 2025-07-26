/**
 * @file Tensor.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-21
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/core/Tensor.hpp"
#include "mllm/engine/Context.hpp"

namespace mllm {

Tensor& Tensor::alloc() {
  Context::instance().memoryManager()->alloc(impl_->storage());
  return *this;
}

Tensor::Tensor(const std::shared_ptr<TensorViewImpl>& impl) : impl_(impl) {
  // TODO Check
}

Tensor Tensor::empty(const std::vector<int32_t>& shape, DataTypes dtype, DeviceTypes device) {
  auto storage = TensorStorage::create(shape, dtype, device);
  auto impl = TensorViewImpl::create(shape, storage);
  return Tensor(impl);
}

Tensor& Tensor::allocExtraTensorView(const std::string& extra_tensor_name, const std::vector<int32_t>& shape, DataTypes dtype,
                                     DeviceTypes device) {
  // TODO
  return *this;
}

Tensor Tensor::getExtraTensorViewInTensor(const std::string& extra_tensor_name) {
  MLLM_RT_ASSERT_EQ(attached_views_.count(extra_tensor_name), 1);
  return Tensor(attached_views_.at(extra_tensor_name));
}

Tensor Tensor::zeros(const std::vector<int32_t>& shape, DataTypes dtype, DeviceTypes device) {
  // TODO
  return Tensor::nil();
}

Tensor Tensor::ones(const std::vector<int32_t>& shape, DataTypes dtype, DeviceTypes device) {
  // TODO
  return Tensor::nil();
}

Tensor Tensor::arange(float start, float end, float step, DataTypes dtype, DeviceTypes device) {
  // TODO
  return Tensor::nil();
}

Tensor Tensor::random(const std::vector<int32_t>& shape, float start, float end, DataTypes dtype, DeviceTypes device) {
  // TODO
  return Tensor::nil();
}

Tensor Tensor::operator+(const Tensor& rhs) {
  // TODO
  return Tensor::nil();
}

Tensor Tensor::operator-(const Tensor& rhs) {
  // TODO
  return Tensor::nil();
}

Tensor Tensor::operator*(const Tensor& rhs) {
  // TODO
  return Tensor::nil();
}

Tensor Tensor::operator/(const Tensor& rhs) {
  // TODO
  return Tensor::nil();
}

Tensor Tensor::operator+(float rhs) {
  // TODO
  return Tensor::nil();
}

Tensor Tensor::operator-(float rhs) {
  // TODO
  return Tensor::nil();
}

Tensor Tensor::operator*(float rhs) {
  // TODO
  return Tensor::nil();
}

Tensor Tensor::operator/(float rhs) {
  // TODO
  return Tensor::nil();
}

Tensor Tensor::operator-() {
  // TODO
  return Tensor::nil();
}

Tensor Tensor::transpose(int dim0, int dim1) {
  // TODO
  return Tensor::nil();
}

Tensor Tensor::T() { return transpose(-1, -2); }

Tensor Tensor::to(DeviceTypes device) {
  if (device == impl_->device()) { return *this; }
  // TODO
  return Tensor::nil();
}

Tensor Tensor::to(DataTypes dtype) {
  if (dtype == impl_->dtype()) { return *this; }
  // TODO
  return Tensor::nil();
}

Tensor Tensor::cpu() {
  if (kCPU == impl_->device()) { return *this; }
  // TODO
  return Tensor::nil();
}

Tensor Tensor::cuda() {
  if (kCUDA == impl_->device()) { return *this; }
  // TODO
  return Tensor::nil();
}

std::string Tensor::name() const { return impl()->name(); }

TensorMemTypes Tensor::memType() const { return impl()->memType(); }

Tensor& Tensor::setMemType(TensorMemTypes mem_type) {
  if (impl_->storage()->mem_type_ != kNormal) {
    MLLM_WARN("You are trying to change a tensor storage whose memory type is not normal. Which "
              "may lead to memory error. Mllm will still change its memory type, but not guarantee "
              "the correctness");
  }
  impl_->storage()->mem_type_ = mem_type;
  return *this;
}

DataTypes Tensor::dtype() const { return impl()->dtype(); }

DeviceTypes Tensor::device() const { return impl()->device(); }

Tensor::shape_t Tensor::shape() const { return impl()->shape(); }

size_t Tensor::numel() const { return impl()->numel(); }

uint32_t Tensor::uuid() const { return impl()->uuid(); }

bool Tensor::isContiguous() const {
  // TODO
  return false;
}

Tensor Tensor::contiguous() {
  // TODO
  return Tensor::nil();
}

Tensor Tensor::reshape(const Tensor::shape_t& shape) {
  // TODO
  return Tensor::nil();
}

Tensor Tensor::view(const Tensor::shape_t& indicies) {
  // TODO
  return Tensor::nil();
}

Tensor Tensor::repeat(int32_t multiplier, int32_t dim) {
  if (multiplier == 1) { return *this; }
  // TODO
  return Tensor::nil();
}

Tensor Tensor::unsqueeze(int32_t dim) {
  auto this_shape = shape();
  this_shape.insert(this_shape.begin() + dim, 1);
  return view(this_shape);
}

Tensor Tensor::clone() {
  // TODO
  return Tensor::nil();
}

Tensor Tensor::permute(const shape_t& indices) {
  // TODO
  return Tensor::nil();
}

size_t Tensor::bytes() { return impl_->size(); }

}  // namespace mllm
