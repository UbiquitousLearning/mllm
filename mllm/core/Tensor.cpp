// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <optional>

#include "mllm/core/Tensor.hpp"
#include "mllm/core/aops/CastTypeOp.hpp"
#include "mllm/core/aops/CloneOp.hpp"
#include "mllm/core/aops/ContiguousOp.hpp"
#include "mllm/core/aops/CopyOp.hpp"
#include "mllm/core/aops/ElewiseOps.hpp"
#include "mllm/core/aops/FillOp.hpp"
#include "mllm/core/aops/IndexOp.hpp"
#include "mllm/core/aops/PermuteOp.hpp"
#include "mllm/core/aops/ReduceOps.hpp"
#include "mllm/core/aops/RepeatOp.hpp"
#include "mllm/core/aops/ReshapeOp.hpp"
#include "mllm/core/aops/SliceOp.hpp"
#include "mllm/core/aops/TransposeOp.hpp"
#include "mllm/core/aops/ViewOp.hpp"
#include "mllm/core/aops/X2XOp.hpp"
#include "mllm/engine/Context.hpp"

namespace mllm {

void Tensor::operator delete(void* ptr) noexcept {
  ((Tensor*)ptr)->impl_.reset();
  for (auto& [a, _] : ((Tensor*)ptr)->attached_views_) { ((Tensor*)ptr)->attached_views_[a].reset(); }
}

Tensor Tensor::operator[](const SliceIndices& slice_index) const {
  return Context::instance().buildOpAndSubmitTask(OpTypes::kSlice,
                                                  aops::SliceOpOptions{
                                                      .indices_ = slice_index,
                                                  },
                                                  {*this})[0];
}

Tensor Tensor::operator[](const ComplexIndexingList& complex_indexing) const {
  return Context::instance().buildOpAndSubmitTask(OpTypes::kIndex,
                                                  aops::IndexOpOptions{
                                                      .indices_ = complex_indexing,
                                                  },
                                                  {*this})[0];
}

Tensor& Tensor::alloc() {
  Context::instance().memoryManager()->alloc(impl_->storage());
  return *this;
}

Tensor::Tensor(const std::shared_ptr<TensorViewImpl>& impl) : impl_(impl) {}

Tensor Tensor::empty(const std::vector<int32_t>& shape, DataTypes dtype, DeviceTypes device) {
  auto storage = TensorStorage::create(shape, dtype, device);
  auto impl = TensorViewImpl::create(shape, storage);
  return Tensor(impl);
}

Tensor& Tensor::allocExtraTensorView(const std::string& extra_tensor_name, const std::vector<int32_t>& shape, DataTypes dtype,
                                     DeviceTypes device) {
  MLLM_RT_ASSERT_EQ(attached_views_.count(extra_tensor_name), 0);
  auto storage = TensorStorage::create(shape, dtype, device);
  auto impl = TensorViewImpl::create(shape, storage);
  attached_views_.insert({extra_tensor_name, impl});
  return *this;
}

Tensor Tensor::getExtraTensorViewInTensor(const std::string& extra_tensor_name) {
  MLLM_RT_ASSERT_EQ(attached_views_.count(extra_tensor_name), 1);
  return Tensor(attached_views_.at(extra_tensor_name));
}

Tensor Tensor::zeros(const std::vector<int32_t>& shape, DataTypes dtype, DeviceTypes device) {
  auto i = Tensor::empty(shape, dtype, device).alloc();
  return Context::instance().buildOpAndSubmitTask(OpTypes::kFill, aops::FillOpOptions{.type = aops::FillOpTypes::kZeros},
                                                  {i})[0];
}

Tensor Tensor::ones(const std::vector<int32_t>& shape, DataTypes dtype, DeviceTypes device) {
  auto i = Tensor::empty(shape, dtype, device).alloc();
  return Context::instance().buildOpAndSubmitTask(OpTypes::kFill, aops::FillOpOptions{.type = aops::FillOpTypes::kOnes},
                                                  {i})[0];
}

Tensor Tensor::arange(float start, float end, float step, DataTypes dtype, DeviceTypes device) {
  auto shape = std::vector<int32_t>{static_cast<int32_t>((end - start) / step)};
  auto i = Tensor::empty(shape, dtype, device).alloc();
  return Context::instance().buildOpAndSubmitTask(
      OpTypes::kFill, aops::FillOpOptions{.type = aops::FillOpTypes::kArange, .start = start, .end = end, .step = step},
      {i})[0];
}

Tensor Tensor::random(const std::vector<int32_t>& shape, float start, float end, DataTypes dtype, DeviceTypes device) {
  auto i = Tensor::empty(shape, dtype, device).alloc();
  return Context::instance().buildOpAndSubmitTask(
      OpTypes::kFill,
      aops::FillOpOptions{
          .type = aops::FillOpTypes::kRandom, .start = start, .end = end, .seed = Context::instance().getRandomSeed()},
      {i})[0];
}

Tensor Tensor::operator+(const Tensor& rhs) {
  return Context::instance().buildOpAndSubmitTask(OpTypes::kAdd, aops::AddOpOptions{}, {*this, rhs})[0];
}

Tensor Tensor::operator-(const Tensor& rhs) {
  return Context::instance().buildOpAndSubmitTask(OpTypes::kSub, aops::SubOpOptions{}, {*this, rhs})[0];
}

Tensor Tensor::operator*(const Tensor& rhs) {
  return Context::instance().buildOpAndSubmitTask(OpTypes::kMul, aops::MulOpOptions{}, {*this, rhs})[0];
}

Tensor Tensor::operator/(const Tensor& rhs) {
  return Context::instance().buildOpAndSubmitTask(OpTypes::kDiv, aops::DivOpOptions{}, {*this, rhs})[0];
}

Tensor Tensor::operator+(float rhs) {
  auto rhs_tensor = Tensor::empty({1}, dtype(), device()).alloc();
  switch (dtype()) {
    case kFloat32: *(rhs_tensor.ptr<float>()) = rhs; break;
    case kFloat16: *(rhs_tensor.ptr<half_float::half>()) = half_float::half(rhs); break;
    case kInt32: *(rhs_tensor.ptr<int32_t>()) = rhs; break;
    case kInt16: *(rhs_tensor.ptr<int16_t>()) = rhs; break;
    case kInt8: *(rhs_tensor.ptr<int8_t>()) = rhs; break;
    default: NYI("Type is not supported"); break;
  }
  return Context::instance().buildOpAndSubmitTask(OpTypes::kAdd, aops::AddOpOptions{}, {*this, rhs_tensor})[0];
}

Tensor Tensor::operator-(float rhs) {
  auto rhs_tensor = Tensor::empty({1}, dtype(), device()).alloc();
  switch (dtype()) {
    case kFloat32: *(rhs_tensor.ptr<float>()) = rhs; break;
    case kFloat16: *(rhs_tensor.ptr<half_float::half>()) = half_float::half(rhs); break;
    case kInt32: *(rhs_tensor.ptr<int32_t>()) = rhs; break;
    case kInt16: *(rhs_tensor.ptr<int16_t>()) = rhs; break;
    case kInt8: *(rhs_tensor.ptr<int8_t>()) = rhs; break;
    default: NYI("Type is not supported"); break;
  }
  return Context::instance().buildOpAndSubmitTask(OpTypes::kSub, aops::SubOpOptions{}, {*this, rhs_tensor})[0];
}

Tensor Tensor::operator*(float rhs) {
  auto rhs_tensor = Tensor::empty({1}, dtype(), device()).alloc();
  switch (dtype()) {
    case kFloat32: *(rhs_tensor.ptr<float>()) = rhs; break;
    case kFloat16: *(rhs_tensor.ptr<half_float::half>()) = half_float::half(rhs); break;
    case kInt32: *(rhs_tensor.ptr<int32_t>()) = rhs; break;
    case kInt16: *(rhs_tensor.ptr<int16_t>()) = rhs; break;
    case kInt8: *(rhs_tensor.ptr<int8_t>()) = rhs; break;
    default: NYI("Type is not supported"); break;
  }
  return Context::instance().buildOpAndSubmitTask(OpTypes::kMul, aops::MulOpOptions{}, {*this, rhs_tensor})[0];
}

Tensor Tensor::operator/(float rhs) {
  auto rhs_tensor = Tensor::empty({1}, dtype(), device()).alloc();
  switch (dtype()) {
    case kFloat32: *(rhs_tensor.ptr<float>()) = rhs; break;
    case kFloat16: *(rhs_tensor.ptr<half_float::half>()) = half_float::half(rhs); break;
    case kInt32: *(rhs_tensor.ptr<int32_t>()) = rhs; break;
    case kInt16: *(rhs_tensor.ptr<int16_t>()) = rhs; break;
    case kInt8: *(rhs_tensor.ptr<int8_t>()) = rhs; break;
    default: NYI("Type is not supported"); break;
  }
  return Context::instance().buildOpAndSubmitTask(OpTypes::kDiv, aops::DivOpOptions{}, {*this, rhs_tensor})[0];
}

Tensor Tensor::abs() {
  return Context::instance().buildOpAndSubmitTask(OpTypes::kAbs, aops::AbsOpOptions{}, {*this})[0];
}

Tensor Tensor::min(bool keep_dim, int32_t dim) {
  return Context::instance().buildOpAndSubmitTask(OpTypes::kReduceMin,
                                                  aops::ReduceMinOpOptions{.dim = dim, .keep_dim = keep_dim}, {*this})[0];
}

Tensor Tensor::max(bool keep_dim, int32_t dim) {
  return Context::instance().buildOpAndSubmitTask(OpTypes::kReduceMax,
                                                  aops::ReduceMaxOpOptions{.dim = dim, .keep_dim = keep_dim}, {*this})[0];
}

Tensor Tensor::sum(bool keep_dim, int32_t dim) {
  return Context::instance().buildOpAndSubmitTask(OpTypes::kReduceSum,
                                                  aops::ReduceSumOpOptions{.dim = dim, .keep_dim = keep_dim}, {*this})[0];
}
Tensor Tensor::operator-() { return Context::instance().buildOpAndSubmitTask(OpTypes::kNeg, aops::NegOpOptions{}, {*this})[0]; }

Tensor Tensor::transpose(int dim0, int dim1) {
  return Context::instance().buildOpAndSubmitTask(OpTypes::kTranspose, aops::TransposeOpOptions{.dim0 = dim0, .dim1 = dim1},
                                                  {*this})[0];
}

Tensor Tensor::T() { return transpose(-1, -2); }

Tensor Tensor::to(DeviceTypes device) {
  if (device == impl_->device()) { return *this; }
  return Context::instance().buildOpAndSubmitTask(OpTypes::kX2X, aops::X2XOpOptions{.device = device}, {*this}, device)[0];
}

Tensor Tensor::to(DataTypes dtype) {
  if (dtype == impl_->dtype()) { return *this; }
  return Context::instance().buildOpAndSubmitTask(OpTypes::kCastType, aops::CastTypeOpOptions{.dtype = dtype}, {*this})[0];
}

Tensor Tensor::cpu() {
  if (kCPU == impl_->device()) { return *this; }
  return to(kCPU);
}

Tensor Tensor::cuda() {
  if (kCUDA == impl_->device()) { return *this; }
  return to(kCUDA);
}

std::string Tensor::name() const { return impl()->name(); }

TensorMemTypes Tensor::memType() const { return impl()->memType(); }

Tensor& Tensor::setName(const std::string& name) {
  if (!this->name().empty()) {
    MLLM_WARN("Tensor name is already set to {}, but want to set to {}. We will still perform this request, but not guarantee "
              "the correction",
              this->name(), name);
  }
  impl_->storage()->name_ = name;
  return *this;
}

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

Tensor::stride_t Tensor::stride() const { return impl()->stride(); }

size_t Tensor::numel() const { return impl()->numel(); }

uint32_t Tensor::uuid() const { return impl()->uuid(); }

bool Tensor::isContiguous() const { return impl()->isContiguous(); }

bool Tensor::isContiguousN(int n) const { return impl()->isContiguousN(n); }

Tensor Tensor::contiguous() {
  return Context::instance().buildOpAndSubmitTask(OpTypes::kContiguous, aops::ContiguousOpOptions{}, {*this})[0];
}

Tensor Tensor::reshape(const Tensor::shape_t& shape) {
  return Context::instance().buildOpAndSubmitTask(OpTypes::kReshape, aops::ReshapeOpOptions{.shape = shape}, {*this})[0];
}

Tensor Tensor::view(const Tensor::shape_t& indicies) {
  return Context::instance().buildOpAndSubmitTask(OpTypes::kView, aops::ViewOpOptions{.to_shape = indicies}, {*this})[0];
}

Tensor Tensor::repeat(int32_t multiplier, int32_t dim) {
  if (multiplier == 1) { return *this; }
  return Context::instance().buildOpAndSubmitTask(OpTypes::kRepeat,
                                                  aops::RepeatOpOptions{.dim = dim, .repeat_times = multiplier}, {*this})[0];
}

Tensor Tensor::unsqueeze(int32_t dim) {
  auto this_shape = shape();
  this_shape.insert(this_shape.begin() + dim, 1);
  return view(this_shape);
}

Tensor Tensor::clone() { return Context::instance().buildOpAndSubmitTask(OpTypes::kClone, aops::CloneOpOptions{}, {*this})[0]; }

void Tensor::copy2(const Tensor& src) {
  (void)Context::instance().buildOpAndSubmitTask(OpTypes::kCopy, aops::CopyOpOptions{}, {*this, src});
}

Tensor Tensor::permute(const shape_t& indices) {
  return Context::instance().buildOpAndSubmitTask(OpTypes::kPermute, aops::PermuteOpOptions{.axis = indices}, {*this})[0];
}

size_t Tensor::bytes() const { return impl_->size(); }

ComplexIndexingBlob::ComplexIndexingBlob(SliceIndicesPair p) : slice_indices_(p) {}

ComplexIndexingBlob::ComplexIndexingBlob(const std::vector<int32_t>& p) : vector_indices_(p) {}

ComplexIndexingBlob::ComplexIndexingBlob(const Tensor& p) : tensor_indices_(p) {}

}  // namespace mllm
