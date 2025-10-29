// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <optional>

#include <xxHash/xxhash.h>

#include "mllm/core/Tensor.hpp"
#include "mllm/core/DataTypes.hpp"
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
#include "mllm/core/aops/TopKOp.hpp"
#include "mllm/core/aops/TransposeOp.hpp"
#include "mllm/core/aops/ViewOp.hpp"
#include "mllm/core/aops/X2XOp.hpp"
#include "mllm/core/aops/ArgsortOp.hpp"
#include "mllm/engine/Context.hpp"

namespace mllm {

void Tensor::operator delete(void* ptr) noexcept {
  ((Tensor*)ptr)->impl_.reset();
  for (auto& [a, _] : ((Tensor*)ptr)->attached_views_) { ((Tensor*)ptr)->attached_views_[a].reset(); }
}

void Tensor::delete_() noexcept {
  this->impl_.reset();
  for (auto& [a, _] : this->attached_views_) { this->attached_views_[a].reset(); }
}

/**
 * @brief Slices the tensor along the specified dimensions.
 *
 * You can use tensor[{1, 2, 3}] or tensor[make_slice(1), 2, 3] to achieve the same effect.
 * (WOW! amazing syntax sugar!)
 */
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

Tensor Tensor::emptyLike(const Tensor& liked_tensor) {
  auto ret = Tensor::empty(liked_tensor.shape(), liked_tensor.dtype(), liked_tensor.device());
  return ret;
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
          .type = aops::FillOpTypes::kRandom, .start = start, .end = end, .seed = Context::instance().getRandomState()},
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

Tensor Tensor::mul_(const Tensor& rhs) {
  auto opts = aops::MulOpOptions{};
  opts.setInplace(true);
  return Context::instance().buildOpAndSubmitTask(OpTypes::kMul, opts, {*this, rhs})[0];
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
  auto opts = aops::AddOpOptions{};
  opts.setInputsConstant(0, 0);
  opts.setInputsConstant(1, 1);
  return Context::instance().buildOpAndSubmitTask(OpTypes::kAdd, opts, {*this, rhs_tensor})[0];
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
  auto opts = aops::SubOpOptions{};
  opts.setInputsConstant(0, 0);
  opts.setInputsConstant(1, 1);
  return Context::instance().buildOpAndSubmitTask(OpTypes::kSub, opts, {*this, rhs_tensor})[0];
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
  auto opts = aops::MulOpOptions{};
  opts.setInputsConstant(0, 0);
  opts.setInputsConstant(1, 1);
  return Context::instance().buildOpAndSubmitTask(OpTypes::kMul, opts, {*this, rhs_tensor})[0];
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
  auto opts = aops::DivOpOptions{};
  opts.setInputsConstant(0, 0);
  opts.setInputsConstant(1, 1);
  return Context::instance().buildOpAndSubmitTask(OpTypes::kDiv, opts, {*this, rhs_tensor})[0];
}

Tensor Tensor::operator+(std::complex<float> rhs) {
  auto rhs_tensor = Tensor::empty({1}, kComplexFloat32, device()).alloc();
  switch (dtype()) {
    case kFloat32: *(rhs_tensor.ptr<std::complex<float>>()) = rhs; break;
    default: NYI("Type is not supported"); break;
  }
  auto opts = aops::AddOpOptions{};
  opts.setInputsConstant(0, 0);
  opts.setInputsConstant(1, 1);
  return Context::instance().buildOpAndSubmitTask(OpTypes::kAdd, opts, {*this, rhs_tensor})[0];
}

Tensor Tensor::operator-(std::complex<float> rhs) {
  auto rhs_tensor = Tensor::empty({1}, kComplexFloat32, device()).alloc();
  switch (dtype()) {
    case kFloat32: *(rhs_tensor.ptr<std::complex<float>>()) = rhs; break;
    default: NYI("Type is not supported"); break;
  }
  auto opts = aops::SubOpOptions{};
  opts.setInputsConstant(0, 0);
  opts.setInputsConstant(1, 1);
  return Context::instance().buildOpAndSubmitTask(OpTypes::kSub, opts, {*this, rhs_tensor})[0];
}

Tensor Tensor::operator*(std::complex<float> rhs) {
  auto rhs_tensor = Tensor::empty({1}, kComplexFloat32, device()).alloc();
  switch (dtype()) {
    case kFloat32: *(rhs_tensor.ptr<std::complex<float>>()) = rhs; break;
    default: NYI("Type is not supported"); break;
  }
  auto opts = aops::MulOpOptions{};
  opts.setInputsConstant(0, 0);
  opts.setInputsConstant(1, 1);
  return Context::instance().buildOpAndSubmitTask(OpTypes::kMul, opts, {*this, rhs_tensor})[0];
}

Tensor Tensor::operator/(std::complex<float> rhs) {
  auto rhs_tensor = Tensor::empty({1}, kComplexFloat32, device()).alloc();
  switch (dtype()) {
    case kFloat32: *(rhs_tensor.ptr<std::complex<float>>()) = rhs; break;
    default: NYI("Type is not supported"); break;
  }
  auto opts = aops::DivOpOptions{};
  opts.setInputsConstant(0, 0);
  opts.setInputsConstant(1, 1);
  return Context::instance().buildOpAndSubmitTask(OpTypes::kDiv, opts, {*this, rhs_tensor})[0];
}

Tensor Tensor::abs() { return Context::instance().buildOpAndSubmitTask(OpTypes::kAbs, aops::AbsOpOptions{}, {*this})[0]; }

Tensor Tensor::argsort(int dim, bool descending) {
  return Context::instance().buildOpAndSubmitTask(OpTypes::kArgsort,
                                                  aops::ArgsortOpOptions{.dim = dim, .descending = descending}, {*this})[0];
}

Tensor Tensor::clip(float min_val, float max_val) {
  return Context::instance().buildOpAndSubmitTask(OpTypes::kClip, aops::ClipOpOptions{.min_val = min_val, .max_val = max_val},
                                                  {*this})[0];
}

std::array<Tensor, 2> Tensor::topk(int32_t k, int32_t dim, bool largest, bool sorted) {
  auto outputs = Context::instance().buildOpAndSubmitTask(
      OpTypes::kTopK, aops::TopKOpOptions{.k = k, .dim = dim, .largest = largest, .sorted = sorted}, {*this});
  return {outputs[0], outputs[1]};
}

Tensor Tensor::min(int32_t dim, bool keep_dim) {
  return Context::instance().buildOpAndSubmitTask(OpTypes::kReduceMin,
                                                  aops::ReduceMinOpOptions{.dim = dim, .keep_dim = keep_dim}, {*this})[0];
}

Tensor Tensor::max(int32_t dim, bool keep_dim) {
  return Context::instance().buildOpAndSubmitTask(OpTypes::kReduceMax,
                                                  aops::ReduceMaxOpOptions{.dim = dim, .keep_dim = keep_dim}, {*this})[0];
}

Tensor Tensor::sum(int32_t dim, bool keep_dim) {
  return Context::instance().buildOpAndSubmitTask(OpTypes::kReduceSum,
                                                  aops::ReduceSumOpOptions{.dim = dim, .keep_dim = keep_dim}, {*this})[0];
}

Tensor Tensor::mean(int32_t dim, bool keep_dim) {
  return Context::instance().buildOpAndSubmitTask(OpTypes::kMean, aops::MeanOpOptions{.dim = dim, .keep_dim = keep_dim},
                                                  {*this})[0];
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

size_t Tensor::rank() const { return shape().size(); }

Tensor::stride_t Tensor::stride() const { return impl()->stride(); }

size_t Tensor::numel() const { return impl()->numel(); }

uint32_t Tensor::uuid() const { return impl()->uuid(); }

size_t Tensor::hash() const {
  constexpr size_t kStackCap = 16;
  uint32_t stack_buf[kStackCap];
  std::vector<uint32_t> heap_buf;

  auto* buf = stack_buf;
  size_t count = 1 + attached_views_.size();
  if (count > kStackCap) {
    heap_buf.resize(count);
    buf = heap_buf.data();
  }
  buf[0] = uuid();
  size_t idx = 1;
  for (const auto& [_, view] : attached_views_) { buf[idx++] = view ? view->uuid() : 0u; }
  return XXH64(buf, count * sizeof(uint32_t), 0);
}

bool Tensor::isContiguous() const { return impl()->isContiguous(); }

bool Tensor::isContiguousN(int n) const { return impl()->isContiguousN(n); }

int32_t Tensor::size(int32_t id) const {
  auto nid = id;
  if (id < 0) { nid = static_cast<int32_t>(rank()) + id; }
  return shape()[nid];
}

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
  if (dim < 0) { dim = static_cast<int32_t>(rank()) + dim; }
  return Context::instance().buildOpAndSubmitTask(OpTypes::kRepeat,
                                                  aops::RepeatOpOptions{.dim = dim, .repeat_times = multiplier}, {*this})[0];
}

Tensor Tensor::unsqueeze(int32_t dim) {
  if (dim < 0) { dim = static_cast<int32_t>(rank()) + dim + 1; }
  auto this_shape = shape();
  this_shape.insert(this_shape.begin() + dim, 1);
  return view(this_shape);
}

Tensor Tensor::squeeze(int32_t dim) {
  auto this_shape = shape();
  if (dim == 0x7fffffff) {
    // Remove all dimensions of size 1
    shape_t new_shape;
    for (const auto& size : this_shape) {
      if (size != 1) { new_shape.push_back(size); }
    }
    // If no dimensions were removed, return original tensor view
    if (new_shape.empty() && !this_shape.empty()) { new_shape.push_back(1); }
    return view(new_shape);
  } else {
    // Handle negative indices
    if (dim < 0) { dim += static_cast<int32_t>(this_shape.size()); }

    // Check if index is valid
    if (dim < 0 || dim >= static_cast<int32_t>(this_shape.size())) {
      return *this;  // Return original tensor
    }

    // If specified dimension has size 1, remove it
    if (this_shape[dim] == 1) {
      shape_t new_shape;
      new_shape.reserve(this_shape.size() - 1);
      for (int i = 0; i < static_cast<int>(this_shape.size()); ++i) {
        if (i != dim) { new_shape.push_back(this_shape[i]); }
      }
      return view(new_shape);
    }

    // If specified dimension does not have size 1, return original tensor
    return *this;
  }
}

Tensor Tensor::flatten(int32_t dim) {
  const auto old_shape = shape();
  const int32_t ndim = static_cast<int32_t>(old_shape.size());

  if (dim == 0x7fffffff) {
    int32_t total = 1;
    for (auto s : old_shape) total *= s;
    return view({total});
  }

  if (ndim == 0) return view({1});

  if (dim < 0) dim += ndim;
  if (dim < 0 || dim >= ndim) throw std::out_of_range("flatten dim out of range");

  std::vector<int32_t> new_shape;
  new_shape.reserve(dim + 1);

  for (int32_t i = 0; i < dim; ++i) new_shape.push_back(old_shape[i]);

  int32_t flatten_size = 1;
  for (int32_t i = dim; i < ndim; ++i) flatten_size *= old_shape[i];
  new_shape.push_back(flatten_size);

  return view(new_shape);
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
