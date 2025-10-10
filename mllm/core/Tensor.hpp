// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <span>
#include <memory>
#include <string>
#include <optional>
#include <unordered_map>

#include "mllm/core/TensorViewImpl.hpp"
#include "mllm/core/SlicePrimitives.hpp"

namespace mllm {

struct ComplexIndexingBlob;

using ComplexIndexingList = std::vector<ComplexIndexingBlob>;

class __LinkedTensor {
 public:
  __LinkedTensor(char* ptr, size_t ele_size) : ptr_(ptr), ele_size_(ele_size) {}

  template<typename T>
  inline __LinkedTensor& operator,(T v) && {
    __step(v);
    return *this;
  }

  template<typename T>
  inline __LinkedTensor& operator,(T v) & {
    __step(v);
    return *this;
  }

  template<typename T>
  inline void __step(T v) {
    static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");
    MLLM_RT_ASSERT_EQ(ele_size_, sizeof(T));
    std::memcpy(ptr_ + offset_, &v, sizeof(T));
    offset_ += sizeof(T);
  }

  [[nodiscard]] size_t offset() const { return offset_; }

 private:
  char* ptr_;
  size_t ele_size_;
  size_t offset_ = 0;
};

class Tensor {
 public:
  using shape_t = TensorViewImpl::shape_t;
  using stride_t = TensorViewImpl::stride_t;
  using dtype_t = TensorViewImpl::dtype_t;
  using device_t = TensorViewImpl::device_t;

  /**
   * @brief Default constructor. Creates an empty (null) tensor.
   */
  Tensor() = default;

  void operator delete(void* ptr) noexcept;

  void delete_() noexcept;

  /**
   * @brief If this tensor is not initialized
   *
   * @return true
   * @return false
   */
  [[nodiscard]] inline bool isNil() const { return impl_ == nullptr; }

  /**
   * @brief  Create a nil tensor
   *
   * @return Tensor
   */
  static inline Tensor nil() { return {}; };

  template<typename T>
  static inline Tensor fromVector(const std::vector<T>& vec, const shape_t& shape, DataTypes dtype = kFloat32,
                                  DeviceTypes device = kCPU) {
    Tensor tensor = Tensor::empty(shape, dtype, device).alloc();
    size_t tensor_size = tensor.numel();
    if (vec.size() != tensor_size) {
      MLLM_ERROR_EXIT(ExitCode::kShapeError, "Tensor size mismatch with std::vector size");
      return Tensor::nil();
    }
    std::copy(vec.begin(), vec.end(), tensor.ptr<T>());
    return tensor;
  }

  template<typename T>
  static inline Tensor fromVector(const std::span<T>& vec, const shape_t& shape, DataTypes dtype = kFloat32,
                                  DeviceTypes device = kCPU) {
    Tensor tensor = Tensor::empty(shape, dtype, device).alloc();
    size_t tensor_size = tensor.numel();
    if (vec.size() != tensor_size) {
      MLLM_ERROR_EXIT(ExitCode::kShapeError, "Tensor size mismatch with std::vector size");
      return Tensor::nil();
    }
    std::copy(vec.begin(), vec.end(), tensor.ptr<T>());
    return tensor;
  }

  /**
   * @brief Create a tensor from a std::vector, but reference the vector data, not copy it.
   *
   * @tparam T
   * @param vec
   * @param shape
   * @param dtype
   * @param device
   * @return Tensor
   */
  template<typename T>
  static inline Tensor refVectorData(const std::vector<T>& vec, const shape_t& shape, DataTypes dtype = kFloat32,
                                     DeviceTypes device = kCPU) {
    size_t expected_size = 1;
    for (auto dim : shape) { expected_size *= dim; }

    if (vec.size() != expected_size) {
      MLLM_ERROR_EXIT(ExitCode::kShapeError, "Tensor shape mismatch with std::vector size");
      return Tensor::nil();
    }

    Tensor tensor = Tensor::empty(shape, dtype, device);
    tensor.impl_->storage()->ptr_ = const_cast<T*>(vec.data());
    tensor.impl_->storage()->mem_type_ = kManual;

    return tensor;
  }

  template<typename T>
  inline std::vector<T> toVector() const {
    std::vector<T> vec;
    vec.reserve(numel());
    std::copy(ptr<T>(), ptr<T>() + numel(), std::back_inserter(vec));
    return vec;
  }

  /**
   * @brief Creates a shallow view (slice) of the tensor.
   * @param slice_index Slice specification.
   * @return New tensor view referencing the sliced data.
   * @note Uses shallow copy when step size is 1; may be unsafe for GPU tensors.
   */
  Tensor operator[](const SliceIndices& slice_index) const;

  /**
   * @brief Creates a deep copy of the tensor. Inputs can be vector and tensor.
   *
   * @param complex_indexing
   * @return Tensor
   */
  Tensor operator[](const ComplexIndexingList& complex_indexing) const;

  /**
   * @brief Constructs a tensor from an existing TensorViewImpl.
   * @param impl Shared pointer to the underlying implementation object.
   */
  explicit Tensor(const std::shared_ptr<TensorViewImpl>& impl);

  /**
   * @brief Creates an uninitialized tensor with specified shape and attributes.
   * @note Empty tensor also has its TensorStorageImpl. Which means it is unique.
   * @param shape Dimensions of the tensor.
   * @param dtype Data type (default: kFloat32).
   * @param device Target device (default: kCPU).
   * @return New tensor but NO MEMORY ALLOCTATED!!!
   */
  static Tensor empty(const std::vector<int32_t>& shape, DataTypes dtype = kFloat32, DeviceTypes device = kCPU);

  /**
   * @brief If this tensor is not initialized
   *
   * @note explicit must be set to avoid auto i = tensor. But i is set as bool type.
   *
   * @return true
   * @return false
   */
  explicit inline operator bool() const noexcept { return impl_ != nullptr; }

  /**
   * @brief Alloc a Tensor. Normally used after Tensor::empty(...);
   *
   * @return Tensor&
   */
  Tensor& alloc();

  /**
   * @brief Creates and attaches an auxiliary tensor view to this tensor.
   * @param extra_tensor_name Unique identifier for the auxiliary view.
   * @param shape Dimensions of the auxiliary tensor.
   * @param dtype Data type (default: kFloat32).
   * @param device Target device (default: kCPU).
   * @return Reference to this tensor for chaining.
   * @note This function is designed for quantized Tensor. If one Tensor is quantized to int8 using
   * per tensor quantization method, you can use this_tensor.allocExtraTensorView("scale", shape,
   * kFloat32, kCPU); to attach a `scale` tensor to this tensor.
   */
  Tensor& allocExtraTensorView(const std::string& extra_tensor_name, const std::vector<int32_t>& shape,
                               DataTypes dtype = kFloat32, DeviceTypes device = kCPU);

  /**
   * @brief Retrieves a previously attached auxiliary tensor view.
   * @param extra_tensor_name Name of the auxiliary tensor.
   * @return The requested tensor view.
   * @note This function is designed for quantized Tensor. If one Tensor is quantized to int8 using
   * per tensor quantization method, you can use
   * this_tensor.getExtraTensorViewInTensor("scale").item<float>(); to get a `scale` tensor from
   * this tensor.
   */
  Tensor getExtraTensorViewInTensor(const std::string& extra_tensor_name);

  /**
   * @brief Creates a tensor filled with zeros.
   * @param shape Dimensions of the tensor.
   * @param dtype Data type (default: kFloat32).
   * @param device Target device (default: kCPU).
   * @return New tensor with initialized zero values.
   */
  static Tensor zeros(const std::vector<int32_t>& shape, DataTypes dtype = kFloat32, DeviceTypes device = kCPU);

  /**
   * @brief Creates a tensor filled with ones.
   * @param shape Dimensions of the tensor.
   * @param dtype Data type (default: kFloat32).
   * @param device Target device (default: kCPU).
   * @return New tensor with initialized one values.
   */
  static Tensor ones(const std::vector<int32_t>& shape, DataTypes dtype = kFloat32, DeviceTypes device = kCPU);

  /**
   * @brief Creates a tensor with evenly spaced values within a specified range.
   * @param start
   * @param end
   * @param step
   * @param dtype
   * @param device
   * @return Tensor
   */
  static Tensor arange(float start, float end, float step, DataTypes dtype = kFloat32, DeviceTypes device = kCPU);

  /**
   * @brief Creates a tensor with random values within a specified range.
   * @param shape
   * @param start
   * @param end
   * @param dtype
   * @param device
   * @return Tensor
   */
  static Tensor random(const std::vector<int32_t>& shape, float start = -1.f, float end = 1.f, DataTypes dtype = kFloat32,
                       DeviceTypes device = kCPU);

  /// @name Arithmetic Operations
  /// Element-wise operations between tensors.
  /// @{
  Tensor operator+(const Tensor& rhs);
  Tensor operator-(const Tensor& rhs);
  Tensor operator*(const Tensor& rhs);
  Tensor operator/(const Tensor& rhs);
  /// @}

  /// @name Scalar Operations
  /// Element-wise operations with scalar values.
  /// @{
  Tensor operator+(float rhs);
  Tensor operator-(float rhs);
  Tensor operator*(float rhs);
  Tensor operator/(float rhs);
  /// @}

  /// @name Scalar Operations with complex rhs type
  /// Element-wise operations with complex rhs type scalar values.
  /// @{
  Tensor operator+(std::complex<float> rhs);
  friend Tensor operator+(std::complex<float> c, Tensor t) { return t + c; }
  Tensor operator-(std::complex<float> rhs);
  friend Tensor operator-(std::complex<float> c, Tensor t) { return t - c; }
  Tensor operator*(std::complex<float> rhs);
  friend Tensor operator*(std::complex<float> c, Tensor t) { return t * c; }
  Tensor operator/(std::complex<float> rhs);
  friend Tensor operator/(std::complex<float> c, Tensor t) { return t / c; }
  /// @}

  /**
   * @brief Computes the absolute value of the tensor elements.
   *
   * @return Tensor with absolute values
   */
  Tensor abs();

  /**
   * @brief Clips (limits) the values in a tensor.
   * @param min_val Minimum value
   * @param max_val Maximum value
   * @return A tensor with clipped values
   */
  Tensor clip(float min_val, float max_val);

  /**
   * @brief Finds the top k largest (or smallest) elements in a tensor.
   * @param k Number of top elements to find
   * @param dim Dimension along which to find top k elements
   * @param largest If true, find the largest elements; otherwise, find the smallest
   * @param sorted If true, the result will be sorted by value
   * @return An array containing values and indices of the top k elements
   */
  std::array<Tensor, 2> topk(int32_t k, int32_t dim = -1, bool largest = true, bool sorted = true);

  /**
   * @brief Get min
   *
   * @param dim if dim == 0x7fffffff, return a scalar value.
   * @return Tensor
   */
  Tensor min(int32_t dim = 0x7fffffff, bool keep_dim = false);

  /**
   * @brief Get max
   *
   * @param dim if dim == 0x7fffffff, return a scalar value.
   * @return Tensor
   */
  Tensor max(int32_t dim = 0x7fffffff, bool keep_dim = false);

  /**
   * @brief Get sum
   *
   * @param dim if dim == 0x7fffffff, return a scalar value.
   * @return Tensor
   */
  Tensor sum(int32_t dim = 0x7fffffff, bool keep_dim = false);

  /**
   * @brief Get mean
   *
   * @param dim if dim == 0x7fffffff, return a scalar value.
   * @return Tensor
   */
  Tensor mean(int32_t dim = 0x7fffffff, bool keep_dim = false);

  /**
   * @brief Negative
   *
   * @return Tensor
   */
  Tensor operator-();

  /**
   * @brief Swaps two dimensions of the tensor.
   * @param dim0 First dimension index.
   * @param dim1 Second dimension index.
   * @return New tensor with transposed dimensions.
   */
  Tensor transpose(int dim0, int dim1);

  /**
   * @brief Transpose Tensor at last 2 dims.
   *
   * @return Tensor
   */
  Tensor T();

  /**
   * @brief Transfers tensor to specified device.
   * @param device Target device.
   * @return New tensor on target device (data copied if needed).
   */
  Tensor to(DeviceTypes device);

  /**
   * @brief Converts tensor to specified data type.
   * @param dtype Target data type.
   * @return New tensor with converted data type.
   */
  Tensor to(DataTypes dtype);

  /**
   * @brief Shortcut for moving tensor to CPU.
   * @return CPU-resident tensor.
   */
  Tensor cpu();

  /**
   * @brief Shortcut for moving tensor to GPU.
   * @return GPU-resident tensor.
   */
  Tensor cuda();

  /**
   * @brief Gets the tensor's name.
   * @return Name string (empty if unnamed).
   */
  [[nodiscard]] std::string name() const;

  /**
   * @brief Gets memory type.
   * @return Memory type identifier.
   */
  [[nodiscard]] TensorMemTypes memType() const;

  /**
   * @brief Sets tensor name.
   * @param name New name for tensor.
   * @return Reference to this tensor for chaining.
   */
  Tensor& setName(const std::string& name);

  /**
   * @brief Sets memory type.
   * @param mem_type New memory type.
   * @return Reference to this tensor for chaining.
   */
  Tensor& setMemType(TensorMemTypes mem_type);

  /**
   * @brief Gets data type.
   * @return Current data type.
   */
  [[nodiscard]] DataTypes dtype() const;

  /**
   * @brief Gets device location.
   * @return Current device type.
   */
  [[nodiscard]] DeviceTypes device() const;

  /**
   * @brief Gets tensor dimensions.
   * @return Shape vector.
   */
  [[nodiscard]] shape_t shape() const;

  /**
   * @brief  Gets tensor rank.
   *
   * @return size_t
   */
  [[nodiscard]] size_t rank() const;

  /**
   * @brief Gets tensor strides.
   *
   * @return stride_t
   */
  [[nodiscard]] stride_t stride() const;

  /**
   * @brief Calculates total number of elements.
   * @return Product of all dimensions.
   */
  [[nodiscard]] size_t numel() const;

  /**
   * @brief Gets unique tensor ID.
   * @return Universally unique identifier.
   */
  [[nodiscard]] uint32_t uuid() const;

  /**
   * @brief Checks memory layout contiguity.
   * @return True if memory is contiguous.
   */
  [[nodiscard]] bool isContiguous() const;

  /**
   * @brief Checks memory layout contiguity.
   * @return True if memory is contiguous.
   */
  [[nodiscard]] bool isContiguousN(int n) const;

  /**
   * @brief Creates contiguous copy if non-contiguous.
   * @return Contiguous tensor (may be a view or copy).
   */
  Tensor contiguous();

  /**
   * @brief Reshapes tensor without changing data order.
   * @param shape New dimensions.
   * @return Reshaped tensor view.
   */
  Tensor reshape(const shape_t& shape);

  /**
   * @brief Experimental: Creates tensor view with custom indexing.
   * @param indicies View specification.
   * @return New tensor view.
   * @warning This function is in an early age.
   */
  Tensor view(const shape_t& indicies);

  /**
   * @brief Repeats tensor along a dimension.
   *
   * @param multiplier
   * @param dim
   * @return Tensor
   */
  Tensor repeat(int32_t multiplier, int32_t dim);

  /**
   * @brief Unsqueeze tensor along a dimension.
   *
   * @param dim
   * @return Tensor
   */
  Tensor unsqueeze(int32_t dim);

  /**
   * @brief
   *
   * @param dim
   * @return Tensor
   */
  Tensor squeeze(int32_t dim = 0x7fffffff);

  /**
   * @brief clone a tensor
   *
   * @return Tensor
   */
  Tensor clone();

  /**
   * @brief copy a tensor to another
   *
   * @param src
   * @return Tensor
   */
  void copy2(const Tensor& src);

  /**
   * @brief Permute tensor to a new shape
   *
   * @param indices
   * @return Tensor
   */
  Tensor permute(const shape_t& indices);

  /**
   * @brief Accesses the underlying implementation object.
   * @return Shared pointer to TensorViewImpl.
   */
  [[nodiscard]] inline TensorViewImpl::ptr_t impl() const { return impl_; }

  /**
   * @brief return how many bytes this tensor alloced.
   *
   * @return size_t
   */
  [[nodiscard]] size_t bytes() const;

  /**
   * @brief Gets base pointer of tensor data.
   * @tparam T Expected data type.
   * @return Typed base pointer.
   */
  template<typename T>
  [[nodiscard]] T* ptr() const {
    return impl_->ptr<T>();
  }

  /**
   * @brief Typed pointer access with offset.
   * @tparam T Expected data type.
   * @param offsets Multi-dimensional indices.
   * @return Typed pointer to the element.
   */
  template<typename T>
  T* offsettedPtr(const std::vector<int32_t>& offsets) {
    return impl_->offsettedPtr<T>(offsets);
  }

  /**
   * @brief Typed pointer access with offset.
   *
   * @tparam T
   * @param offsets
   * @return T*
   */
  template<typename T>
  T* coffsettedPtr(const std::vector<int32_t>& offsets) const {
    return impl_->offsettedPtr<T>(offsets);
  }

  /**
   * @brief Typed pointer access with offset.
   * @tparam T Expected data type.
   * @param offsets Multi-dimensional indices.
   * @return Typed pointer to the element.
   */
  template<typename T>
  T* ptrAt(const std::vector<int32_t>& offsets) {
    return impl_->offsettedPtr<T>(offsets);
  }

  template<typename T>
  T* cptrAt(const std::vector<int32_t>& offsets) const {
    return impl_->offsettedPtr<T>(offsets);
  }

  /**
   * @brief Accesses a tensor element at specified coordinates.
   * @tparam T Expected data type (must match tensor dtype).
   * @param offsets Multi-dimensional indices.
   * @return Reference to the element.
   */
  template<typename T>
  T& at(const std::vector<int32_t>& offsets) {
    return *(offsettedPtr<T>(offsets));
  }

  /**
   * @brief Accesses a tensor element at specified coordinates (const version).
   * @tparam T Expected data type (must match tensor dtype).
   * @param offsets Multi-dimensional indices.
   * @return Const reference to the element.
   */
  template<typename T>
  const T& constAt(const std::vector<int32_t>& offsets) const {
    return *(const_cast<Tensor*>(this)->offsettedPtr<T>(offsets));
  }

  [[nodiscard]] std::unordered_map<std::string, TensorViewImpl::ptr_t>& attachedViews() { return attached_views_; }

  void attach(const std::string& name, const TensorViewImpl::ptr_t& view) { attached_views_[name] = view; }

 private:
  template<typename T>
  friend __LinkedTensor operator<<(const Tensor& t, T first);

  std::shared_ptr<TensorViewImpl> impl_ = nullptr;
  std::unordered_map<std::string, TensorViewImpl::ptr_t> attached_views_;
};

template<typename T>
inline __LinkedTensor operator<<(const Tensor& t, T first) {
  static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");
  __LinkedTensor lt(t.ptr<char>(), bytesOfType(t.dtype()));
  lt.__step(first);
  return lt;
}

struct ComplexIndexingBlob {
  ComplexIndexingBlob(SliceIndicesPair p);  // NOLINT

  ComplexIndexingBlob(const std::vector<int32_t>& p);  // NOLINT

  ComplexIndexingBlob(const Tensor& p);  // NOLINT

  std::optional<SliceIndicesPair> slice_indices_ = std::nullopt;
  std::optional<std::vector<int32_t>> vector_indices_ = std::nullopt;
  std::optional<Tensor> tensor_indices_ = std::nullopt;
};

}  // namespace mllm
