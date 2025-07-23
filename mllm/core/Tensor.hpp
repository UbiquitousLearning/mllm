/**
 * @file Tensor.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-21
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "mllm/core/TensorViewImpl.hpp"

namespace mllm {

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
   * @return New tensor but NO MEMORY ALLOCTED!!!
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
   * @brief clone a tensor
   *
   * @return Tensor
   */
  Tensor clone();

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
  size_t bytes();

 private:
  std::shared_ptr<TensorViewImpl> impl_ = nullptr;
  std::unordered_map<std::string, TensorViewImpl::ptr_t> attached_views_;
};

}  // namespace mllm
