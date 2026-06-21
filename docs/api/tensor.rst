Tensor API
==========

The Tensor class is the fundamental data structure in MLLM for representing multi-dimensional arrays. It provides a comprehensive set of operations for manipulating tensor data across different devices and data types.

.. code-block:: cpp

   #include "mllm/core/Tensor.hpp"

Creating Tensors
----------------

empty
~~~~~

.. cpp:function:: static Tensor Tensor::empty(const std::vector<int32_t>& shape, DataTypes dtype = kFloat32, DeviceTypes device = kCPU)

   Creates an uninitialized tensor with specified shape and attributes.

   :param shape: Dimensions of the tensor
   :param dtype: Data type (default: kFloat32)
   :param device: Target device (default: kCPU)
   :return: New tensor but NO MEMORY ALLOCATED!!!

zeros
~~~~~

.. cpp:function:: static Tensor Tensor::zeros(const std::vector<int32_t>& shape, DataTypes dtype = kFloat32, DeviceTypes device = kCPU)

   Creates a tensor filled with zeros.

   :param shape: Dimensions of the tensor
   :param dtype: Data type (default: kFloat32)
   :param device: Target device (default: kCPU)
   :return: New tensor with initialized zero values

ones
~~~~

.. cpp:function:: static Tensor Tensor::ones(const std::vector<int32_t>& shape, DataTypes dtype = kFloat32, DeviceTypes device = kCPU)

   Creates a tensor filled with ones.

   :param shape: Dimensions of the tensor
   :param dtype: Data type (default: kFloat32)
   :param device: Target device (default: kCPU)
   :return: New tensor with initialized one values

arange
~~~~~~

.. cpp:function:: static Tensor Tensor::arange(float start, float end, float step, DataTypes dtype = kFloat32, DeviceTypes device = kCPU)

   Creates a tensor with evenly spaced values within a specified range.

   :param start: Starting value
   :param end: Ending value
   :param step: Step size
   :param dtype: Data type (default: kFloat32)
   :param device: Target device (default: kCPU)
   :return: Tensor with evenly spaced values

random
~~~~~~

.. cpp:function:: static Tensor Tensor::random(const std::vector<int32_t>& shape, float start = -1.f, float end = 1.f, DataTypes dtype = kFloat32, DeviceTypes device = kCPU)

   Creates a tensor with random values within a specified range.

   :param shape: Dimensions of the tensor
   :param start: Minimum random value (default: -1.f)
   :param end: Maximum random value (default: 1.f)
   :param dtype: Data type (default: kFloat32)
   :param device: Target device (default: kCPU)
   :return: Tensor with random values

fromVector
~~~~~~~~~~

.. cpp:function:: template<typename T> static Tensor Tensor::fromVector(const std::vector<T>& vec, const shape_t& shape, DataTypes dtype = kFloat32, DeviceTypes device = kCPU)

   Creates a tensor from a std::vector.

   :param vec: Source vector
   :param shape: Dimensions of the tensor
   :param dtype: Data type (default: kFloat32)
   :param device: Target device (default: kCPU)
   :return: Tensor containing the vector data

Tensor Properties
-----------------

name
~~~~

.. cpp:function:: std::string Tensor::name() const

   Gets the tensor's name.

   :return: Name string (empty if unnamed)

dtype
~~~~~

.. cpp:function:: DataTypes Tensor::dtype() const

   Gets data type.

   :return: Current data type

device
~~~~~~

.. cpp:function:: DeviceTypes Tensor::device() const

   Gets device location.

   :return: Current device type

shape
~~~~~

.. cpp:function:: shape_t Tensor::shape() const

   Gets tensor dimensions.

   :return: Shape vector

stride
~~~~~~

.. cpp:function:: stride_t Tensor::stride() const

   Gets tensor strides.

   :return: Stride vector

numel
~~~~~

.. cpp:function:: size_t Tensor::numel() const

   Calculates total number of elements.

   :return: Product of all dimensions

uuid
~~~~

.. cpp:function:: uint32_t Tensor::uuid() const

   Gets unique tensor ID.

   :return: Universally unique identifier

bytes
~~~~~

.. cpp:function:: size_t Tensor::bytes() const

   Return how many bytes this tensor allocated.

   :return: Size in bytes

Memory Operations
-----------------

alloc
~~~~~

.. cpp:function:: Tensor& Tensor::alloc()

   Allocates memory for a Tensor. Normally used after Tensor::empty(...).

   :return: Reference to this tensor for chaining

isContiguous
~~~~~~~~~~~~

.. cpp:function:: bool Tensor::isContiguous() const

   Checks memory layout contiguity.

   :return: True if memory is contiguous

isContiguousN
~~~~~~~~~~~~~

.. cpp:function:: bool Tensor::isContiguousN(int n) const

   Checks memory layout contiguity for the last n dimensions.

   :param n: Number of last dimensions to check
   :return: True if memory is contiguous for the last n dimensions

contiguous
~~~~~~~~~~

.. cpp:function:: Tensor Tensor::contiguous()

   Creates contiguous copy if non-contiguous.

   :return: Contiguous tensor (may be a view or copy)

clone
~~~~~

.. cpp:function:: Tensor Tensor::clone()

   Clone a tensor.

   :return: A copy of the tensor

copy2
~~~~~

.. cpp:function:: void Tensor::copy2(const Tensor& src)

   Copy data from another tensor.

   :param src: Source tensor

reshape
~~~~~~~

.. cpp:function:: Tensor Tensor::reshape(const shape_t& shape)

   Reshapes tensor without changing data order.

   :param shape: New dimensions
   :return: Reshaped tensor view

view
~~~~

.. cpp:function:: Tensor Tensor::view(const shape_t& indicies)

   Experimental: Creates tensor view with custom indexing.

   :param indicies: View specification
   :return: New tensor view
   :warning: This function is in an early age

permute
~~~~~~~

.. cpp:function:: Tensor Tensor::permute(const shape_t& indices)

   Permute tensor to a new shape.

   :param indices: New axis order
   :return: Permuted tensor

transpose
~~~~~~~~~

.. cpp:function:: Tensor Tensor::transpose(int dim0, int dim1)

   Swaps two dimensions of the tensor.

   :param dim0: First dimension index
   :param dim1: Second dimension index
   :return: New tensor with transposed dimensions

T
~

.. cpp:function:: Tensor Tensor::T()

   Transpose Tensor at last 2 dims.

   :return: Transposed tensor

repeat
~~~~~~

.. cpp:function:: Tensor Tensor::repeat(int32_t multiplier, int32_t dim)

   Repeats tensor along a dimension.

   :param multiplier: Repeat multiplier
   :param dim: Dimension to repeat along
   :return: Repeated tensor

unsqueeze
~~~~~~~~~

.. cpp:function:: Tensor Tensor::unsqueeze(int32_t dim)

   Unsqueeze tensor along a dimension.

   :param dim: Dimension to unsqueeze
   :return: Unsqueezed tensor

to (device)
~~~~~~~~~~~

.. cpp:function:: Tensor Tensor::to(DeviceTypes device)

   Transfers tensor to specified device.

   :param device: Target device
   :return: New tensor on target device (data copied if needed)

to (dtype)
~~~~~~~~~~

.. cpp:function:: Tensor Tensor::to(DataTypes dtype)

   Converts tensor to specified data type.

   :param dtype: Target data type
   :return: New tensor with converted data type

cpu
~~~

.. cpp:function:: Tensor Tensor::cpu()

   Shortcut for moving tensor to CPU.

   :return: CPU-resident tensor

cuda
~~~~

.. cpp:function:: Tensor Tensor::cuda()

   Shortcut for moving tensor to GPU.

   :return: GPU-resident tensor

Element-wise Arithmetic Operations
----------------------------------

operator+
~~~~~~~~~

.. cpp:function:: Tensor Tensor::operator+(const Tensor& rhs)

   Element-wise addition with another tensor.

   :param rhs: Right-hand side tensor
   :return: Result tensor

.. cpp:function:: Tensor Tensor::operator+(float rhs)

   Element-wise addition with a scalar.

   :param rhs: Right-hand side scalar value
   :return: Result tensor

operator-
~~~~~~~~~

.. cpp:function:: Tensor Tensor::operator-(const Tensor& rhs)

   Element-wise subtraction with another tensor.

   :param rhs: Right-hand side tensor
   :return: Result tensor

.. cpp:function:: Tensor Tensor::operator-(float rhs)

   Element-wise subtraction with a scalar.

   :param rhs: Right-hand side scalar value
   :return: Result tensor

.. cpp:function:: Tensor Tensor::operator-()

   Negative.

   :return: Negated tensor

operator*
~~~~~~~~~

.. cpp:function:: Tensor Tensor::operator*(const Tensor& rhs)

   Element-wise multiplication with another tensor.

   :param rhs: Right-hand side tensor
   :return: Result tensor

.. cpp:function:: Tensor Tensor::operator*(float rhs)

   Element-wise multiplication with a scalar.

   :param rhs: Right-hand side scalar value
   :return: Result tensor

operator/
~~~~~~~~~

.. cpp:function:: Tensor Tensor::operator/(const Tensor& rhs)

   Element-wise division with another tensor.

   :param rhs: Right-hand side tensor
   :return: Result tensor

.. cpp:function:: Tensor Tensor::operator/(float rhs)

   Element-wise division with a scalar.

   :param rhs: Right-hand side scalar value
   :return: Result tensor

abs
~~~

.. cpp:function:: Tensor Tensor::abs()

   Computes the absolute value of the tensor elements.

   :return: Tensor with absolute values

clip
~~~~

.. cpp:function:: Tensor Tensor::clip(float min_val, float max_val)

   Clips (limits) the values in a tensor.

   :param min_val: Minimum value
   :param max_val: Maximum value
   :return: A tensor with clipped values

Reduction Operations
--------------------

topk
~~~~

.. cpp:function:: std::array<Tensor, 2> Tensor::topk(int32_t k, int32_t dim = -1, bool largest = true, bool sorted = true)

   Finds the top k largest (or smallest) elements in a tensor.

   :param k: Number of top elements to find
   :param dim: Dimension along which to find top k elements (default: -1)
   :param largest: If true, find the largest elements; otherwise, find the smallest (default: true)
   :param sorted: If true, the result will be sorted by value (default: true)
   :return: An array containing values and indices of the top k elements

min
~~~

.. cpp:function:: Tensor Tensor::min(bool keep_dim = false, int32_t dim = 0x7fffffff)

   Get minimum values.

   :param keep_dim: If true, keep the reduced dimension (default: false)
   :param dim: Dimension to reduce. If 0x7fffffff, return a scalar value (default: 0x7fffffff)
   :return: Tensor with minimum values

max
~~~

.. cpp:function:: Tensor Tensor::max(bool keep_dim = false, int32_t dim = 0x7fffffff)

   Get maximum values.

   :param keep_dim: If true, keep the reduced dimension (default: false)
   :param dim: Dimension to reduce. If 0x7fffffff, return a scalar value (default: 0x7fffffff)
   :return: Tensor with maximum values

sum
~~~

.. cpp:function:: Tensor Tensor::sum(bool keep_dim = false, int32_t dim = 0x7fffffff)

   Get sum of elements.

   :param keep_dim: If true, keep the reduced dimension (default: false)
   :param dim: Dimension to reduce. If 0x7fffffff, return a scalar value (default: 0x7fffffff)
   :return: Tensor with sum of elements

mean
~~~~

.. cpp:function:: Tensor Tensor::mean(bool keep_dim = false, int32_t dim = 0x7fffffff)

   Get mean of elements.

   :param keep_dim: If true, keep the reduced dimension (default: false)
   :param dim: Dimension to reduce. If 0x7fffffff, return a scalar value (default: 0x7fffffff)
   :return: Tensor with mean of elements

Indexing and Slicing
--------------------

operator[]
~~~~~~~~~~

.. cpp:function:: Tensor Tensor::operator[](const SliceIndices& slice_index) const

   Creates a shallow view (slice) of the tensor.

   :param slice_index: Slice specification
   :return: New tensor view referencing the sliced data
   :note: Uses shallow copy when step size is 1; may be unsafe for GPU tensors

.. cpp:function:: Tensor Tensor::operator[](const ComplexIndexingList& complex_indexing) const

   Creates a deep copy of the tensor with complex indexing.

   :param complex_indexing: Complex indexing specification
   :return: New tensor with indexed data

setName
~~~~~~~

.. cpp:function:: Tensor& Tensor::setName(const std::string& name)

   Sets tensor name.

   :param name: New name for tensor
   :return: Reference to this tensor for chaining

setMemType
~~~~~~~~~~

.. cpp:function:: Tensor& Tensor::setMemType(TensorMemTypes mem_type)

   Sets memory type.

   :param mem_type: New memory type
   :return: Reference to this tensor for chaining

memType
~~~~~~~

.. cpp:function:: TensorMemTypes Tensor::memType() const

   Gets memory type.

   :return: Memory type identifier

Auxiliary Tensor Views
----------------------

allocExtraTensorView
~~~~~~~~~~~~~~~~~~~~

.. cpp:function:: Tensor& Tensor::allocExtraTensorView(const std::string& extra_tensor_name, const std::vector<int32_t>& shape, DataTypes dtype = kFloat32, DeviceTypes device = kCPU)

   Creates and attaches an auxiliary tensor view to this tensor.

   :param extra_tensor_name: Unique identifier for the auxiliary view
   :param shape: Dimensions of the auxiliary tensor
   :param dtype: Data type (default: kFloat32)
   :param device: Target device (default: kCPU)
   :return: Reference to this tensor for chaining
   :note: This function is designed for quantized Tensor. If one Tensor is quantized to int8 using per tensor quantization method, you can use this_tensor.allocExtraTensorView("scale", shape, kFloat32, kCPU); to attach a `scale` tensor to this tensor.

getExtraTensorViewInTensor
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. cpp:function:: Tensor Tensor::getExtraTensorViewInTensor(const std::string& extra_tensor_name)

   Retrieves a previously attached auxiliary tensor view.

   :param extra_tensor_name: Name of the auxiliary tensor
   :return: The requested tensor view
   :note: This function is designed for quantized Tensor. If one Tensor is quantized to int8 using per tensor quantization method, you can use this_tensor.getExtraTensorViewInTensor("scale").item<float>(); to get a `scale` tensor from this tensor.

Utility Functions
-----------------

isNil
~~~~~

.. cpp:function:: bool Tensor::isNil() const

   Check if this tensor is not initialized.

   :return: true if tensor is nil, false otherwise

nil
~~~

.. cpp:function:: static Tensor Tensor::nil()

   Create a nil tensor.

   :return: Nil tensor

operator bool
~~~~~~~~~~~~~

.. cpp:function:: explicit Tensor::operator bool() const noexcept

   Check if this tensor is initialized.

   :return: true if tensor is initialized, false otherwise

delete_
~~~~~~~

.. cpp:function:: void Tensor::delete_() noexcept

   Delete tensor resources.

operator delete
~~~~~~~~~~~~~~~

.. cpp:function:: void Tensor::operator delete(void* ptr) noexcept

   Custom delete operator.

ptr
~~~

.. cpp:function:: template<typename T> T* Tensor::ptr() const

   Gets base pointer of tensor data.

   :return: Typed base pointer

offsettedPtr
~~~~~~~~~~~~

.. cpp:function:: template<typename T> T* Tensor::offsettedPtr(const std::vector<int32_t>& offsets)

   Typed pointer access with offset.

   :param offsets: Multi-dimensional indices
   :return: Typed pointer to the element

coffsettedPtr
~~~~~~~~~~~~~

.. cpp:function:: template<typename T> T* Tensor::coffsettedPtr(const std::vector<int32_t>& offsets) const

   Typed pointer access with offset (const version).

   :param offsets: Multi-dimensional indices
   :return: Typed pointer to the element

ptrAt
~~~~~

.. cpp:function:: template<typename T> T* Tensor::ptrAt(const std::vector<int32_t>& offsets)

   Typed pointer access with offset.

   :param offsets: Multi-dimensional indices
   :return: Typed pointer to the element

cptrAt
~~~~~~

.. cpp:function:: template<typename T> const T* Tensor::cptrAt(const std::vector<int32_t>& offsets) const

   Typed pointer access with offset (const version).

   :param offsets: Multi-dimensional indices
   :return: Typed pointer to the element

at
~~

.. cpp:function:: template<typename T> T& Tensor::at(const std::vector<int32_t>& offsets)

   Accesses a tensor element at specified coordinates.

   :param offsets: Multi-dimensional indices
   :return: Reference to the element

constAt
~~~~~~~~~~

.. cpp:function:: template<typename T> const T& Tensor::constAt(const std::vector<int32_t>& offsets) const

   Accesses a tensor element at specified coordinates (const version).

   :param offsets: Multi-dimensional indices
   :return: Const reference to the element
