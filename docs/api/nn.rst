Neural Network Layers API
=========================

The layers directory contains implementations of various neural network layers that can be used to build models in MLLM. These layers are the building blocks for constructing neural networks.

.. code-block:: cpp

   #include "mllm/nn/Nn.hpp"

Linear Layer
------------

.. cpp:class:: Linear

   A fully connected linear layer that applies a linear transformation to the input data.

   .. cpp:function:: Linear::Linear()

      Default constructor.

   .. cpp:function:: Linear::Linear(int32_t in_channels, int32_t out_channels, bool bias = true, aops::LinearImplTypes impl_type = aops::LinearImplTypes::kDefault)

      Constructor with layer parameters.

      :param in_channels: Number of input features
      :param out_channels: Number of output features
      :param bias: Whether to include a bias term (default: true)
      :param impl_type: Implementation type (default: kDefault)

   .. cpp:function:: Linear::Linear(const aops::LinearOpOptions& options)

      Constructor with options.

      :param options: Linear operation options

   .. cpp:function:: Tensor Linear::weight() const

      Get the weight tensor of the layer.

      :return: Weight tensor

   .. cpp:function:: Tensor Linear::bias() const

      Get the bias tensor of the layer.

      :return: Bias tensor

RMSNorm Layer
-------------

.. cpp:class:: RMSNorm

   Root Mean Square Layer Normalization.

   .. cpp:function:: RMSNorm::RMSNorm()

      Default constructor with epsilon=1e-5 and add_unit_offset=false.

   .. cpp:function:: RMSNorm::RMSNorm(float epsilon, bool add_unit_offset = false)

      Constructor with normalization parameters.

      :param epsilon: Small value added to the denominator for numerical stability (default: 1e-5)
      :param add_unit_offset: Whether to add a unit offset (default: false)

   .. cpp:function:: RMSNorm::RMSNorm(const aops::RMSNormOpOptions& options)

      Constructor with options.

      :param options: RMSNorm operation options

   .. cpp:function:: Tensor RMSNorm::weight() const

      Get the weight tensor of the layer.

      :return: Weight tensor

SiLU Layer
----------

.. cpp:class:: SiLU

   Sigmoid Linear Unit activation function (also known as Swish).

   .. cpp:function:: SiLU::SiLU()

      Default constructor.

   .. cpp:function:: SiLU::SiLU(const aops::SiLUOpOptions& options)

      Constructor with options.

      :param options: SiLU operation options

Embedding Layer
---------------

.. cpp:class:: Embedding

   Embedding layer that maps indices to dense vectors.

   .. cpp:function:: Embedding::Embedding()

      Default constructor.

   .. cpp:function:: Embedding::Embedding(const aops::EmbeddingOpOptions& options)

      Constructor with options.

      :param options: Embedding operation options

   .. cpp:function:: Embedding::Embedding(int32_t vocab_size, int32_t hidden_size)

      Constructor with vocabulary and hidden size.

      :param vocab_size: Size of the vocabulary
      :param hidden_size: Dimension of each embedding vector

   .. cpp:function:: Tensor Embedding::weight() const

      Get the embedding weight matrix.

      :return: Weight tensor of shape [vocab_size, hidden_size]

GELU Layer
----------

.. cpp:class:: GELU

   Gaussian Error Linear Unit activation function.

   .. cpp:function:: GELU::GELU()

      Default constructor.

   .. cpp:function:: GELU::GELU(const aops::GELUOpOptions& options)

      Constructor with options.

      :param options: GELU operation options

QuickGELU Layer
---------------

.. cpp:class:: QuickGELU

   An approximation of GELU that is faster to compute.

   .. cpp:function:: QuickGELU::QuickGELU()

      Default constructor.

   .. cpp:function:: QuickGELU::QuickGELU(const aops::QuickGELUOpOptions& options)

      Constructor with options.

      :param options: QuickGELU operation options

ReLU Layer
----------

.. cpp:class:: ReLU

   Rectified Linear Unit activation function.

   .. cpp:function:: ReLU::ReLU()

      Default constructor.

   .. cpp:function:: ReLU::ReLU(const aops::ReLUOpOptions& options)

      Constructor with options.

      :param options: ReLU operation options

LayerNorm Layer
---------------

.. cpp:class:: LayerNorm

   Layer Normalization.

   .. cpp:function:: LayerNorm::LayerNorm()

      Default constructor.

   .. cpp:function:: LayerNorm::LayerNorm(const aops::LayerNormOpOptions& options)

      Constructor with options.

      :param options: LayerNorm operation options

   .. cpp:function:: LayerNorm::LayerNorm(const std::vector<int32_t>& normalized_shape, bool elementwise_affine = true, bool bias = true, float eps = 1e-6)

      Constructor with normalization parameters.

      :param normalized_shape: Shape of the normalized dimensions
      :param elementwise_affine: Whether to use learnable affine parameters (default: true)
      :param bias: Whether to include bias term (default: true)
      :param eps: Small value added to the denominator for numerical stability (default: 1e-6)

Softmax Layer
-------------

.. cpp:class:: Softmax

   Softmax activation function.

   .. cpp:function:: Softmax::Softmax()

      Default constructor.

   .. cpp:function:: Softmax::Softmax(const aops::SoftmaxOpOptions& options)

      Constructor with options.

      :param options: Softmax operation options

   .. cpp:function:: Softmax::Softmax(int32_t dim)

      Constructor with dimension parameter.

      :param dim: Dimension along which to apply softmax

VisionRoPE Layer
----------------

.. cpp:class:: VisionRoPE

   Rotary Positional Encoding for vision tasks.

   .. cpp:function:: VisionRoPE::VisionRoPE()

      Default constructor.

   .. cpp:function:: VisionRoPE::VisionRoPE(const aops::VisionRoPEOpOptions& Options)

      Constructor with options.

      :param Options: VisionRoPE operation options

   .. cpp:function:: VisionRoPE::VisionRoPE(const aops::VisionRoPEOpOptionsType type, const aops::Qwen2VLRoPEOpOptions& Options)

      Constructor with type and Qwen2VL options.

      :param type: Type of VisionRoPE operation
      :param Options: Qwen2VL RoPE operation options

Conv3D Layer
------------

.. cpp:class:: Conv3D

   3D Convolutional layer.

   .. cpp:function:: Conv3D::Conv3D()

      Default constructor.

   .. cpp:function:: Conv3D::Conv3D(int32_t in_channels, int32_t out_channels, const std::vector<int32_t>& kernel_size, const std::vector<int32_t>& stride_size, bool bias = true, aops::Conv3DOpImplType impl_type = aops::Conv3DOpImplType::kDefault)

      Constructor with convolution parameters.

      :param in_channels: Number of input channels
      :param out_channels: Number of output channels
      :param kernel_size: Size of the convolution kernel
      :param stride_size: Stride of the convolution
      :param bias: Whether to include a bias term (default: true)
      :param impl_type: Implementation type (default: kDefault)

   .. cpp:function:: Conv3D::Conv3D(const aops::Conv3DOpOptions& options)

      Constructor with options.

      :param options: Conv3D operation options

   .. cpp:function:: Tensor Conv3D::weight() const

      Get the weight tensor of the layer.

      :return: Weight tensor

   .. cpp:function:: Tensor Conv3D::bias() const

      Get the bias tensor of the layer.

      :return: Bias tensor

CausalMask Layer
----------------

.. cpp:class:: CausalMask

   Causal (autoregressive) attention mask.

   .. cpp:function:: CausalMask::CausalMask()

      Default constructor.

   .. cpp:function:: CausalMask::CausalMask(const aops::CausalMaskOpOptions& options)

      Constructor with options.

      :param options: CausalMask operation options

   .. cpp:function:: CausalMask::CausalMask(bool sliding_window, int32_t window_size)

      Constructor with sliding window parameters.

      :param sliding_window: Whether to use sliding window attention
      :param window_size: Size of the sliding window

MultimodalRoPE Layer
--------------------

.. cpp:class:: MultimodalRoPE

   Rotary Positional Encoding for multimodal tasks.

   .. cpp:function:: MultimodalRoPE::MultimodalRoPE()

      Default constructor.

   .. cpp:function:: MultimodalRoPE::MultimodalRoPE(const aops::MultimodalRoPEOpOptions& options)

      Constructor with options.

      :param options: MultimodalRoPE operation options

   .. cpp:function:: MultimodalRoPE::MultimodalRoPE(const aops::Qwen2VLMultimodalRoPEOpOptions& options)

      Constructor with Qwen2VL multimodal options.

      :param options: Qwen2VL MultimodalRoPE operation options

Param Layer
-----------

.. cpp:class:: Param

   Parameter layer that holds trainable parameters.

   .. cpp:function:: Param::Param()

      Default constructor.

   .. cpp:function:: Param::Param(const aops::ParamOpOptions& options)

      Constructor with options.

      :param options: Param operation options

   .. cpp:function:: Param::Param(const std::string& name, const Tensor::shape_t& shape = {})

      Constructor with name and shape.

      :param name: Name of the parameter
      :param shape: Shape of the parameter tensor (default: empty)

   .. cpp:function:: Tensor Param::weight() const

      Get the parameter tensor.

      :return: Weight tensor

KVCache Layer
-------------

.. cpp:class:: KVCache

   Key-Value cache for autoregressive generation.

   .. cpp:function:: KVCache::KVCache()

      Default constructor.

   .. cpp:function:: KVCache::KVCache(const aops::KVCacheOpOptions& options)

      Constructor with options.

      :param options: KVCache operation options

   .. cpp:function:: KVCache::KVCache(int32_t layer_idx, int32_t q_head, int32_t kv_head, int32_t head_dim, bool use_fa2 = true)

      Constructor with cache parameters.

      :param layer_idx: Layer index
      :param q_head: Number of query heads
      :param kv_head: Number of key/value heads
      :param head_dim: Dimension of each head
      :param use_fa2: Whether to use FlashAttention-2 (default: true)

   .. cpp:function:: void KVCache::setLayerIndex(int32_t layer_idx)

      Set the layer index.

      :param layer_idx: Layer index

STFT Layer
----------

.. cpp:class:: STFT

   Short-Time Fourier Transform layer for signal processing.

   .. cpp:function:: STFT::STFT()

      Default constructor.

   .. cpp:function:: STFT::STFT(const aops::STFTOpOptions& options)

      Constructor with options.

      :param options: STFT operation options

   .. cpp:function:: STFT::STFT(int n_fft, int hop_length, int win_length, bool onesided = true, bool center = false, const std::string& pad_mode = "constant", bool return_complex = false)

      Constructor with STFT parameters.

      :param n_fft: Size of Fourier transform
      :param hop_length: Distance between neighboring sliding window frames
      :param win_length: Size of window frame
      :param onesided: Whether to return only non-negative frequency bins (default: true)
      :param center: Whether to pad input on both sides (default: false)
      :param pad_mode: Padding mode (default: "constant")
      :param return_complex: Whether to return complex tensor (default: false)