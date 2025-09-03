# Copyright (c) MLLM Team.
# Licensed under the MIT License.

from .._C import Tensor, CXXLayer, LayerImpl, OpTypes, LinearImplTypes, LinearOpOptions
from .._C import (
    RMSNormOpOptions,
    SiLUOpOptions,
    EmbeddingOpOptions,
    GELUOpOptions,
    LayerNormOpOptions,
    SoftmaxOpOptions,
    CausalMaskOpOptions,
    KVCacheOpOptions,
)


class Layer:
    def __init__(self, op_type: OpTypes, options):
        self.__cxx_impl = LayerImpl(op_type, options)
        self.__cxx_layer = CXXLayer(self.__cxx_impl)

    def set_name(self, name):
        """
        Set the name of the layer
        """
        self.__cxx_impl.set_name(name)

    def set_absolute_name(self, name):
        """
        Set the absolute name of the layer
        """
        self.__cxx_impl.set_absolute_name(name)

    def cxx_impl(self):
        """
        Get the underlying C++ implementation
        """
        return self.__cxx_impl

    def forward(self, *args):
        for arg in args:
            if not isinstance(arg, Tensor):
                raise TypeError("All arguments must be C++ Tensor objects")
        return self.__cxx_layer.forward(list(args))

    def __call__(self, *args, **kwargs):
        """
        Call the forward method
        """
        return self.forward(*args, **kwargs)

    def __repr__(self):
        """
        Return a string representation of the Layer.
        """
        return f"{self.__class__.__name__}()"


class Linear(Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        bias=True,
        impl_type: LinearImplTypes = LinearImplTypes.Default,
    ):
        super().__init__(
            OpTypes.Linear, LinearOpOptions(in_channels, out_channels, bias, impl_type)
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias
        self.impl_type = impl_type

    def forward(self, *args):
        return super().forward(*args)[0]

    def __repr__(self):
        """
        Return a string representation of the Layer.
        """
        return f"{self.__class__.__name__}(in_channels={self.in_channels}, out_channels={self.out_channels}, bias={self.bias}, impl_type={self.impl_type})"


class RMSNorm(Layer):
    def __init__(self, epsilon=1e-5, add_unit_offset=False):
        options = RMSNormOpOptions()
        options.epsilon = epsilon
        options.add_unit_offset = add_unit_offset
        super().__init__(OpTypes.RMSNorm, options)
        self.epsilon = epsilon
        self.add_unit_offset = add_unit_offset

    def forward(self, *args):
        return super().forward(*args)[0]

    def __repr__(self):
        return f"{self.__class__.__name__}(epsilon={self.epsilon}, add_unit_offset={self.add_unit_offset})"


class SiLU(Layer):
    def __init__(self):
        super().__init__(OpTypes.SiLU, SiLUOpOptions())

    def forward(self, *args):
        return super().forward(*args)[0]

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class Embedding(Layer):
    def __init__(self, vocab_size, hidden_size):
        options = EmbeddingOpOptions()
        options.vocab_size = vocab_size
        options.hidden_size = hidden_size
        super().__init__(OpTypes.Embedding, options)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

    def forward(self, *args):
        return super().forward(*args)[0]

    def __repr__(self):
        return f"{self.__class__.__name__}(vocab_size={self.vocab_size}, hidden_size={self.hidden_size})"


class GELU(Layer):
    def __init__(self):
        super().__init__(OpTypes.GELU, GELUOpOptions())

    def forward(self, *args):
        return super().forward(*args)[0]

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class LayerNorm(Layer):
    def __init__(self, normalized_shape, elementwise_affine=True, bias=True, eps=1e-6):
        options = LayerNormOpOptions()
        options.normalized_shape = normalized_shape
        options.elementwise_affine = elementwise_affine
        options.bias = bias
        options.eps = eps
        super().__init__(OpTypes.LayerNorm, options)
        self.normalized_shape = normalized_shape
        self.elementwise_affine = elementwise_affine
        self.bias = bias
        self.eps = eps

    def forward(self, *args):
        return super().forward(*args)[0]

    def __repr__(self):
        return f"{self.__class__.__name__}(normalized_shape={self.normalized_shape}, elementwise_affine={self.elementwise_affine}, bias={self.bias}, eps={self.eps})"


class Softmax(Layer):
    def __init__(self, dim):
        options = SoftmaxOpOptions()
        options.axis = dim
        super().__init__(OpTypes.Softmax, options)
        self.dim = dim

    def forward(self, *args):
        return super().forward(*args)[0]

    def __repr__(self):
        return f"{self.__class__.__name__}(dim={self.dim})"


class CausalMask(Layer):
    def __init__(self, sliding_window=False, window_size=0):
        options = CausalMaskOpOptions()
        options.sliding_window = sliding_window
        options.window_size = window_size
        super().__init__(OpTypes.CausalMask, options)
        self.sliding_window = sliding_window
        self.window_size = window_size

    def forward(self, *args):
        return super().forward(*args)[0]

    def __repr__(self):
        return f"{self.__class__.__name__}(sliding_window={self.sliding_window}, window_size={self.window_size})"


class KVCache(Layer):
    def __init__(self, layer_idx, q_head, kv_head, head_dim, use_fa2=True):
        options = KVCacheOpOptions()
        options.layer_idx = layer_idx
        options.q_head = q_head
        options.kv_head = kv_head
        options.head_dim = head_dim
        options.use_fa2 = use_fa2
        super().__init__(OpTypes.KVCache, options)
        self.layer_idx = layer_idx
        self.q_head = q_head
        self.kv_head = kv_head
        self.head_dim = head_dim
        self.use_fa2 = use_fa2

    def forward(self, *args):
        return super().forward(*args)

    def __repr__(self):
        return f"{self.__class__.__name__}(layer_idx={self.layer_idx}, q_head={self.q_head}, kv_head={self.kv_head}, head_dim={self.head_dim}, use_fa2={self.use_fa2})"
