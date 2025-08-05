# Copyright (c) MLLM Team.
# Licensed under the MIT License.

from .._C import LayerImpl, OpTypes, LinearImplTypes, LinearOpOptions


class Layer:
    def __init__(self, op_type: OpTypes, options):
        self.__cxx_impl = LayerImpl(op_type, options)

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
        """
        Forward pass of the layer
        """
        raise NotImplementedError("forward method must be implemented in subclass")

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
        # TODO: Impl details
        return super().forward(*args)

    def __repr__(self):
        """
        Return a string representation of the Layer.
        """
        return f"{self.__class__.__name__}(in_channels={self.in_channels}, out_channels={self.out_channels}, bias={self.bias}, impl_type={self.impl_type})"
