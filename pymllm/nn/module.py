# Copyright (c) MLLM Team.
# Licensed under the MIT License.

from .._C import ModuleImpl, Context
from .layer import Layer


class Module:
    def __init__(self):
        self.__cxx_impl = ModuleImpl()
        object.__setattr__(self, "_layers", {})
        object.__setattr__(self, "_modules", {})

        # Set self name as the class name by default
        self.set_name("<top_module>")
        self.set_absolute_name("<top_module>")

    def __setattr__(self, name, value):
        """
        The __setattr__ method is used to intercept attribute assignments.
        We will register module and layers here.
        """
        if isinstance(value, Layer):
            value.set_name(name)
            if self.__cxx_impl.get_absolute_name() == "<top_module>":
                value.set_absolute_name(name)
            else:
                value.set_absolute_name(
                    self.__cxx_impl.get_absolute_name() + "." + name
                )
            self._layers[name] = value
            self.__cxx_impl.reg_child_node(value.cxx_impl())
            _op = (
                Context.instance()
                .get_backend(value.cxx_impl().get_device())
                .create_op(value.cxx_impl().op_type(), value.cxx_impl().ref_options())
            )
            _op.set_name(value.cxx_impl().get_absolute_name())
            value.cxx_impl().set_instanced_op(_op)
        if isinstance(value, Module):
            value.set_name(name)
            if self.__cxx_impl.get_absolute_name() == "<top_module>":
                value.set_absolute_name(name)
            else:
                value.set_absolute_name(
                    self.__cxx_impl.get_absolute_name() + "." + name
                )
            self._modules[name] = value
            self.__cxx_impl.reg_child_node(value.cxx_impl())
        object.__setattr__(self, name, value)

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

    def named_layers(self):
        return self._layers.items()

    def cxx_impl(self):
        """
        Get the underlying C++ implementation
        """
        return self.__cxx_impl

    def load(self, param_file):
        """
        Load parameters from a ParameterFile
        """
        self.__cxx_impl.load(param_file)

    def to(self, device_type):
        """
        Move the module to specified device
        """
        self.__cxx_impl.to(device_type)

    def params(self, version):
        """
        Get parameters of the module
        """
        return self.__cxx_impl.params(version)

    def forward(self, *args):
        """
        Forward pass of the layer
        """
        raise NotImplementedError("forward method must be implemented in subclass")

    def __call__(self, *args, **kwargs):
        """
        Call the forward method
        """
        # TODO Dispatch Graph Begin and End Op
        return self.forward(*args, **kwargs)

    def __repr__(self, indent=0):
        """
        Pretty print the module structure.
        """
        s = " " * indent + self.__class__.__name__ + "(\n"
        for name, module in self._modules.items():
            s += " " * (indent + 2) + f"({name}): {module.__repr__(indent + 2)}"
        for name, layer in self._layers.items():
            s += " " * (indent + 2) + f"({name}): {layer.__repr__()}\n"
        s += " " * indent + ")\n"
        return s
