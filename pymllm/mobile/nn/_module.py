# Copyright (c) MLLM Team.
# Licensed under the MIT License.

from .. import ffi
from ._layers import _Layer


class Module:
    def __init__(self, name: str = "model"):
        super().__setattr__("this_module_name", name)
        super().__setattr__("absolute_name", name)
        super().__setattr__("module_layer_list", {})
        super().__setattr__("_is_initializing", True)
        self.device: ffi.Device = ffi.cpu_()

    def __setattr__(self, name: str, value):
        super().__setattr__(name, value)
        if (
            getattr(self, "_is_initializing", False)
            and not name.startswith("_")
            and isinstance(value, (Module, _Layer))
        ):
            value.this_module_name = name
            value.absolute_name = f"{self.absolute_name}.{name}"
            self.module_layer_list[name] = value
            if isinstance(value, Module):
                value._is_initializing = True

    def to(self, x):
        if isinstance(x, str):
            if x == "qnn":
                pass
            self.device = ffi.device(x)
        elif isinstance(x, ffi.Device):
            self.device = x
        elif isinstance(x, ffi.DType):
            raise NotImplementedError("Module.to(DType) is not supported")
        else:
            raise TypeError("device must be str or Device, but got {}".format(type(x)))
        return self

    def re_naming_finish_initialization(self):
        self._is_initializing = False

    def load(self, pf: ffi.ParameterFile):
        for module_layer in self.module_layer_list.values():
            if isinstance(module_layer, Module, _Layer):
                module_layer.load(pf)
            else:
                raise TypeError(
                    "Module layer must be Module or _Layer, but got {}".format(
                        type(module_layer)
                    )
                )

    def trace(self, *args):
        pass

    def forward(self, *args):
        pass

    def __call__(self, *args, **kwds):
        # __send_graph_begin()
        if kwds.get("__mllm_trace_mode_enabled", False):
            return self.trace(*args, **kwds)
        return self.forward(*args, **kwds)
        # __send_graph_end()

    def __str__(self):
        return self._repr_helper()

    def __repr__(self):
        return self._repr_helper()

    def _repr_helper(self, indent_level: int = 0) -> str:
        indent = "  " * indent_level
        next_indent = "  " * (indent_level + 1)
        module_str = f"{self.__class__.__name__}("
        if self.module_layer_list:
            child_lines = []
            for name, child in self.module_layer_list.items():
                if isinstance(child, Module):
                    child_repr = child._repr_helper(indent_level + 1)
                    child_lines.append(f"{next_indent}({name}): {child_repr}")
                elif isinstance(child, _Layer):
                    child_lines.append(f"{next_indent}({name}): {repr(child)}")
                else:
                    child_lines.append(
                        f"{next_indent}({name}): {type(child).__name__}()"
                    )
            module_str += "\n" + "\n".join(child_lines) + "\n" + indent + ")"
        else:
            module_str += ")"

        return module_str
